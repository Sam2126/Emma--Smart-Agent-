"""
Runtime bridge between synchronous tool entry points and the async app.

Every tool exposes a synchronous `_run` whose real work is an async body. The
agent engine calls tools through `asyncio.to_thread`, so `_run` executes in a
worker thread. From there, `run_sync` schedules the async body onto the main
event loop — the loop that owns the Playwright browser connection — and waits.
That wait is safe precisely because it happens in the worker thread: the main
loop stays free to run the coroutine.

When no main loop is registered (tests, one-off scripts), the body simply runs
in a fresh event loop.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable

import structlog

from app.utils.loop import get_main_loop

logger = structlog.get_logger(__name__)

# Upper bound for a single tool call. Individual tools enforce their own,
# shorter limits; this only has to exceed the longest of them (wait_for_login
# defaults to 180 s). The previous bridge used 60 s, which silently cut
# wait_for_login and slow app launches short.
DEFAULT_TOOL_TIMEOUT_SECONDS = 300.0


def run_sync(coro_fn: Callable[[], Any], timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS) -> Any:
    """Run an async tool body from a synchronous tool entry point."""
    main_loop = get_main_loop()
    if main_loop is not None and main_loop.is_running():
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        if current is main_loop:
            # Blocking the loop thread on a future that needs the same loop
            # would deadlock forever. Fail loudly instead.
            logger.error("tool_called_on_event_loop_thread")
            return (
                "Error executing tool: it was called on the event-loop thread. "
                "The engine must call tools via asyncio.to_thread."
            )
        try:
            future = asyncio.run_coroutine_threadsafe(coro_fn(), main_loop)
            return future.result(timeout=timeout)
        except TimeoutError:
            logger.error("tool_execution_timed_out", seconds=timeout)
            return f"Error executing tool: timed out after {int(timeout)}s."
        except Exception as e:
            logger.error("tool_execution_failed", error=str(e)[:300])
            return f"Error executing tool: {e}"

    try:
        return asyncio.run(coro_fn())
    except Exception as e:
        logger.error("tool_execution_failed", error=str(e)[:300])
        return f"Error executing tool: {e}"


def run_in_worker(coro_fn: Callable[[], Any]) -> Any:
    """Run a tool body that does BLOCKING work in the calling worker thread.

    For local and desktop tools, whose bodies call PowerShell, scan folders or
    watch a window for up to a minute. Found 2026-09-16: they used run_sync,
    which ships the body to the main loop, so the whole server froze for the
    length of the tool — /health stopped answering (the desktop app then
    reported "The agent is not running"), progress lines stopped streaming,
    and the wake word could not transcribe. Here the body gets an event loop of
    its own in this thread. Browser tools keep run_sync: Playwright objects
    belong to the main loop. A body that needs the main loop for one call (a
    vision request) awaits it through `on_main_loop`.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        in_loop = False
    else:
        in_loop = True

    if not in_loop:
        try:
            return asyncio.run(coro_fn())
        except Exception as e:
            logger.error("tool_execution_failed", error=str(e)[:300])
            return f"Error executing tool: {e}"

    # Called directly from a thread that is running a loop (a test, a script):
    # asyncio.run cannot nest, so the body gets a short-lived thread of its own.
    outcome: dict[str, Any] = {}

    def _target() -> None:
        try:
            outcome["value"] = asyncio.run(coro_fn())
        except Exception as e:  # pragma: no cover - reported below
            outcome["error"] = e

    worker = threading.Thread(target=_target, name="tool-body", daemon=True)
    worker.start()
    worker.join()
    if "error" in outcome:
        logger.error("tool_execution_failed", error=str(outcome["error"])[:300])
        return f"Error executing tool: {outcome['error']}"
    return outcome.get("value")


async def on_main_loop(coro: Any) -> Any:
    """Await `coro` on the main loop when the caller runs on another loop.

    Shared async clients (LiteLLM's HTTP connections) belong to the main loop,
    so a worker-thread tool body sends its one network call there.
    """
    main_loop = get_main_loop()
    try:
        current = asyncio.get_running_loop()
    except RuntimeError:
        current = None
    if main_loop is None or not main_loop.is_running() or current is main_loop:
        return await coro
    return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, main_loop))
