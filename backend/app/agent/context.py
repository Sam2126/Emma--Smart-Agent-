"""
Per-run context for the agent engine.

LangGraph state should hold data, not live objects. Everything a run needs that
is not plain data — the status callback, the tool set, the deadline, background
learning tasks — lives on a RunContext passed through the graph config as
config["configurable"]["run"].
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import structlog

logger = structlog.get_logger(__name__)

StatusCallback = Callable[[dict[str, Any]], None]

# Keep references to fire-and-forget learning tasks so they are not garbage
# collected mid-flight, and so shutdown can wait for them briefly.
_background_tasks: set[asyncio.Task] = set()


@dataclass
class RunContext:
    status_callback: StatusCallback | None = None
    started_at: float = field(default_factory=time.monotonic)
    deadline: float = 0.0
    toolset: dict[str, Any] = field(default_factory=dict)
    tool_schemas: list[dict[str, Any]] = field(default_factory=list)
    browser_tools_deferred: bool = False

    def notify(self, status: str, step: str, progress: float, details: str = "") -> None:
        if not self.status_callback:
            return
        try:
            self.status_callback({
                "status": status,
                "current_step": step,
                "progress": max(0.0, min(1.0, progress)),
                "details": details,
            })
        except Exception as e:
            logger.warning("status_callback_failed", error=str(e)[:200])

    def seconds_left(self) -> float:
        return self.deadline - time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at


def run_context(config: dict[str, Any] | None) -> RunContext:
    configurable = (config or {}).get("configurable") or {}
    run = configurable.get("run")
    if run is None:
        # Nodes invoked directly (tests) still get a usable context.
        run = RunContext(deadline=time.monotonic() + 180)
        configurable["run"] = run
    return run


def schedule_background(coro, name: str) -> asyncio.Task:
    task = asyncio.get_running_loop().create_task(coro, name=name)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def drain_background_tasks(timeout: float = 10.0) -> None:
    """Give in-flight learning a chance to finish (used at shutdown)."""
    pending = [t for t in _background_tasks if not t.done()]
    if pending:
        await asyncio.wait(pending, timeout=timeout)


# Tasks currently inside AgentRunner.run_task, from every entry point.
# Found 2026-09-15: Ctrl+C shut the server down while a wake-word task kept
# running; it re-opened the database after shutdown and only then finished.
_active_runs: set[asyncio.Task] = set()


def track_run() -> asyncio.Task | None:
    task = asyncio.current_task()
    if task is not None:
        _active_runs.add(task)
    return task


def active_run_count() -> int:
    return sum(1 for task in _active_runs if not task.done())


def untrack_run(task: asyncio.Task | None) -> None:
    if task is not None:
        _active_runs.discard(task)


async def cancel_active_runs(timeout: float = 8.0) -> int:
    """Cancel every running task and wait for their cleanup (used at shutdown)."""
    current = asyncio.current_task()
    running = [t for t in _active_runs if not t.done() and t is not current]
    for task in running:
        task.cancel()
    if running:
        await asyncio.wait(running, timeout=timeout)
    return len(running)
