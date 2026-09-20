"""
FastAPI application entrypoint.

Starts the FastAPI REST API and the WebSocket server together.
On startup: initializes the database, registers the browser controller,
validates the model routing against the providers, warms semantic memory and
starts the wake-word listener. On shutdown: lets in-flight learning finish,
then disconnects everything.

Every task — extension, Local Agent UI, voice, wake word, REST — runs on the
single LangGraph engine in app/agent (AgentRunner).

Key rotation: reads GROQ_API_KEYS (comma-separated) from .env; the LiteLLM
patch hot-swaps keys on 429 and falls back to Gemini when every key is
exhausted (GEMINI_API_KEY).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager

# Force UTF-8 on Windows console for emoji/unicode compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from app.utils.litellm_patch import apply_litellm_patch

# Key rotation + payload sanitisation + tool guards + Gemini fallback — one
# shared implementation (also used by app/evals/run_eval.py).
apply_litellm_patch()

# =============================================================================
# App setup
# =============================================================================

import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from app.browser.controller import BrowserController
from app.config import get_settings
from app.state.database import close_database, init_database
from app.state.episodic_log import EpisodicLogger
from app.utils.loop import set_main_loop
from app.websocket.server import WebSocketServer

logger = structlog.get_logger(__name__)

# Shared instances
main_loop: asyncio.AbstractEventLoop | None = None
browser_controller = BrowserController()
ws_server = WebSocketServer()
episodic_logger = EpisodicLogger()

# Serializes lazy browser launches so two clients can't race two windows
_browser_connect_lock = asyncio.Lock()


class _QuietHealthChecks(logging.Filter):
    """The desktop app checks /health every 15 s; keep those requests out of the console."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "GET /health" not in record.getMessage()


_HEALTH_LOG_FILTER = _QuietHealthChecks()


async def ensure_browser_connection() -> bool:
    """
    Connect the browser on demand (called when the extension / Local Agent
    actually opens, or right before a task needs the page). Attaches to a
    running debug-port Chrome when possible, launches the agent window only
    when there is none. Startup itself never calls this.
    """
    if browser_controller.is_connected:
        return True
    async with _browser_connect_lock:
        if browser_controller.is_connected:
            return True
        try:
            await browser_controller.connect(get_settings().cdp_endpoint)
            return browser_controller.is_connected
        except Exception as e:
            logger.error("ensure_browser_failed", error=str(e)[:200])
            return False


async def _startup_model_routing() -> None:
    try:
        from app.utils.llm import refresh_model_availability

        await refresh_model_availability()
    except Exception as e:
        logger.warning("model_routing_check_failed", error=str(e)[:200])


async def _startup_warm_semantic_memory() -> None:
    try:
        from app.state.semantic_memory import get_semantic_memory

        ok = await asyncio.to_thread(get_semantic_memory().warm)
        logger.info("semantic_memory_warmed", ok=ok)
    except Exception as e:
        logger.warning("semantic_memory_warm_failed", error=str(e)[:200])


# Set by the desktop app for the backend it starts (desktop/agent_desktop.pyw).
PARENT_PID_ENV = "AGENT_PARENT_PID"
_PARENT_POLL_SECONDS = 2.0


async def watch_parent_app(pid: int) -> bool:
    """
    Stop this backend when the desktop app that started it has gone.

    Found 2026-09-17: a backend outlived its app and kept listening for the
    wake word, so talk in the room started tasks after the user had stopped the
    project. Returns True when it asked for the shutdown.
    """
    import ctypes
    import signal

    synchronize, wait_timeout, invalid_parameter = 0x00100000, 0x102, 87
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel32.OpenProcess(synchronize, False, pid)
    if not handle:
        if ctypes.get_last_error() != invalid_parameter:
            logger.warning("parent_app_not_watchable", pid=pid)
            return False
    else:
        try:
            while kernel32.WaitForSingleObject(handle, 0) == wait_timeout:
                await asyncio.sleep(_PARENT_POLL_SECONDS)
        finally:
            kernel32.CloseHandle(handle)
    logger.warning("parent_app_gone_shutting_down", pid=pid)
    # Same clean path as Ctrl+C: wake word stopped, tasks cancelled, Chrome closed.
    signal.raise_signal(signal.SIGINT)
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup:
      1. Initialize database
      2. Register the browser controller (attach-only; never auto-launches)
      3. Start the WebSocket server
      4. In the background: validate model routing, warm semantic memory
      5. Start the hands-free wake word listener (if enabled)

    Shutdown:
      1. Stop the wake listener and WebSocket server
      2. Give in-flight background learning a few seconds to finish
      3. Disconnect the browser, close the database
    """
    global main_loop
    main_loop = asyncio.get_running_loop()
    set_main_loop(main_loop)
    settings = get_settings()
    # Added here, after uvicorn has configured its loggers.
    access_log = logging.getLogger("uvicorn.access")
    if _HEALTH_LOG_FILTER not in access_log.filters:
        access_log.addFilter(_HEALTH_LOG_FILTER)

    logger.info("app_starting")

    await init_database()
    logger.info("database_ready")

    try:
        from app.utils.key_pool import get_key_pool

        pool = get_key_pool()
        logger.info("key_pool_ready", total_keys=pool.size, statuses=pool.status())
    except Exception as ke:
        logger.warning("key_pool_unavailable", error=str(ke))

    from app.browser.registry import set_browser_controller

    set_browser_controller(browser_controller)
    try:
        # Attach-only at startup: never auto-launch a Chrome window. The
        # browser launches lazily when a task needs it.
        await browser_controller.connect(settings.cdp_endpoint, allow_launch=False)
    except ConnectionError as e:
        logger.error("browser_connection_failed", error=str(e))
        logger.warning("app_starting_without_browser", hint="Start Chrome with: chrome.exe --remote-debugging-port=9222")

    ws_task = asyncio.create_task(ws_server.start(settings.ws_host, settings.ws_port))
    background = [asyncio.create_task(_startup_model_routing())]
    if settings.semantic_memory_enabled:
        background.append(asyncio.create_task(_startup_warm_semantic_memory()))
    # The lines Emma says most often are synthesized in the background now, so
    # the first spoken reply is instant instead of waiting for a request.
    try:
        from app.tts import prewarm

        prewarm()
    except Exception as e:
        logger.debug("tts_prewarm_skipped", error=str(e)[:120])

    logger.info("app_started", fastapi_port=settings.fastapi_port, ws_port=settings.ws_port)

    parent_watch = None
    parent_pid = os.environ.get(PARENT_PID_ENV, "").strip()
    if parent_pid.isdigit():
        parent_watch = asyncio.create_task(watch_parent_app(int(parent_pid)))

    _wake_listener = None
    if settings.wake_word_enabled:
        try:
            from app.wake_listener import start_wake_listener

            _wake_listener = start_wake_listener(
                wake_word=settings.wake_word,
                stop_word=settings.wake_word_stop,
                listen_timeout=settings.wake_word_listen_timeout,
                loop=main_loop,
            )
            logger.info("wake_listener_enabled", wake_word=settings.wake_word, stop_word=settings.wake_word_stop)
        except Exception as e:
            logger.warning(
                "wake_listener_start_failed",
                error=str(e)[:200],
                hint="Install SpeechRecognition + PyAudio to enable hands-free mode",
            )

    yield

    logger.info("app_shutting_down")

    # Emma's voice stops the moment the agent does: no half-spoken sentence
    # carrying on after the user has pressed Ctrl+C.
    try:
        from app.tts import get_speaker

        get_speaker().shutdown()
    except Exception as e:
        logger.debug("tts_shutdown_failed", error=str(e)[:120])

    if _wake_listener is not None:
        try:
            from app.wake_listener import stop_wake_listener

            stop_wake_listener()
        except Exception:
            pass

    # Stop tasks that are still running (extension, Local Agent, wake word,
    # REST) while the database is still open, so they record the cancellation.
    try:
        from app.agent.context import cancel_active_runs

        cancelled = await cancel_active_runs(timeout=8.0)
        if cancelled:
            logger.info("running_tasks_cancelled_for_shutdown", count=cancelled)
    except Exception as e:
        logger.warning("cancel_running_tasks_failed", error=str(e)[:200])

    for task in background:
        task.cancel()
    if parent_watch is not None:
        parent_watch.cancel()
    ws_task.cancel()
    try:
        await asyncio.wait_for(ws_task, timeout=2.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass

    try:
        from app.agent.context import drain_background_tasks

        await drain_background_tasks(timeout=8.0)
    except Exception:
        pass

    await browser_controller.disconnect()
    await close_database()
    set_main_loop(None)
    logger.info("app_stopped")


app = FastAPI(
    title="Self-Improving Browser Agent",
    description="Universal browser and desktop automation with self-improving memory",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REST API endpoints
# =============================================================================

@app.get("/local", response_class=HTMLResponse)
async def local_agent_ui():
    """Local Agent UI — voice/text tasks with local-device and browser access."""
    from pathlib import Path as _P

    html = (_P(__file__).parent / "static" / "local_agent.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "running",
        "service": "Self-Improving Browser Agent",
        "version": "0.3.0",
        "browser_connected": browser_controller.is_connected,
    }


def _backend_code_stamp() -> float:
    """Newest mtime of this backend's own code, measured once at startup.

    /health reports it, so the code this process actually loaded is visible from
    outside. The desktop app compares it with the code on disk to decide whether
    opening the icon should restart the agent: for a backend it did not start
    itself, it has no other way to tell that the running agent is out of date.
    """
    from pathlib import Path as _P

    app_dir = _P(__file__).resolve().parent
    newest = 0.0
    for path in (*app_dir.rglob("*.py"), *(app_dir / "static").glob("*.html")):
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            pass
    return newest


_CODE_STAMP = _backend_code_stamp()


@app.post("/talk")
async def start_talking() -> dict:
    """Begin listening at once, without the wake word.

    The desktop app's "Talk to Emma" button posts here. Whatever is said is
    answered out loud if it was a question, or carried out if it was a job.
    """
    listener = _wake_listener
    if listener is None or not listener.is_running():
        return {
            "ok": False,
            "detail": "The wake word listener is not running, so there is no microphone to listen with.",
        }
    started = listener.request_conversation()
    return {"ok": bool(started), "detail": "Listening now." if started else "The listener is stopping."}


@app.get("/health")
async def health():
    """Detailed health check."""
    try:
        from app.utils.key_pool import get_key_pool

        key_pool_status = get_key_pool().status()
    except Exception:
        key_pool_status = []

    try:
        from app.wake_listener import get_wake_listener_status

        wake = get_wake_listener_status()
    except Exception:
        wake = {"running": False}
    from app.agent.context import active_run_count

    return {
        "status": "healthy",
        "browser_connected": browser_controller.is_connected,
        "database": "connected",
        "key_pool": key_pool_status,
        "wake_word_enabled": get_settings().wake_word_enabled,
        "wake_listener": wake,
        "task_running": active_run_count() > 0,
        "code_stamp": _CODE_STAMP,
    }


@app.post("/app/shutdown", include_in_schema=False)
async def app_shutdown(request: Request):
    """Stop the backend cleanly: the desktop app's Quit, start_agent.ps1's Ctrl+C. Local callers only."""
    import signal

    host = request.client.host if request.client else ""
    if host not in ("127.0.0.1", "::1", "localhost"):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    logger.info("shutdown_requested")
    # Same path as Ctrl+C: uvicorn's signal handler runs the normal shutdown
    # (running tasks cancelled and recorded, learning drained, database closed).
    asyncio.get_running_loop().call_later(0.3, signal.raise_signal, signal.SIGINT)
    return {"status": "shutting_down"}


@app.get("/tasks")
async def get_tasks(limit: int = 20):
    """Get recent task history."""
    history = await episodic_logger.get_task_history(limit=limit)
    return {"tasks": history}


@app.post("/task")
async def submit_task(instruction: str, scope: str = "browser"):
    """
    Submit a task via REST (alternative to the WebSocket UIs).

    Runs on the same LangGraph engine. There is no confirmation channel here,
    so irreversible actions (checkout, payment) are refused.
    """
    from app.agent.runner import AgentRunner

    task_id = str(uuid.uuid4())
    if scope == "browser":
        await ensure_browser_connection()
    try:
        result = await AgentRunner().run_task(instruction=instruction, task_id=task_id, scope=scope)
        result.pop("trajectory", None)
        return result
    except Exception as e:
        return {"task_id": task_id, "success": False, "error": str(e)}


@app.post("/tasks/{task_id}/feedback")
async def task_feedback(task_id: str, rating: int, comment: str = ""):
    """Rate a finished task: rating=1 (👍) or rating=-1 (👎)."""
    if rating not in (1, -1):
        return {"error": "rating must be 1 or -1"}
    from app.agent.services import brain

    return await brain.record_feedback(task_id, rating, comment)


# =============================================================================
# Brain Dashboard endpoints
# =============================================================================

# The engine's own instance (app/agent/services.py), not a second BrainMemory.
from app.agent.services import brain as _brain


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Return a minimal transparent favicon to silence browser 404 log spam."""
    from fastapi.responses import Response

    ico = (
        b"\x00\x00\x01\x00\x01\x00\x01\x01\x00\x00\x01\x00\x18\x00"
        b"\x30\x00\x00\x00\x16\x00\x00\x00\x28\x00\x00\x00\x01\x00"
        b"\x00\x00\x02\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x4f\x8c\xff\x00\x00\x00"
    )
    return Response(content=ico, media_type="image/x-icon")


@app.get("/brain/stats")
async def brain_stats():
    """Get global brain evolution metrics."""
    stats = await _brain.get_brain_stats()
    try:
        from app.state.semantic_memory import get_semantic_memory

        stats["experiences_stored"] = await asyncio.to_thread(get_semantic_memory().count)
    except Exception:
        pass
    return stats


@app.get("/brain/skills")
async def brain_skills(limit: int = 50):
    """Get all learned reusable skills."""
    skills = await _brain.get_all_skills(limit=limit)
    return {"skills": skills, "count": len(skills)}


@app.get("/brain/failures")
async def brain_failures(limit: int = 50):
    """Get known failure patterns and their learned solutions."""
    patterns = await _brain.get_all_failure_patterns(limit=limit)
    return {"failure_patterns": patterns, "count": len(patterns)}


@app.get("/brain/experiences")
async def brain_experiences(query: str, k: int = 5):
    """Semantic recall: the stored experiences most similar to `query`."""
    experiences = await _brain.recall_semantic(query, k=k)
    return {
        "query": query,
        "experiences": [
            {
                "task_id": e.task_id,
                "instruction": e.instruction,
                "domain": e.domain,
                "success": e.effective_success,
                "similarity": round(e.similarity, 3),
                "score": round(e.score, 3),
                "tools": e.tools,
                "lesson": e.lesson,
                "feedback": e.feedback,
                "age": e.age_text(),
            }
            for e in experiences[:k]
        ],
    }


@app.get("/brain/models")
async def model_routing():
    """Which model serves each role, and what the providers actually offer."""
    from app.utils.llm import ROLES, model_for, routing_report

    return {"routing": {role: model_for(role)[0] for role in ROLES}, "startup_check": routing_report()}


@app.get("/brain/keys")
async def key_pool_status():
    """Show current status of all API keys in the rotation pool."""
    try:
        from app.utils.key_pool import get_key_pool

        pool = get_key_pool()
        return {"total_keys": pool.size, "keys": pool.status()}
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# Main entry point
# =============================================================================

def _free_port(port: int) -> None:
    """
    Kill only the server process LISTENING on `port` so uvicorn can bind cleanly on restart.
    Never kills client connections (which would kill Chrome/browsers).
    """
    import os
    import subprocess

    try:
        result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, timeout=5)
        own_pid = os.getpid()
        for line in result.stdout.splitlines():
            line_clean = line.strip()
            if "LISTENING" in line_clean:
                parts = line_clean.split()
                if len(parts) >= 5 and parts[0].upper().startswith("TCP"):
                    local_addr = parts[1]
                    pid_str = parts[-1]
                    if (local_addr.endswith(f":{port}") or local_addr.endswith(f":[{port}]")) and pid_str.isdigit():
                        pid = int(pid_str)
                        if pid != own_pid and pid != 0:
                            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=3)
    except Exception:
        pass  # If this fails, uvicorn will give a clear port-in-use error


def _agent_backend_running(port: int, timeout: float = 5.0) -> bool:
    """True when a healthy Self-Improving Agent backend already answers on `port`."""
    import json
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{port}/health", timeout=timeout) as response:
            if response.status != 200:
                return False
            return "wake_listener" in json.loads(response.read().decode("utf-8", "replace"))
    except Exception:
        return False


# Uvicorn's own formatter rewrites each record through a copy of it to add
# colour, and in this console that left its placeholders raw:
#     INFO:     Started server process [%d]
#     INFO:     Uvicorn running on %s://%s:%d
# A plain formatter fills them from the record's arguments, like every other
# line in the log.
_UVICORN_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"plain": {"format": "%(levelname)s:     %(message)s"}},
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "plain",
            "stream": "ext://sys.stdout",
        }
    },
    "loggers": {
        "uvicorn": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["console"], "level": "INFO", "propagate": False},
    },
}


def main():
    """Run the application."""
    import logging
    import os

    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Never replace an agent that is already running. _free_port kills whatever
    # listens on the ports; found 2026-09-15, the desktop app's backend killed
    # the one running in the start_agent.ps1 console this way.
    if _agent_backend_running(settings.fastapi_port):
        print(
            f"The agent is already running on port {settings.fastapi_port} "
            "(started by the desktop app or another window). Not starting a second copy."
        )
        return

    _free_port(settings.fastapi_port)
    _free_port(settings.ws_port)

    try:
        uvicorn.run(
            "app.main:app",
            host=settings.fastapi_host,
            port=settings.fastapi_port,
            reload=False,
            log_level=settings.log_level.lower(),
            log_config=_UVICORN_LOG_CONFIG,
        )
    except KeyboardInterrupt:
        pass
    finally:
        # Force immediate process exit so lingering worker threads never hang
        os._exit(0)


if __name__ == "__main__":
    main()
