"""
Fixes from the 2026-09-15 "open gemini and search ..." run, which did the task
(typed and submitted the question on gemini.google.com) but was shown as FAILED:

  * typing / clicking in the browser counts as in-app work
  * descriptive tool output ("no popup ... or error") is not a failed step
  * negated problems in the final report are not failures
  * an unknown app name is not launched blindly; web apps go to the browser
  * a 👍 on a run marked failed turns it into a success in memory
  * the desktop app starts a new chat when it is opened again
"""

from __future__ import annotations

import asyncio
import importlib.machinery
import importlib.util
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agent.context import active_run_count, track_run, untrack_run
from app.agent.explain import build_explanation
from app.agent.toolkit import (
    instruction_needs_web,
    local_completion_shortfall,
    output_indicates_failure,
    report_indicates_failure,
)
from app.config import get_settings
from app.state.brain import BrainMemory
from app.state.reflection import ReflectionEngine
from app.state.semantic_memory import Experience
from tests.conftest import ScriptedCompletion

GEMINI_RUN = ["open_app", "navigate_browser", "see_page", "type_into_element"]


# =============================================================================
# Completion check and failure detection
# =============================================================================

def test_browser_typing_counts_as_in_app_work():
    trajectory = [{"action_type": tool} for tool in GEMINI_RUN]
    assert local_completion_shortfall("open gemini and search is system design worth it", trajectory) is None
    assert local_completion_shortfall("open WhatsApp and search Rakesh", [{"action_type": "open_app"}])


def test_page_and_file_descriptions_are_not_failures():
    description = (
        "Page URL: https://gemini.google.com/app\nVisual description:\nThere is no popup, modal, cookie banner, "
        "login or sign-in wall, CAPTCHA, or error message. The main input says 'Ask Gemini'."
    )
    assert output_indicates_failure(description, "see_page") is False
    assert output_indicates_failure("Vision is unavailable right now, so I cannot see the page.", "see_page") is True
    assert output_indicates_failure("Contents of C:/app.log:\nERROR 42 failed to connect", "read_file") is False
    assert output_indicates_failure("File not found: C:/x.txt", "read_file") is True
    assert output_indicates_failure("Captured WhatsApp :: WhatsApp (WINDOW, 900x700 px).", "see_window") is False
    # Action tools keep the broad check.
    assert output_indicates_failure("Command failed: boom", "run_command") is True


def test_reports_that_deny_a_problem_are_not_failures():
    assert report_indicates_failure("Typed the query and pressed Enter. No errors occurred.") is False
    assert report_indicates_failure("The page loaded without any problems and nothing failed.") is False
    assert report_indicates_failure("There is no popup, modal, cookie banner, CAPTCHA, or error on the page.") is False
    assert report_indicates_failure(
        "I opened the Gemini web app in a browser, typed the query and pressed Enter to submit it."
    ) is False
    assert report_indicates_failure("I could not find the search box.") is True
    assert report_indicates_failure("The file was not found.") is True


def test_web_apps_are_web_tasks():
    assert instruction_needs_web("open gemini and search is system designing worth learning")
    assert instruction_needs_web("ask chatgpt about arrays")
    assert not instruction_needs_web("open WhatsApp and message Rakesh hello")


# =============================================================================
# open_app never launches an unknown name blindly
# =============================================================================

@pytest.fixture
def local_tools(monkeypatch):
    from app.tools import local

    monkeypatch.setattr(local, "_check_enabled", lambda: None)
    monkeypatch.setattr(local.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="NOTFOUND\n"))
    launched: list = []
    monkeypatch.setattr(local.subprocess, "Popen", lambda *a, **k: launched.append(a))
    return local, launched


def test_unknown_app_name_is_not_launched(local_tools):
    local, launched = local_tools
    out = local.OpenAppTool().run(name="Gemini")
    assert out.startswith("Could not open 'Gemini'") and "navigate_browser" in out
    assert launched == []
    assert output_indicates_failure(out, "open_app") is True


def test_a_real_program_is_started_and_checked(local_tools, monkeypatch):
    local, launched = local_tools
    monkeypatch.setattr(local, "_launchable_command", lambda target: True)
    monkeypatch.setattr(local, "_wait_for_app_window", lambda hint, timeout_ms=0: ("notepad :: Untitled", "ready"))
    out = local.OpenAppTool().run(name="notepad.exe")
    assert out.startswith("Launched: notepad.exe") and launched
    assert output_indicates_failure(out, "open_app") is False


def test_launchable_command():
    from app.tools.local import _launchable_command

    assert _launchable_command("cmd")
    assert _launchable_command("ms-settings:display")
    assert not _launchable_command("Gemini")
    assert not _launchable_command("")


# =============================================================================
# 👍 on a run marked failed
# =============================================================================

def test_thumbs_up_on_a_failed_run_makes_it_a_success(semantic_memory, monkeypatch):
    monkeypatch.setattr(get_settings(), "semantic_min_similarity", 0.2)
    semantic_memory.upsert_sync(task_id="g1", instruction="open gemini and search system design", domain="local",
                                scope="local", success=False, trajectory=[{"action_type": "type_into_element"}],
                                lesson="")
    exp = semantic_memory.record_feedback_sync("g1", +1)
    assert exp.user_confirmed and exp.effective_success
    recalled = semantic_memory.recall_sync("open gemini and search system design")[0]
    assert recalled.user_confirmed and recalled.effective_success


def test_thumbs_up_before_the_failed_run_is_stored(semantic_memory, monkeypatch):
    monkeypatch.setattr(get_settings(), "semantic_min_similarity", 0.2)
    semantic_memory.record_feedback_sync("g0", +1)
    semantic_memory.upsert_sync(task_id="g0", instruction="ask gemini about trees", domain="local", scope="local",
                                success=False, trajectory=[], lesson="")
    assert semantic_memory.recall_sync("ask gemini about trees")[0].effective_success


async def test_feedback_message_says_the_run_now_counts(semantic_memory, monkeypatch):
    monkeypatch.setattr(get_settings(), "semantic_memory_enabled", True)
    semantic_memory.upsert_sync(task_id="g2", instruction="open gemini", domain="local", scope="local",
                                success=False, trajectory=[], lesson="")
    brain = BrainMemory(semantic=semantic_memory, reflection=ReflectionEngine(ScriptedCompletion([])))
    result = await brain.record_feedback("g2", 1)
    assert result["applied"] and "now counts as a success" in result["message"]


def test_explanation_shows_user_confirmed_runs():
    exp = Experience(task_id="g3", instruction="open gemini and search", domain="local", scope="local", success=False,
                     tools=GEMINI_RUN, lesson="", error="", feedback=1, user_disputed=False,
                     created_at=datetime.now(timezone.utc).isoformat(), similarity=0.8, user_confirmed=True)
    text = build_explanation(trajectory=[], success=True, experiences=[exp], strategy="", strategy_from_llm=False,
                             retried=False, first_attempt_error="", retry_blocked_reason="", error="")
    assert "you said it worked" in text


# =============================================================================
# A running agent is never replaced by a second copy
# =============================================================================

def _serve_health(body: bytes):
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_backend_detects_an_agent_that_is_already_running():
    from app.main import _agent_backend_running

    agent = _serve_health(b'{"status": "healthy", "wake_listener": {"running": true}}')
    other = _serve_health(b'{"status": "ok"}')
    try:
        assert _agent_backend_running(agent.server_address[1], timeout=3) is True
        assert _agent_backend_running(other.server_address[1], timeout=3) is False
    finally:
        agent.shutdown()
        other.shutdown()
    assert _agent_backend_running(agent.server_address[1], timeout=1) is False


# =============================================================================
# Desktop app: a new chat each time it is opened
# =============================================================================

async def test_active_run_count():
    started = asyncio.Event()

    async def run():
        me = track_run()
        started.set()
        try:
            await asyncio.sleep(5)
        finally:
            untrack_run(me)

    task = asyncio.create_task(run())
    await started.wait()
    assert active_run_count() == 1
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    assert active_run_count() == 0


@pytest.fixture
def desktop(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "desktop" / "agent_desktop.pyw"
    loader = importlib.machinery.SourceFileLoader("agent_desktop_chat", str(path))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader("agent_desktop_chat", loader))
    loader.exec_module(module)
    monkeypatch.setattr(module, "LOG_DIR", tmp_path)
    monkeypatch.setattr(module, "DESKTOP_LOG", tmp_path / "desktop.log")
    return module


def _app_with_fake_window(desktop):
    app = desktop.DesktopApp()
    loads: list[str] = []
    app.window = SimpleNamespace(load_url=loads.append, show=lambda: None, restore=lambda: None)
    app.showing_loading_page = False
    return app, loads


def test_opening_again_starts_a_new_chat(desktop, monkeypatch):
    app, loads = _app_with_fake_window(desktop)
    monkeypatch.setattr(desktop, "task_running", lambda: False)
    app.show_window(new_chat=True)
    assert loads == [desktop.UI_URL] and app.window_hidden is False


def test_a_running_task_is_not_wiped_by_a_new_chat(desktop, monkeypatch):
    app, loads = _app_with_fake_window(desktop)
    monkeypatch.setattr(desktop, "task_running", lambda: True)
    app.show_window(new_chat=True)
    assert loads == []


def test_bringing_up_a_visible_window_keeps_the_chat(desktop, monkeypatch):
    app, loads = _app_with_fake_window(desktop)
    monkeypatch.setattr(desktop, "task_running", lambda: False)
    app.show_window(new_chat=False)
    assert loads == []
