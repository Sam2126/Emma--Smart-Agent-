"""Tests for the desktop app (desktop/agent_desktop.pyw) that need no window."""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path

import pytest

APP_PATH = Path(__file__).resolve().parents[2] / "desktop" / "agent_desktop.pyw"


@pytest.fixture
def desktop(tmp_path, monkeypatch):
    loader = importlib.machinery.SourceFileLoader("agent_desktop", str(APP_PATH))
    spec = importlib.util.spec_from_loader("agent_desktop", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    # Tests must not write into the real app log.
    monkeypatch.setattr(module, "LOG_DIR", tmp_path)
    monkeypatch.setattr(module, "DESKTOP_LOG", tmp_path / "desktop.log")
    return module


def test_describe_health(desktop):
    assert desktop.describe_health(None) == "The agent is not running"
    listening = {"wake_listener": {"running": True, "engine": "vosk", "wake_word": "hello"}}
    assert desktop.describe_health(listening) == 'Listening for "hello" (offline wake word)'
    assert "online" in desktop.describe_health({"wake_listener": {"running": True, "engine": "google", "wake_word": "hello"}})
    assert desktop.describe_health({"wake_word_enabled": False, "wake_listener": {"running": False}}) == "Ready (wake word is turned off)"
    assert desktop.describe_health({}) == "Ready"


def test_only_the_agent_page_counts_as_the_agent(desktop):
    assert desktop.is_agent_url("http://localhost:8000/local")
    assert desktop.is_agent_url("http://127.0.0.1:8000/")
    assert not desktop.is_agent_url("http://localhost:8001/local")
    assert not desktop.is_agent_url("https://evil.example/http://localhost:8000")


def test_login_command_starts_hidden_with_pythonw(desktop):
    command = desktop.login_command()
    assert "pythonw.exe" in command and "agent_desktop.pyw" in command and command.endswith("--hidden")


def test_backend_attaches_to_a_running_backend(desktop, monkeypatch):
    statuses, ready = [], []
    monkeypatch.setattr(desktop, "health", lambda **kw: {"wake_listener": {"running": True, "engine": "vosk", "wake_word": "hello"}})
    spawned = []
    backend = desktop.Backend(statuses.append, ready.append)
    monkeypatch.setattr(backend, "_spawn", lambda: spawned.append(True))
    assert backend.bring_up() is not None
    assert ready and not spawned and not backend.owned


def test_backend_reports_when_it_cannot_start(desktop, monkeypatch):
    statuses = []
    monkeypatch.setattr(desktop, "health", lambda **kw: None)
    monkeypatch.setattr(desktop.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(desktop, "BACKEND_START_SECONDS", 0)
    backend = desktop.Backend(statuses.append, lambda h: None)
    monkeypatch.setattr(backend, "_spawn", lambda: None)
    assert backend.bring_up() is None
    assert "did not start" in statuses[-1]


def test_a_slow_health_check_does_not_start_a_second_backend(desktop, monkeypatch):
    answers = iter([None, {"wake_listener": {"running": True, "engine": "vosk", "wake_word": "hello"}}])
    monkeypatch.setattr(desktop, "health", lambda **kw: next(answers))
    monkeypatch.setattr(desktop.time, "sleep", lambda seconds: None)
    spawned = []
    backend = desktop.Backend(lambda text: None, lambda h: None)
    monkeypatch.setattr(backend, "_spawn", lambda: spawned.append(True))
    assert backend.bring_up() is not None
    assert spawned == []


def test_waiting_for_the_single_instance_lock_gives_up(desktop, monkeypatch):
    import os
    import time
    import uuid

    monkeypatch.setattr(desktop, "MUTEX_NAME", f"Local\\SelfImprovingAgentWait-{os.getpid()}-{uuid.uuid4().hex}")
    assert desktop.acquire_single_instance() is True
    started = time.monotonic()
    assert desktop.acquire_single_instance(wait_seconds=0.3) is False
    assert time.monotonic() - started >= 0.25


def _app_recording_calls(desktop, monkeypatch):
    app = desktop.DesktopApp()
    calls: list = []
    monkeypatch.setattr(app, "restart_app", lambda: calls.append("restart_app"))
    monkeypatch.setattr(app, "show_window", lambda new_chat=False: calls.append(("show", new_chat)))
    monkeypatch.setattr(app, "_restart_or_start", lambda: calls.append("restart_backend"))
    return app, calls


def test_opening_the_icon_restarts_an_outdated_app(desktop, monkeypatch):
    """Found 2026-09-15: a copy started before an update kept showing the old chat."""
    app, calls = _app_recording_calls(desktop, monkeypatch)
    monkeypatch.setattr(desktop, "app_code_stamp", lambda: app.app_stamp + 60)
    app.on_show_request()
    assert calls == ["restart_app"]


def test_opening_the_icon_starts_a_stopped_agent(desktop, monkeypatch):
    """Found 2026-09-16: the app had attached to a backend started elsewhere; when
    that stopped, opening the icon showed an empty window and started nothing."""
    import threading

    app, calls = _app_recording_calls(desktop, monkeypatch)
    monkeypatch.setattr(desktop, "app_code_stamp", lambda: app.app_stamp)
    monkeypatch.setattr(desktop, "health", lambda **kw: None)
    monkeypatch.setattr(app.backend, "bring_up", lambda: calls.append("bring_up"))
    app.window = type("W", (), {"load_html": lambda self, html: None})()

    app.on_show_request()
    for thread in threading.enumerate():
        if thread.name == "start-on-open":
            thread.join(timeout=2)
    assert calls == [("show", False), "bring_up"]


def test_the_app_never_starts_the_agent_on_its_own(desktop, monkeypatch):
    """Found 2026-09-17: the user stopped the agent, the app started it again,
    and the wake word ran a conversation as a task. Only opening the icon or
    "Start agent" starts it now."""
    app, _ = _app_recording_calls(desktop, monkeypatch)
    started, cycles = [], []
    monkeypatch.setattr(desktop, "health", lambda **kw: None)
    monkeypatch.setattr(app.backend, "bring_up", lambda: started.append(True))

    def fake_sleep(seconds):
        cycles.append(seconds)
        app.quitting = len(cycles) > 2

    monkeypatch.setattr(desktop.time, "sleep", fake_sleep)
    app.poll_health()
    assert started == []
    assert "not running" in app.status and "Start agent" in app.status


def test_opening_the_icon_on_current_code_starts_a_new_chat(desktop, monkeypatch):
    app, calls = _app_recording_calls(desktop, monkeypatch)
    monkeypatch.setattr(desktop, "app_code_stamp", lambda: app.app_stamp)
    monkeypatch.setattr(desktop, "health", lambda **kw: {"wake_listener": {"running": True, "engine": "vosk", "wake_word": "hello"}})
    app.on_show_request()
    assert calls == [("show", True)]


def test_outdated_backend_started_by_the_app_is_restarted(desktop, monkeypatch):
    import threading

    app, calls = _app_recording_calls(desktop, monkeypatch)
    monkeypatch.setattr(desktop, "app_code_stamp", lambda: app.app_stamp)
    monkeypatch.setattr(desktop, "health", lambda **kw: {"wake_listener": {"running": True, "engine": "vosk", "wake_word": "hello"}})
    monkeypatch.setattr(desktop, "backend_code_stamp", lambda: 200.0)
    monkeypatch.setattr(desktop, "task_running", lambda: False)
    app.backend.process = type("Running", (), {"poll": lambda self: None})()
    app.backend.code_stamp = 100.0
    app.on_show_request()
    for thread in threading.enumerate():
        if thread.name == "update-restart":
            thread.join(timeout=2)
    assert calls == [("show", False), "restart_backend"]


def test_backend_code_stamp_sees_backend_files(desktop):
    assert desktop.backend_code_stamp() > 0
    assert desktop.app_code_stamp() > 0


def test_outdated_backend_is_detected_from_health(desktop, monkeypatch):
    monkeypatch.setattr(desktop, "backend_code_stamp", lambda: 200.0)
    assert desktop.backend_is_outdated({"code_stamp": 100.0}) is True
    assert desktop.backend_is_outdated({"code_stamp": 200.0}) is False
    assert desktop.backend_is_outdated({"code_stamp": 300.0}) is False
    assert desktop.backend_is_outdated({"code_stamp": 199.5}) is False  # within the tolerance
    assert desktop.backend_is_outdated({}) is False  # a backend too old to report it
    assert desktop.backend_is_outdated(None) is False
    assert desktop.backend_is_outdated({"code_stamp": True}) is False
    assert desktop.backend_is_outdated({"code_stamp": "200"}) is False


def test_opening_the_icon_restarts_an_outdated_backend_it_did_not_start(desktop, monkeypatch):
    """Found 2026-09-16: the app had attached to a backend started at 14:54, so it
    owned no process, and the icon's update check only looked at backends the app
    had started itself. A fix saved at 15:13 never reached the running agent, and a
    task at 15:06 wrote its folder where the user could not see it."""
    import threading

    app, calls = _app_recording_calls(desktop, monkeypatch)
    monkeypatch.setattr(desktop, "app_code_stamp", lambda: app.app_stamp)
    monkeypatch.setattr(desktop, "health", lambda **kw: {"code_stamp": 100.0, "task_running": False})
    monkeypatch.setattr(desktop, "backend_code_stamp", lambda: 200.0)

    assert app.backend.owned is False
    app.on_show_request()
    for thread in threading.enumerate():
        if thread.name == "update-restart":
            thread.join(timeout=2)
    assert calls == [("show", False), "restart_backend"]


def test_a_running_task_is_never_interrupted_by_the_update_check(desktop, monkeypatch):
    app, calls = _app_recording_calls(desktop, monkeypatch)
    monkeypatch.setattr(desktop, "app_code_stamp", lambda: app.app_stamp)
    monkeypatch.setattr(desktop, "health", lambda **kw: {"code_stamp": 100.0, "task_running": True})
    monkeypatch.setattr(desktop, "backend_code_stamp", lambda: 200.0)
    app.on_show_request()
    assert calls == [("show", True)]


def test_stopping_a_backend_this_app_did_not_start(desktop, monkeypatch):
    """stop() only ends a process the app holds a handle to; an attached backend
    has to be asked over HTTP, or it keeps serving old code."""
    calls: list = []
    alive = iter([{"status": "healthy"}, {"status": "healthy"}, None])
    monkeypatch.setattr(desktop, "health", lambda **kw: next(alive, None))
    monkeypatch.setattr(desktop, "http", lambda method, route, timeout=2.0: calls.append((method, route)) or (200, ""))
    monkeypatch.setattr(desktop.time, "sleep", lambda seconds: None)

    backend = desktop.Backend(lambda text: None, lambda h: None)
    assert backend.owned is False
    assert backend.stop_any() is True
    assert calls == [("POST", "/app/shutdown")]


def test_starting_up_replaces_an_outdated_running_backend(desktop, monkeypatch):
    """Otherwise the app attaches to the old agent again and the update is lost."""
    monkeypatch.setattr(desktop, "health", lambda **kw: {"code_stamp": 100.0, "task_running": False})
    monkeypatch.setattr(desktop, "backend_code_stamp", lambda: 200.0)
    monkeypatch.setattr(desktop.time, "sleep", lambda seconds: None)
    stopped, spawned = [], []

    backend = desktop.Backend(lambda text: None, lambda h: None)
    monkeypatch.setattr(backend, "stop_any", lambda: stopped.append(True) or True)
    monkeypatch.setattr(backend, "_spawn", lambda: spawned.append(True))
    backend.bring_up()
    assert stopped == [True] and spawned == [True]


def test_a_current_backend_is_reused_not_restarted(desktop, monkeypatch):
    monkeypatch.setattr(desktop, "health", lambda **kw: {"code_stamp": 200.0, "task_running": False})
    monkeypatch.setattr(desktop, "backend_code_stamp", lambda: 200.0)
    stopped, spawned = [], []

    backend = desktop.Backend(lambda text: None, lambda h: None)
    monkeypatch.setattr(backend, "stop_any", lambda: stopped.append(True) or True)
    monkeypatch.setattr(backend, "_spawn", lambda: spawned.append(True))
    assert backend.bring_up() is not None
    assert stopped == [] and spawned == []


def test_a_restart_the_app_asked_for_is_not_treated_as_a_crash(desktop, monkeypatch):
    """The watcher can wake after restart() has cleared `stopping`. Found
    2026-09-16: a planned restart logged "stopped unexpectedly" and asked for a
    second start while the first was still running."""
    class FakeProcess:
        def __init__(self) -> None:
            self.polls = 0

        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else 0

        def wait(self, timeout=None):
            return 0

    statuses: list = []
    monkeypatch.setattr(desktop, "http", lambda *args, **kwargs: (200, ""))
    backend = desktop.Backend(statuses.append, lambda h: None)
    process = FakeProcess()
    backend.process = process

    backend.stop()
    assert backend.process is None
    backend.stopping = False  # what restart() does immediately after stop()
    backend._watch(process)  # the watcher finally wakes up
    assert not any("unexpectedly" in status for status in statuses)


def test_the_app_and_the_backend_measure_the_same_code(desktop):
    """The icon's update check compares these two numbers. If they counted
    different files, the app would restart the agent endlessly or never."""
    from app.main import _CODE_STAMP, _backend_code_stamp

    assert _CODE_STAMP > 0
    assert abs(desktop.backend_code_stamp() - _backend_code_stamp()) < 0.01


def test_single_instance_mutex(desktop, monkeypatch):
    import os
    import uuid

    # A name of its own: the real app may be running while tests run.
    monkeypatch.setattr(desktop, "MUTEX_NAME", f"Local\\SelfImprovingAgentTest-{os.getpid()}-{uuid.uuid4().hex}")
    assert desktop.acquire_single_instance() is True
    assert desktop.acquire_single_instance() is False
