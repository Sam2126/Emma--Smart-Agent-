"""
Stopping the agent stops everything: the backend, its Chrome window and the
wake word. Found 2026-09-17: after the user stopped the project, the agent kept
listening in the background, heard people talking and ran it as a task.
"""

from __future__ import annotations

import asyncio
import importlib.machinery
import importlib.util
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

APP_PATH = Path(__file__).resolve().parents[2] / "desktop" / "agent_desktop.pyw"


@pytest.fixture
def desktop(tmp_path, monkeypatch):
    loader = importlib.machinery.SourceFileLoader("agent_desktop_stop", str(APP_PATH))
    spec = importlib.util.spec_from_loader("agent_desktop_stop", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    monkeypatch.setattr(module, "LOG_DIR", tmp_path)
    monkeypatch.setattr(module, "DESKTOP_LOG", tmp_path / "desktop.log")
    monkeypatch.setattr(module, "BACKEND_LOG", tmp_path / "backend.log")
    return module


class _Window:
    def __init__(self):
        self.calls: list[str] = []

    def hide(self):
        self.calls.append("hide")

    def destroy(self):
        self.calls.append("destroy")

    def load_html(self, html):
        self.calls.append("load_html")

    def load_url(self, url):
        self.calls.append("load_url")

    def evaluate_js(self, script):
        return None


def _join(name: str) -> None:
    for thread in threading.enumerate():
        if thread.name == name:
            thread.join(timeout=5)


# =============================================================================
# Desktop app
# =============================================================================

def test_closing_the_window_stops_everything(desktop, monkeypatch):
    app = desktop.DesktopApp()
    app.window = _Window()
    stopped = []
    monkeypatch.setattr(app.backend, "stop_any", lambda: stopped.append(True) or True)

    assert app.on_closing() is False  # quit() closes the window after stopping the agent
    _join("quit")
    time.sleep(0.1)  # the window is hidden from a thread of its own
    assert app.quitting and stopped == [True]
    assert "hide" in app.window.calls and "destroy" in app.window.calls
    assert app.on_closing() is True  # the final close goes through


def test_quit_also_stops_an_agent_the_app_did_not_start(desktop, monkeypatch):
    requests = []
    answers = iter([{"ok": True}, None])
    monkeypatch.setattr(desktop, "health", lambda **kw: next(answers))
    monkeypatch.setattr(desktop, "http", lambda method, route, timeout=2.0: requests.append((method, route)) or (200, ""))
    monkeypatch.setattr(desktop.time, "sleep", lambda seconds: None)
    app = desktop.DesktopApp()
    assert not app.backend.owned
    app.quit()
    _join("quit")
    assert requests == [("POST", "/app/shutdown")]


def test_hiding_to_the_tray_is_an_explicit_choice(desktop, monkeypatch):
    app = desktop.DesktopApp()
    app.window = _Window()
    notes = []
    app.tray = SimpleNamespace(notify=lambda text, title: notes.append((title, text)))
    app.hide_to_tray()
    time.sleep(0.1)
    assert app.window_hidden and app.window.calls == ["hide"] and not app.quitting
    assert notes and "Quit" in notes[0][1]


def test_an_agent_stopped_on_purpose_closes_the_app_too(desktop, monkeypatch):
    app = desktop.DesktopApp()
    quits = []
    monkeypatch.setattr(desktop, "health", lambda **kw: None)
    monkeypatch.setattr(app, "quit", lambda: quits.append(True))
    app.on_backend_exit(0, served=True)
    assert quits == [True]


def test_losing_the_port_race_never_stops_the_other_agent(desktop, monkeypatch):
    """Found 2026-09-18: the app and start_agent.ps1 each started an agent within three
    seconds; the app's copy exited with code 0 ("already running, not starting a second
    copy"), the app read that as "the user stopped it" and shut the other one down."""
    app = desktop.DesktopApp()
    app.window = _Window()
    app.showing_loading_page = False
    running = {"ok": True, "wake_word_enabled": True}
    stopped, quits = [], []
    monkeypatch.setattr(desktop, "health", lambda **kw: running)
    monkeypatch.setattr(desktop, "http", lambda *a, **kw: stopped.append(a) or (200, ""))
    monkeypatch.setattr(app, "quit", lambda: quits.append(True))

    app.on_backend_exit(0, served=False)

    assert stopped == [] and quits == [], "the agent that is answering was not started by this app"
    assert app.window.calls == ["load_url"] and app.showing_loading_page is False  # attached to it instead


def test_an_agent_that_never_started_is_reported_not_stopped(desktop, monkeypatch):
    app = desktop.DesktopApp()
    app.window = _Window()
    app.showing_loading_page = False
    notes, quits = [], []
    monkeypatch.setattr(desktop, "health", lambda **kw: None)
    monkeypatch.setattr(app, "quit", lambda: quits.append(True))
    app.tray = SimpleNamespace(notify=lambda text, title: notes.append(title), title="", update_menu=lambda: None)

    app.on_backend_exit(0, served=False)

    assert quits == [] and notes == ["The agent could not start"]
    assert "could not start" in app.status


def test_an_agent_already_starting_on_the_port_is_waited_for(desktop, monkeypatch):
    answers = iter([None, None, {"ok": True}])
    spawned = []
    monkeypatch.setattr(desktop, "health", lambda **kw: next(answers, {"ok": True}))
    monkeypatch.setattr(desktop, "port_in_use", lambda port=8000: True)
    monkeypatch.setattr(desktop.time, "sleep", lambda seconds: None)
    backend = desktop.Backend(lambda text: None, lambda h: None)
    monkeypatch.setattr(backend, "_spawn", lambda: spawned.append(True))

    assert backend.bring_up() == {"ok": True}
    assert spawned == [], "a second agent must never be started on a port that is already taken"


def test_a_crashed_agent_is_not_restarted(desktop, monkeypatch):
    app = desktop.DesktopApp()
    app.window = _Window()
    app.showing_loading_page = False
    starts, notes = [], []
    monkeypatch.setattr(app.backend, "bring_up", lambda: starts.append(True))
    app.tray = SimpleNamespace(notify=lambda text, title: notes.append(title), title="", update_menu=lambda: None)
    app.on_backend_exit(1)
    assert starts == []
    assert "stopped unexpectedly" in app.status and "Start agent" in app.status
    assert app.window.calls == ["load_html"] and notes == ["The agent stopped"]


def test_the_watcher_reports_an_exit_the_app_did_not_ask_for(desktop):
    exits = []
    backend = desktop.Backend(lambda text: None, lambda h: None, lambda code, served: exits.append((code, served)))
    process = SimpleNamespace(wait=lambda: 3, poll=lambda: 3)
    backend.process = process
    backend.served = True
    backend._watch(process)
    assert exits == [(3, True)] and backend.process is None


def test_the_backend_is_told_who_started_it(desktop, monkeypatch, tmp_path):
    seen = {}

    class FakePopen:
        pid = 4321

        def __init__(self, args, **kwargs):
            seen.update(kwargs)

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(desktop.subprocess, "Popen", FakePopen)
    backend = desktop.Backend(lambda text: None, lambda h: None)
    backend.stopping = True  # the watcher thread must not report this fake exit
    backend._spawn()
    assert seen["env"]["AGENT_PARENT_PID"] == str(__import__("os").getpid())


def test_the_running_app_can_be_asked_to_quit(desktop, monkeypatch):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setattr(desktop, "SIGNAL_PORT", port)
    shows, quits = [], []
    desktop.listen_for_requests(lambda: shows.append(True), lambda: quits.append(True))
    deadline = time.monotonic() + 5
    while not desktop.send_to_running_instance(b"quit") and time.monotonic() < deadline:
        time.sleep(0.05)
    while not quits and time.monotonic() < deadline:
        time.sleep(0.05)
    assert quits == [True] and shows == []


def test_quit_flag_stops_the_app_and_any_agent(desktop, monkeypatch):
    sent, stops = [], []
    monkeypatch.setattr(desktop, "send_to_running_instance", lambda command: sent.append(command) or True)
    monkeypatch.setattr(desktop.Backend, "stop_any", lambda self: stops.append(True) or True)
    assert desktop.quit_everything() == 0
    assert sent == [b"quit"] and stops == [True]


def test_the_wake_word_shown_in_the_app_comes_from_env(desktop, monkeypatch, tmp_path):
    app_dir = tmp_path / "desktop"
    app_dir.mkdir()
    monkeypatch.setattr(desktop, "APP_DIR", app_dir)
    assert desktop.wake_word() == "emma"  # no .env: the backend's default
    (tmp_path / ".env").write_text("GROQ_API_KEYS=x\nWAKE_WORD='jarvis'  # mine\n", encoding="utf-8")
    assert desktop.wake_word() == "jarvis"
    assert '"Jarvis"' in desktop.loading_html() and "{wake}" not in desktop.loading_html()


# =============================================================================
# Backend: it stops itself when its desktop app is gone
# =============================================================================

async def test_backend_stops_when_its_app_is_gone(monkeypatch):
    import signal

    from app import main

    raised = []
    monkeypatch.setattr(signal, "raise_signal", lambda sig: raised.append(sig))
    monkeypatch.setattr(main, "_PARENT_POLL_SECONDS", 0.05)
    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.5)"])
    try:
        assert await asyncio.wait_for(main.watch_parent_app(parent.pid), timeout=10) is True
    finally:
        parent.wait(timeout=10)
    assert raised == [signal.SIGINT]


async def test_backend_keeps_running_while_its_app_runs(monkeypatch):
    import signal

    from app import main

    raised = []
    monkeypatch.setattr(signal, "raise_signal", lambda sig: raised.append(sig))
    monkeypatch.setattr(main, "_PARENT_POLL_SECONDS", 0.05)
    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(main.watch_parent_app(parent.pid), timeout=0.5)
    finally:
        parent.kill()
        parent.wait(timeout=10)
    assert raised == []


async def test_backend_stops_when_its_app_already_exited(monkeypatch):
    import signal

    from app import main

    raised = []
    monkeypatch.setattr(signal, "raise_signal", lambda sig: raised.append(sig))
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait(timeout=10)
    time.sleep(0.2)
    assert await main.watch_parent_app(gone.pid) is True
    assert raised == [signal.SIGINT]


# =============================================================================
# The agent's own Chrome is closed on shutdown, even one from an earlier run
# =============================================================================

async def test_an_attached_agent_chrome_is_closed_on_shutdown():
    from app.browser.controller import BrowserController

    sent = []

    class _Session:
        async def send(self, method, params=None):
            sent.append(method)

    async def new_session():
        return _Session()

    controller = BrowserController()
    controller._connected = True
    controller._browser = SimpleNamespace(new_browser_cdp_session=new_session, contexts=[])
    controller._close_on_disconnect = True
    await controller.disconnect()
    assert sent == ["Browser.close"]


async def test_a_chrome_that_is_not_the_agents_is_left_alone():
    from app.browser.controller import BrowserController

    async def must_not_open():
        raise AssertionError("the user's Chrome must not be closed")

    controller = BrowserController()
    controller._connected = True
    controller._browser = SimpleNamespace(new_browser_cdp_session=must_not_open, contexts=[])
    controller._close_on_disconnect = False
    await controller.disconnect()


def test_the_agents_chrome_is_recognised_on_its_port(monkeypatch, tmp_path):
    from app.browser import controller as ctl

    chrome = ctl._find_chrome_exe()
    if not chrome:
        pytest.skip("Chrome is not installed")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setattr(ctl, "AGENT_PROFILE_DIR", tmp_path / "agent_profile")
    proc = subprocess.Popen(ctl._agent_chrome_command(chrome, port) + ["--headless=new"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert asyncio.run(ctl._wait_for_cdp_port("127.0.0.1", port, timeout=20))
        assert ctl._agent_chrome_on_port(port) is True
        monkeypatch.setattr(ctl, "AGENT_PROFILE_DIR", tmp_path / "some_other_profile")
        assert ctl._agent_chrome_on_port(port) is False
    finally:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)


# =============================================================================
# Wake word: nothing runs without the stop word
# =============================================================================

class _Mic:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_a_recording_without_the_stop_word_is_not_run(monkeypatch):
    from app import wake_listener

    monkeypatch.setattr(wake_listener, "_beep_error", lambda: None)
    listener = wake_listener.WakeWordListener(wake_word="emma", stop_word="done", listen_timeout=-1)
    dispatched = []
    monkeypatch.setattr(listener, "_dispatch_task", lambda text: dispatched.append(text))
    monkeypatch.setattr(listener, "_dispatch_feedback", lambda *a: dispatched.append(a))
    monkeypatch.setattr(listener, "_transcribe_via_whisper", lambda wav: "people talking in the room")
    wake_audio = SimpleNamespace(sample_rate=16000, sample_width=2, frame_data=b"\x00\x01" * 16000)
    recognizer = SimpleNamespace(pause_threshold=1.0)

    listener._record_instruction(recognizer, _Mic(), initial_text="emma", wake_audio=wake_audio)
    assert dispatched == []


@pytest.mark.parametrize("heard, wakes", [
    ("emma", True),
    ("Emma,", True),
    ("hey emma", True),
    ("emma open notepad", True),
    ("amber emma", False),
    ("gemma emma", False),
    ("what's your name emma", False),
    ("tell emma i'm late", False),
])
def test_emma_must_start_the_phrase(heard, wakes):
    from app.wake_listener import WakeWordListener

    assert WakeWordListener(wake_word="emma", stop_word="done")._matches_wake_word(heard) is wakes
