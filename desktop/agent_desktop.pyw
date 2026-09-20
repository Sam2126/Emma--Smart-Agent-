"""
Self-Improving Agent — desktop app.

One click on the Desktop or Start-menu icon and the agent is running:
  * the Python backend starts hidden in the background (or the app attaches to
    a backend that is already running)
  * the Local Agent page opens in a native window
  * say "Emma" (the wake word in .env), the task, then "done"
  * closing the window stops everything: the agent, its Chrome window and the
    wake word. To keep it listening with no window, choose "Hide window, keep
    listening" in the tray menu (the tray icon then shows it is still running)
  * tray menu: Open Agent, Hide window, Start with Windows, Restart agent,
    Open log, Quit

Why this shape: the window is Microsoft Edge WebView2, which ships with
Windows 11, so there is no ~100 MB browser engine to download and start (an
Electron app bundles its own). The things an Electron app does well — tray
icon, single instance, start at login, notifications, restart after a crash —
come from small pieces: pywebview (window), pystray (tray), winreg (login
item) and ctypes (single-instance mutex).

    Run:      backend\\.venv\\Scripts\\pythonw.exe desktop\\agent_desktop.pyw
    Install:  backend\\.venv\\Scripts\\python.exe backend\\scripts\\install_desktop_app.py
    Test:     backend\\.venv\\Scripts\\python.exe desktop\\agent_desktop.pyw --smoke-test
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path

APP_NAME = "Self-Improving Agent"
APP_ID = "SelfImprovingAgent.Desktop"
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent / "backend"
VENV_SCRIPTS = BACKEND_DIR / ".venv" / "Scripts"
PYTHON = VENV_SCRIPTS / "python.exe"
PYTHONW = VENV_SCRIPTS / "pythonw.exe"
LOG_DIR = BACKEND_DIR / "data" / "logs"
BACKEND_LOG = LOG_DIR / "backend.log"
DESKTOP_LOG = LOG_DIR / "desktop.log"
WEBVIEW_DATA = BACKEND_DIR / "data" / "desktop_webview"
ICON_PNG = APP_DIR / "assets" / "icon.png"
ICON_ICO = APP_DIR / "assets" / "icon.ico"

API = "http://127.0.0.1:8000"
API_PORT = 8000
UI_URL = "http://localhost:8000/local"
SIGNAL_PORT = 8767  # a second launch asks the running app to show its window (or to quit)
MUTEX_NAME = "Local\\SelfImprovingAgentDesktop"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
BACKEND_START_SECONDS = 120
OTHER_AGENT_WAIT_SECONDS = 90  # an agent already starting on the port is waited for, never raced
START_HINT = "Open the icon again, or choose Start agent in the tray menu, to start it."
CREATE_NO_WINDOW = 0x08000000

SMOKE_TEST = "--smoke-test" in sys.argv
START_HIDDEN = "--hidden" in sys.argv
QUIT = "--quit" in sys.argv  # stop the running app and agent, then exit
AFTER_UPDATE = "--after-update" in sys.argv  # started by an older copy that saw new code on disk
BACKEND_APP_DIR = BACKEND_DIR / "app"
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200

def wake_word() -> str:
    """The wake word the backend listens for: WAKE_WORD in .env, else its default."""
    try:
        for line in (APP_DIR.parent / ".env").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            value = value.split("#")[0].strip().strip("'\"")
            if key.strip().upper() == "WAKE_WORD" and value:
                return value
    except OSError:
        pass
    return "emma"


LOADING_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Self-Improving Agent</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0f1117; color: #e8eaf2; font-family: "Segoe UI", system-ui, sans-serif; height: 100vh;
         display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 16px;
         padding: 32px; text-align: center; }
  .logo { width: 72px; height: 72px; border-radius: 20px; background: linear-gradient(135deg, #4f8cff, #8f5cff);
          display: flex; align-items: center; justify-content: center; font-size: 36px; }
  h1 { font-size: 20px; font-weight: 600; }
  #status { font-size: 14px; color: #c9cddb; min-height: 20px; }
  .spinner { width: 28px; height: 28px; border: 3px solid #262a38; border-top-color: #4f8cff; border-radius: 50%;
             animation: spin 0.9s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .hint { font-size: 12px; color: #8b90a3; line-height: 1.6; max-width: 360px; }
  b { color: #e8eaf2; }
</style></head>
<body>
  <div class="logo">🎙️</div>
  <h1>Self-Improving Agent</h1>
  <p id="status">Starting…</p>
  <div class="spinner"></div>
  <p class="hint">When it is ready, say <b>"{wake}"</b>, give your task, then say <b>"done"</b>.<br>
     Closing this window keeps the agent listening in the system tray.</p>
  <script>window.setStatus = (text) => { document.getElementById("status").textContent = text; };</script>
</body></html>"""


def loading_html() -> str:
    return LOADING_HTML.replace("{wake}", wake_word().capitalize())

# Requests to the local backend must never go through a system proxy or VPN.
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_log_lock = threading.Lock()


# =============================================================================
# Helpers
# =============================================================================

def log(message: str) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with _log_lock, open(DESKTOP_LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} {message}\n")
    except OSError:
        pass


def http(method: str, route: str, timeout: float = 2.0) -> tuple[int | None, str]:
    request = urllib.request.Request(API + route, method=method, data=b"" if method == "POST" else None)
    try:
        with _opener.open(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except Exception:
        return None, ""


def port_in_use(port: int = API_PORT) -> bool:
    """True when something already holds the agent's port — usually an agent still starting."""
    with socket.socket() as probe:
        probe.settimeout(0.5)
        try:
            return probe.connect_ex(("127.0.0.1", port)) == 0
        except OSError:
            return False


def health(timeout: float = 2.0) -> dict | None:
    status, body = http("GET", "/health", timeout=timeout)
    if status != 200:
        return None
    try:
        return json.loads(body)
    except ValueError:
        return {}


def describe_health(h: dict | None) -> str:
    if h is None:
        return "The agent is not running"
    wake = h.get("wake_listener") or {}
    if wake.get("running"):
        engine = "offline" if wake.get("engine") == "vosk" else "online"
        return f'Listening for "{wake.get("wake_word")}" ({engine} wake word)'
    if h.get("wake_word_enabled") is False:
        return "Ready (wake word is turned off)"
    return "Ready"


def task_running() -> bool:
    h = health()
    return bool(h and h.get("task_running"))


# --- Code on disk vs. code running --------------------------------------------
# Found 2026-09-15: the app lives in the tray for hours, so after an update
# the icon kept showing a copy started before the update (with its old chat
# and old behaviour). Opening the icon now compares the code on disk with the
# code that is running and restarts what is outdated.

def _newest_mtime(paths) -> float:
    newest = 0.0
    for path in paths:
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            pass
    return newest


def app_code_stamp() -> float:
    return _newest_mtime([Path(__file__).resolve()])


def backend_code_stamp() -> float:
    return _newest_mtime([*BACKEND_APP_DIR.rglob("*.py"), *(BACKEND_APP_DIR / "static").glob("*.html")])


def backend_is_outdated(running_health: dict | None) -> bool:
    """True when the agent on disk is newer than the agent that is running.

    The backend reports in /health the stamp of the code it loaded at startup,
    so this works for a backend this app did not start. Found 2026-09-16: the
    app had attached to a backend running since 14:54, and the icon's update
    check only looked at backends the app had started itself. A fix saved at
    15:13 therefore never reached the running agent, and a task at 15:06 wrote
    a folder to a path the user could not see, looking like it did nothing.
    """
    running = (running_health or {}).get("code_stamp")
    if not isinstance(running, (int, float)) or isinstance(running, bool) or running <= 0:
        return False
    return backend_code_stamp() > float(running) + 1.0


def is_agent_url(url: str) -> bool:
    return bool(re.match(r"^http://(localhost|127\.0\.0\.1):8000(/|$)", url or ""))


def message_box(text: str) -> None:
    try:
        ctypes.windll.user32.MessageBoxW(None, text, APP_NAME, 0x40)
    except Exception:
        pass


# =============================================================================
# Single instance, start with Windows
# =============================================================================

_mutex = None


def acquire_single_instance(wait_seconds: float = 0.0) -> bool:
    """True for the first instance. Uses a named mutex the OS frees on exit.

    wait_seconds: keep trying that long (a restarted app waits for the old copy to exit).
    """
    global _mutex
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    deadline = time.monotonic() + wait_seconds
    while True:
        handle = kernel32.CreateMutexW(None, False, MUTEX_NAME)
        if ctypes.get_last_error() != 183:  # ERROR_ALREADY_EXISTS
            _mutex = handle
            return True
        kernel32.CloseHandle(handle)
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.25)


def send_to_running_instance(command: bytes) -> bool:
    """Send "show" or "quit" to the copy of the app that is already running."""
    try:
        with socket.create_connection(("127.0.0.1", SIGNAL_PORT), timeout=2) as conn:
            conn.sendall(command)
        return True
    except OSError:
        return False


def ask_running_instance_to_show() -> None:
    send_to_running_instance(b"show")


def listen_for_requests(on_show, on_quit) -> None:
    """ "show" from a second launch of the icon, "quit" from start_agent.ps1 or --quit."""
    def serve() -> None:
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            server.bind(("127.0.0.1", SIGNAL_PORT))
            server.listen(2)
        except OSError as e:
            log(f"show-request listener unavailable: {e}")
            return
        while True:
            try:
                conn, _ = server.accept()
                with conn:
                    message = conn.recv(16)
                    if message.startswith(b"show"):
                        on_show()
                    elif message.startswith(b"quit"):
                        on_quit()
            except OSError:
                time.sleep(1)

    threading.Thread(target=serve, name="show-requests", daemon=True).start()


def login_command() -> str:
    return f'"{PYTHONW}" "{Path(__file__).resolve()}" --hidden'


def starts_with_windows() -> bool:
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            value, _ = winreg.QueryValueEx(key, APP_NAME)
            return value == login_command()
    except OSError:
        return False


def set_start_with_windows(enabled: bool) -> None:
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
        if enabled:
            winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, login_command())
        else:
            try:
                winreg.DeleteValue(key, APP_NAME)
            except FileNotFoundError:
                pass
    log(f"start with Windows: {enabled}")


# =============================================================================
# Backend process
# =============================================================================

class Backend:
    """Starts, watches, restarts and stops the backend process."""

    def __init__(self, on_status, on_ready, on_exit=None) -> None:
        self.on_status = on_status
        self.on_ready = on_ready
        # Called with the exit code when the backend ends without this app asking.
        self.on_exit = on_exit
        self.process: subprocess.Popen | None = None
        self.code_stamp = 0.0  # backend code on disk when this app started the backend
        self.served = False    # did the backend this app started ever answer /health?
        self.stopping = False
        self._bring_up_lock = threading.Lock()

    @property
    def owned(self) -> bool:
        """True while a backend this app started is running."""
        return self.process is not None and self.process.poll() is None

    def bring_up(self) -> dict | None:
        if not self._bring_up_lock.acquire(blocking=False):
            return None  # already starting
        try:
            self.on_status("Starting the agent…")
            h = health()
            if h is None:
                # Ask again before starting a second backend. Found 2026-09-15: one
                # missed 2 s check made the app start its own backend, whose startup
                # frees the ports and so killed the backend already running in the
                # start_agent.ps1 console.
                for _ in range(2):
                    time.sleep(1)
                    h = health(timeout=5.0)
                    if h is not None:
                        break
            if h is not None and backend_is_outdated(h) and not h.get("task_running"):
                # Attaching to it would keep old code running for as long as the
                # app stays in the tray, which is how a fix saved at 15:13 was
                # still not running at 15:22 on 2026-09-16.
                log("the running agent is out of date: replacing it with the code on disk")
                self.on_status("Updating the agent…")
                if self.stop_any():
                    h = None
            if h is None and port_in_use():
                # Another agent is on the port but not answering yet — it is still
                # starting (start_agent.ps1, or another copy of this app). Found
                # 2026-09-18: both started within three seconds, this one lost the
                # port, and its exit was read as "the user stopped the agent".
                log("the agent's port is already taken: waiting for that agent instead of starting a second one")
                self.on_status("Another agent is starting…")
                deadline = time.monotonic() + OTHER_AGENT_WAIT_SECONDS
                while h is None and time.monotonic() < deadline:
                    time.sleep(1.0)
                    h = health(timeout=3.0)
            if h is not None:
                log("attached to a backend that was already running")
            else:
                if not PYTHON.exists():
                    self.on_status(f"Python environment not found: {PYTHON}")
                    return None
                self._spawn()
                deadline = time.monotonic() + BACKEND_START_SECONDS
                while h is None and time.monotonic() < deadline:
                    if not self.owned:
                        break  # it exited during startup; see the log
                    time.sleep(0.5)
                    h = health()
                self.served = h is not None
            if h is None:
                self.on_status("The agent did not start. Open the log from the tray menu.")
                return None
            self.on_ready(h)
            return h
        finally:
            self._bring_up_lock.release()

    def _spawn(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            if BACKEND_LOG.stat().st_size > 5 * 1024 * 1024:
                BACKEND_LOG.replace(BACKEND_LOG.with_suffix(".log.1"))
        except OSError:
            pass
        # AGENT_PARENT_PID: the backend stops itself if this app is closed or killed.
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
               "AGENT_PARENT_PID": str(os.getpid())}
        if SMOKE_TEST:
            env["WAKE_WORD_ENABLED"] = "false"  # a test run must not open the microphone
        self.code_stamp = backend_code_stamp()
        self.served = False
        with open(BACKEND_LOG, "ab") as out:
            self.process = subprocess.Popen(
                [str(PYTHON), "-m", "app.main"],
                cwd=str(BACKEND_DIR),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                creationflags=CREATE_NO_WINDOW,
            )
        log(f"backend started pid={self.process.pid}")
        threading.Thread(target=self._watch, args=(self.process,), name="backend-watch", daemon=True).start()

    def _watch(self, process: subprocess.Popen) -> None:
        code = process.wait()
        log(f"backend exited code={code}")
        if self.stopping or process is not self.process:
            return
        # Never started again behind the user's back. Found 2026-09-17: the user
        # stopped the agent, the app started it again, and the wake word heard a
        # conversation and ran it as a task.
        self.process = None
        if self.on_exit is not None:
            self.on_exit(code, self.served)

    def stop(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        self.stopping = True
        log("stopping backend")
        # Same clean path as Ctrl+C: running tasks are cancelled and recorded,
        # learning is drained and the database is closed.
        http("POST", "/app/shutdown", timeout=3)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            log("backend did not stop in time, ending it")
            subprocess.run(
                ["taskkill", "/pid", str(process.pid), "/T", "/F"],
                capture_output=True, creationflags=CREATE_NO_WINDOW,
            )
        # Let go of it, so the watcher thread knows this exit was asked for even
        # if it wakes after restart() has cleared `stopping`. Found 2026-09-16: a
        # planned restart logged "The agent stopped unexpectedly, restarting…"
        # and asked for a second start on top of the one already under way.
        self.process = None

    def stop_any(self) -> bool:
        """Stop the running backend, including one this app did not start.

        stop() can only end a process this app holds a handle to. When the app
        attached to a backend started elsewhere (start_agent.ps1, or an earlier
        copy of the app), the only way to replace it with current code is to ask
        it to shut down over HTTP and wait for it to go.
        """
        if self.owned:
            self.stop()
            self.stopping = False
            return True
        if health() is None:
            return True
        log("stopping a backend this app did not start")
        http("POST", "/app/shutdown", timeout=3)
        for _ in range(30):
            time.sleep(0.5)
            if health() is None:
                return True
        log("that backend did not stop; leaving it alone")
        return False

    def restart(self) -> None:
        self.stop()
        self.stopping = False
        self.bring_up()


# =============================================================================
# Native window setup
# =============================================================================

_native_handlers: list = []  # .NET event handlers must stay referenced for the process lifetime


def native_window_setup(window) -> bool:
    """Window icon, and microphone access for the agent page only (its mic button)."""
    try:
        from Microsoft.Web.WebView2.Core import CoreWebView2PermissionKind, CoreWebView2PermissionState
        from System import Action
        from System.Drawing import Icon

        form = window.native

        def on_permission(sender, args):
            if args.PermissionKind == CoreWebView2PermissionKind.Microphone and is_agent_url(str(args.Uri)):
                args.State = CoreWebView2PermissionState.Allow

        def setup():
            if ICON_ICO.exists():
                form.Icon = Icon(str(ICON_ICO))
            form.webview.CoreWebView2.PermissionRequested += on_permission

        _native_handlers.append(on_permission)
        form.Invoke(Action(setup))
        log("window ready: icon set, microphone allowed for the agent page")
        return True
    except Exception as e:
        log(f"window setup skipped: {e}")
        return False


# =============================================================================
# The app: window + tray
# =============================================================================

class DesktopApp:
    def __init__(self) -> None:
        self.status = "Starting…"
        self.window = None
        self.tray = None
        self.quitting = False
        self.showing_loading_page = True
        self.window_hidden = START_HIDDEN
        self.app_stamp = app_code_stamp()
        self._native_ready = False
        self.backend = Backend(self.set_status, self.on_backend_ready, self.on_backend_exit)

    # --- status ---------------------------------------------------------------

    def set_status(self, text: str) -> None:
        if text != self.status:
            log(f"status: {text}")
        self.status = text
        if self.tray is not None:
            self.tray.title = f"{APP_NAME} — {text}"[:127]
            self.tray.update_menu()
        if self.window is not None and self.showing_loading_page:
            try:
                self.window.evaluate_js(f"window.setStatus && window.setStatus({json.dumps(text)})")
            except Exception:
                pass

    def on_backend_ready(self, h: dict) -> None:
        self.set_status(describe_health(h))
        if self.window is not None:
            self.showing_loading_page = False
            self.window.load_url(UI_URL)

    def on_backend_exit(self, code: int, served: bool = True) -> None:
        """The backend ended without the app asking.

        `served` is False when it never answered /health — it could not start,
        almost always because another agent already holds the port. Found
        2026-09-18: this app and start_agent.ps1 started one each within three
        seconds; the loser exited with code 0, which was read as "the user
        stopped the agent", so the app quit and stopped the other one as well.
        """
        if self.quitting:
            return
        other = health()
        if other is not None:
            # Something is still serving: it is not this app's to stop.
            log("this app's agent ended, but another agent is answering: attaching to it")
            self.on_backend_ready(other)
            return
        if not served:
            self.show_not_running(f"The agent could not start. Open the log from the tray menu. {START_HINT}")
            self.notify("The agent could not start", "Open the log from the tray menu to see why.")
            return
        if code == 0:
            # It was stopped on purpose (start_agent.ps1, /app/shutdown): stop everything.
            log("the agent was stopped outside the app: closing the app too")
            self.quit()
            return
        self.show_not_running(f"The agent stopped unexpectedly (exit code {code}). {START_HINT}")
        self.notify("The agent stopped", "It is not listening any more. " + START_HINT)

    def show_not_running(self, text: str) -> None:
        self.set_status(text)
        if self.window is not None and not self.showing_loading_page:
            self.showing_loading_page = True
            self.window.load_html(loading_html())

    def poll_health(self) -> None:
        while not self.quitting:
            time.sleep(15)
            if self.quitting or self.backend.stopping:
                continue
            h = health()
            if h is not None:
                if self.showing_loading_page:
                    # An agent is answering again (started in a console, or by the
                    # tray menu): show it instead of leaving the "not running" page.
                    log("an agent is answering again: attaching to it")
                    self.on_backend_ready(h)
                self.set_status(describe_health(h))
            elif not self.backend.owned:
                # Nothing is running, and nothing is started silently (2026-09-17):
                # the user may have stopped it. Opening the icon starts it again.
                self.show_not_running(f"The agent is not running. {START_HINT}")

    # --- window ---------------------------------------------------------------

    def show_window(self, new_chat: bool = False) -> None:
        """Bring the window up. new_chat=True starts a fresh conversation (the page is
        reloaded), except while a task is running, whose progress would disappear."""
        if self.window is None or self.quitting:
            return
        if new_chat and not self.showing_loading_page and not task_running():
            self.window.load_url(UI_URL)
            log("new chat")
        self.window.show()
        self.window.restore()
        self.window_hidden = False

    def on_show_request(self) -> None:
        """The desktop icon was opened while the app was already running."""
        if self.quitting:
            return
        if app_code_stamp() != self.app_stamp:
            self.restart_app()
            return
        running = health()
        if running is None:
            # The agent is not running (it stopped, or something else stopped it).
            # Opening the icon must bring it back, not just show an empty window.
            log("icon opened while the agent was not running: starting it")
            self.showing_loading_page = True
            self.show_window()
            if self.window is not None:
                self.window.load_html(loading_html())
            threading.Thread(target=self.backend.bring_up, name="start-on-open", daemon=True).start()
            return
        outdated = backend_is_outdated(running) or (
            self.backend.owned
            and bool(self.backend.code_stamp)
            and backend_code_stamp() != self.backend.code_stamp
        )
        if outdated and not bool(running.get("task_running")):
            log("agent code changed on disk: restarting the agent")
            self.show_window()
            threading.Thread(target=self._restart_or_start, name="update-restart", daemon=True).start()
            return
        self.show_window(new_chat=True)

    def restart_app(self) -> None:
        """Hand over to a fresh copy of the app: its code changed on disk since this one started."""
        log("desktop app updated on disk: restarting it")
        self.quitting = True

        def work() -> None:
            # stop_any: the backend may be one this copy attached to rather than
            # started, and leaving it running would hand the fresh copy the same
            # out-of-date agent.
            self.backend.stop_any()
            try:
                subprocess.Popen(
                    [str(PYTHONW), str(Path(__file__).resolve()), "--after-update"],
                    cwd=str(APP_DIR),
                    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                    close_fds=True,
                )
            except OSError as e:
                log(f"could not start the updated app: {e}")
            if self.tray is not None:
                self.tray.stop()
            if self.window is not None:
                self.window.destroy()

        threading.Thread(target=work, name="app-restart", daemon=True).start()

    def on_closing(self) -> bool:
        """
        Closing the window stops everything: the agent, its Chrome and the wake word.

        Found 2026-09-17: closing only hid the window, the agent kept listening
        in the tray, heard people talking and ran what they said as a task.
        Running with no window is now an explicit tray choice (hide_to_tray).
        """
        if self.quitting:
            return True
        self.quit()
        return False  # quit() closes the window once the agent has stopped

    def backend_is_up(self) -> bool:
        """True when there is a running agent to talk to."""
        return port_in_use(API_PORT)

    def talk(self) -> None:
        """Start a spoken conversation without saying the wake word.

        Emma starts listening the moment this is chosen; say what you want and
        finish with "done". A question comes back as a spoken answer, and a job
        is carried out and then reported.
        """
        status, body = http("POST", "/talk", timeout=6.0)
        if status == 200 and '"ok": true' in body.replace(" ", "").replace("'", '"').lower():
            self.set_status("Listening - say what you want, then say \"done\"")
            log("talk requested")
            return
        if status is None:
            message_box(
                "The agent is not running, so there is nothing listening yet.\n\n"
                + START_HINT
            )
            return
        message_box(
            "Emma could not start listening.\n\n"
            "The wake word listener may be switched off, or another task is using "
            "the microphone. Open the log from the tray menu to see why."
        )

    def hide_to_tray(self) -> None:
        if self.window is None or self.quitting:
            return
        self.window_hidden = True
        threading.Thread(target=self.window.hide, daemon=True).start()
        self.notify(
            "Still listening",
            f'Say "{wake_word().capitalize()}" to give it a task. Choose Quit in this tray icon\'s menu to stop it.',
        )

    def notify(self, title: str, text: str) -> None:
        if self.tray is None:
            return
        try:
            self.tray.notify(text, title)
        except Exception:
            pass

    def on_loaded(self) -> None:
        if not self._native_ready:
            self._native_ready = True
            native_window_setup(self.window)
        if self.showing_loading_page:
            self.set_status(self.status)

    # --- tray -----------------------------------------------------------------

    def start_tray(self) -> None:
        import pystray
        from PIL import Image

        item = pystray.MenuItem
        menu = pystray.Menu(
            # A window that was hidden in the tray reopens with a new chat.
            item("Open Agent", lambda icon, it: self.show_window(new_chat=self.window_hidden), default=True),
            item(lambda it: self.status, None, enabled=False),
            pystray.Menu.SEPARATOR,
            item("Talk to Emma", lambda icon, it: threading.Thread(target=self.talk, daemon=True).start(),
                 enabled=lambda it: self.backend_is_up()),
            item("Hide window, keep listening", lambda icon, it: self.hide_to_tray(),
                 enabled=lambda it: not self.window_hidden),
            item("Start with Windows", lambda icon, it: set_start_with_windows(not starts_with_windows()),
                 checked=lambda it: starts_with_windows()),
            item(lambda it: "Restart agent" if self.backend.owned else "Start agent",
                 lambda icon, it: threading.Thread(target=self._restart_or_start, daemon=True).start()),
            item("Open log", lambda icon, it: os.startfile(BACKEND_LOG if BACKEND_LOG.exists() else LOG_DIR)),
            pystray.Menu.SEPARATOR,
            item("Quit (stops the agent and the wake word)", lambda icon, it: self.quit()),
        )
        image = Image.open(ICON_PNG) if ICON_PNG.exists() else Image.new("RGB", (64, 64), (79, 140, 255))
        self.tray = pystray.Icon("self-improving-agent", image, f"{APP_NAME} — {self.status}", menu)
        self.tray.run_detached()

    def _restart_or_start(self) -> None:
        self.showing_loading_page = True
        if self.window is not None:
            self.window.load_html(loading_html())
        if self.backend.owned:
            self.backend.restart()
        else:
            # A backend started elsewhere has to be asked to stop; otherwise
            # bring_up() simply attaches to the same out-of-date agent again.
            self.backend.stop_any()
            self.backend.bring_up()

    def quit(self) -> None:
        if self.quitting:
            return
        self.quitting = True
        self.set_status("Stopping…")
        if self.window is not None:
            threading.Thread(target=self.window.hide, daemon=True).start()

        def work() -> None:
            # stop_any: also an agent this app attached to rather than started.
            # Found 2026-09-17: Quit left such an agent running and listening.
            self.backend.stop_any()
            if self.tray is not None:
                self.tray.stop()
            if self.window is not None:
                self.window.destroy()

        threading.Thread(target=work, name="quit", daemon=True).start()

    # --- run ------------------------------------------------------------------

    def after_gui_start(self) -> None:
        self.start_tray()
        # Opening the desktop icon again starts a new chat (and restarts outdated code);
        # start_agent.ps1 and "--quit" ask this copy to stop everything.
        listen_for_requests(self.on_show_request, self.quit)
        if START_HIDDEN:
            self.notify(
                "Self-Improving Agent is running",
                f'Listening for "{wake_word().capitalize()}" in the background. Choose Quit in this tray icon\'s menu to stop it.',
            )
        self.backend.bring_up()
        threading.Thread(target=self.poll_health, name="health", daemon=True).start()

    def run(self) -> None:
        import webview

        WEBVIEW_DATA.mkdir(parents=True, exist_ok=True)
        self.window = webview.create_window(
            APP_NAME, html=loading_html(), width=520, height=800, min_size=(380, 520),
            background_color="#0f1117", hidden=START_HIDDEN,
        )
        self.window.events.closing += self.on_closing
        self.window.events.loaded += self.on_loaded
        # Not private: the agent page's microphone permission and settings persist.
        webview.start(self.after_gui_start, gui="edgechromium", private_mode=False, storage_path=str(WEBVIEW_DATA))


# =============================================================================
# Smoke test: backend up, page served, WebView2 renders it, clean stop
# =============================================================================

def _webview_check() -> dict:
    """Render the agent page in a hidden WebView2 window and run the native setup on it."""
    import webview

    seen = {"webview_rendered": False, "native_window_setup": False}
    window = webview.create_window("smoke test", UI_URL, hidden=True)

    def check() -> None:
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            try:
                if "Local Agent" in (window.evaluate_js("document.title") or ""):
                    seen["webview_rendered"] = True
                    seen["native_window_setup"] = native_window_setup(window)
                    break
            except Exception:
                pass
            time.sleep(0.5)
        window.destroy()

    webview.start(check, gui="edgechromium", private_mode=True)
    return seen


def _tray_starts() -> bool:
    import pystray
    from PIL import Image

    image = Image.open(ICON_PNG) if ICON_PNG.exists() else Image.new("RGB", (64, 64), (79, 140, 255))
    icon = pystray.Icon("smoke-test", image, "smoke test")
    try:
        icon.run_detached()
        time.sleep(1.0)
        return bool(icon.visible)
    finally:
        icon.stop()


def run_smoke_test() -> int:
    result = {"backend_ready": False, "started_backend": False, "ui_page_served": False,
              "webview_rendered": False, "native_window_setup": False, "tray_started": False,
              "backend_exit_code": None}
    backend = Backend(lambda text: log(f"smoke test: {text}"), lambda h: None)
    try:
        h = backend.bring_up()
        result["backend_ready"] = h is not None
        result["started_backend"] = backend.owned
        if h is not None:
            status, body = http("GET", "/local", timeout=10)
            result["ui_page_served"] = status == 200 and "Local Agent" in body
            result.update(_webview_check())
            result["tray_started"] = _tray_starts()
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        process = backend.process
        backend.stop()
        result["backend_exit_code"] = process.returncode if process is not None else None
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "desktop-smoke-test.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    ok = all(result[key] for key in ("backend_ready", "ui_page_served", "webview_rendered",
                                     "native_window_setup", "tray_started"))
    return 0 if ok else 1


# =============================================================================
# Entry point
# =============================================================================

def quit_everything() -> int:
    """--quit: stop the running app and any agent, from a script or a shortcut."""
    if send_to_running_instance(b"quit"):
        log("asked the running app to quit")
    backend = Backend(lambda text: None, lambda h: None)
    stopped = backend.stop_any()
    return 0 if stopped else 1


def main() -> int:
    if SMOKE_TEST:
        return run_smoke_test()
    if QUIT:
        return quit_everything()
    # After an update the old copy is still shutting down: wait for it.
    if not acquire_single_instance(wait_seconds=30.0 if AFTER_UPDATE else 0.0):
        ask_running_instance_to_show()
        return 0
    try:
        import pystray  # noqa: F401
        import webview  # noqa: F401
    except ImportError:
        message_box("The desktop app is not installed yet.\n\nRun:\n"
                    "backend\\.venv\\Scripts\\python.exe backend\\scripts\\install_desktop_app.py")
        return 1
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception:
        pass
    log("app starting" + (" hidden" if START_HIDDEN else ""))
    DesktopApp().run()
    log("app closed")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception:
        log("crashed:\n" + traceback.format_exc())
        message_box(f"The desktop app stopped with an error. Details are in:\n{DESKTOP_LOG}")
        exit_code = 1
    # Tray, health and watcher threads must not keep the process alive.
    os._exit(exit_code)
