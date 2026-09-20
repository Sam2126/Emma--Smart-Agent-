"""
Spoken feedback, the agent staying out of the user's own Chrome, and the tools
added on 2026-09-17: download_file, add_to_cart and install_app.
"""

from __future__ import annotations

import asyncio
import base64
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agent.toolkit import output_indicates_failure
from app.tools import browser as browser_tools
from app.tools import desktop, local
from app.voice_feedback import VoiceFeedback, parse_voice_feedback


# =============================================================================
# Spoken feedback
# =============================================================================

@pytest.mark.parametrize("spoken, rating, note", [
    ("Good job.", 1, ""),
    ("good job", 1, ""),
    ("Perfect, thank you!", 1, "thank you!"),
    ("Yes, correct.", 1, ""),
    ("It worked.", 1, ""),
    ("Wrong.", -1, ""),
    ("Wrong, you opened my own Chrome profile.", -1, "you opened my own Chrome profile."),
    ("wrong you should use install_app", -1, "you should use install_app"),
    ("No, that's wrong because it used Person 1.", -1, "because it used Person 1."),
    ("That was wrong. Next time open the agent's window.", -1, "Next time open the agent's window."),
    ("Not correct, the size was M.", -1, "the size was M."),
    ("It didn't work, nothing was downloaded.", -1, "nothing was downloaded."),
    ("Feedback good", 1, ""),
    ("Feedback: bad, use the Microsoft Store tool.", -1, "use the Microsoft Store tool."),
    ("Thumbs down, it added the wrong shoe.", -1, "it added the wrong shoe."),
])
def test_spoken_ratings_are_recognised(spoken, rating, note):
    assert parse_voice_feedback(spoken) == VoiceFeedback(rating, note)


@pytest.mark.parametrize("spoken", [
    "Correct the spelling in notes.txt",
    "correct it",
    "Wrong number on the invoice, fix it",
    "open WhatsApp and say good job to Rakesh",
    "search for perfect shoes on Myntra",
    "Add this to my cart",
    "okay",
    "",
])
def test_tasks_are_not_mistaken_for_feedback(spoken):
    assert parse_voice_feedback(spoken) is None


def test_feedback_without_a_rating_asks_for_one():
    assert parse_voice_feedback("feedback, hmm") == VoiceFeedback(0, "")


def test_the_last_finished_task_is_remembered():
    from app.agent import runner

    runner._remember_finished("t-1", "open notepad", True)
    last = runner.last_finished_task()
    assert last["task_id"] == "t-1" and last["instruction"] == "open notepad" and last["success"] is True
    last["task_id"] = "changed"
    assert runner.last_finished_task()["task_id"] == "t-1"  # a copy is handed out


async def test_spoken_rating_goes_to_the_last_task(monkeypatch):
    from app import voice_feedback
    from app.agent import runner, services

    recorded = []

    async def fake_record(task_id, rating, comment=""):
        recorded.append((task_id, rating, comment))
        return {"applied": True, "queued": False, "message": "Got it."}

    monkeypatch.setattr(services.brain, "record_feedback", fake_record)
    monkeypatch.setattr(runner, "_last_finished", {
        "task_id": "t-9", "instruction": "add shoes to my cart", "success": True, "finished_at": 1000.0,
    })
    result = await voice_feedback.record_voice_feedback(VoiceFeedback(-1, "it added the wrong shoe"), now=1060.0)
    assert recorded == [("t-9", -1, "it added the wrong shoe")]
    assert result["applied"] and result["task_id"] == "t-9" and result["instruction"] == "add shoes to my cart"


async def test_no_recent_task_means_nothing_is_rated(monkeypatch):
    from app import voice_feedback
    from app.agent import runner, services

    async def must_not_record(*args, **kwargs):
        raise AssertionError("nothing should be recorded")

    monkeypatch.setattr(services.brain, "record_feedback", must_not_record)
    monkeypatch.setattr(runner, "_last_finished", {
        "task_id": "old", "instruction": "x", "success": True, "finished_at": 0.0,
    })
    result = await voice_feedback.record_voice_feedback(VoiceFeedback(1, ""), now=voice_feedback.FEEDBACK_WINDOW_SECONDS + 1)
    assert not result["applied"] and not result["task_id"]
    assert "15 minutes" in result["message"]

    monkeypatch.setattr(runner, "_last_finished", None)
    assert not (await voice_feedback.record_voice_feedback(VoiceFeedback(1, "")))["applied"]
    unclear = await voice_feedback.record_voice_feedback(VoiceFeedback(0, ""))
    assert "feedback good" in unclear["message"]


async def test_wake_word_feedback_is_saved_and_shown_in_every_window(monkeypatch):
    from app import voice_feedback, wake_listener
    from app.websocket.protocol import FeedbackRecordedMessage, VoiceTranscriptMessage

    sent, beeps = [], []

    async def fake_record(feedback):
        return {"applied": True, "queued": False, "task_id": "t-5", "instruction": "open chrome", "message": "Got it."}

    monkeypatch.setattr(voice_feedback, "record_voice_feedback", fake_record)
    monkeypatch.setattr(wake_listener, "_ui_broadcaster", lambda: sent.append)
    monkeypatch.setattr(wake_listener, "_beep_feedback", lambda rating: beeps.append(rating))
    monkeypatch.setattr(wake_listener, "_beep_error", lambda: beeps.append("error"))

    listener = wake_listener.WakeWordListener(wake_word="hello", stop_word="done")
    await listener._async_feedback(VoiceFeedback(-1, "you opened my Chrome"), "Wrong, you opened my Chrome.")

    assert isinstance(sent[0], VoiceTranscriptMessage) and sent[0].text == "Wrong, you opened my Chrome."
    assert isinstance(sent[1], FeedbackRecordedMessage)
    assert (sent[1].task_id, sent[1].rating, sent[1].applied) == ("t-5", -1, True)
    assert beeps == [-1]


async def test_mic_button_feedback_rates_instead_of_running_a_task(monkeypatch):
    from app import voice, voice_feedback
    from app.websocket.protocol import VoiceTaskMessage
    from app.websocket.server import WebSocketServer

    async def fake_transcribe(audio, filename="", **options):
        # The window's microphone now asks Whisper for English and gives it the
        # vocabulary, after a real run turned spoken English into Korean.
        assert options.get("language") == "en"
        assert "Myntra" in (options.get("prompt") or "")
        return "Good job."

    async def fake_record(feedback):
        assert feedback == VoiceFeedback(1, "")
        return {"applied": True, "queued": False, "task_id": "t-7", "message": "Thanks."}

    monkeypatch.setattr(voice, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(voice_feedback, "record_voice_feedback", fake_record)

    frames: list[str] = []

    class _Socket:
        async def send(self, data):
            frames.append(data)

    server = WebSocketServer()
    socket = _Socket()
    server._clients.add(socket)

    async def no_task(*args, **kwargs):
        raise AssertionError("feedback must not start a task")

    monkeypatch.setattr(server, "_handle_task_submit", no_task)
    await server._handle_voice_task(socket, VoiceTaskMessage(audio_base64=base64.b64encode(b"x" * 2000).decode(), scope="local"))
    assert any('"voice_transcript"' in f for f in frames)
    assert any('"feedback_recorded"' in f and '"t-7"' in f for f in frames)


# =============================================================================
# The agent's Chrome only: desktop tools and shell commands keep out of the user's
# =============================================================================

def _no_powershell(*args, **kwargs):
    raise AssertionError("nothing may be sent to Windows for a Chrome window")


@pytest.mark.parametrize("hint", ["Chrome", "Google Chrome", "chrome.exe", "browser", "Chrome - Gemini"])
def test_window_tools_refuse_chrome(monkeypatch, hint):
    monkeypatch.setattr(desktop, "_ps", _no_powershell)
    outputs = {
        "focus_window": desktop.FocusWindowTool().run(window=hint),
        "see_window": desktop.SeeWindowTool().run(window=hint),
        "click_window": desktop.ClickWindowTool().run(x=10, y=10, window=hint),
        "send_keys": local.SendKeysTool().run(window_hint=hint, text="hello"),
    }
    for tool, out in outputs.items():
        assert out.startswith("Refused:"), (tool, out)
        assert "browser tools" in out
        assert output_indicates_failure(out, tool) is True


@pytest.mark.parametrize("hint", ["WhatsApp", "Gemini", "Notepad"])
def test_other_windows_are_not_refused(hint):
    assert desktop.chrome_window_refusal(hint) is None


def test_the_window_finder_skips_chrome_windows():
    assert 'if (proc == "chrome") return true;' in desktop._PS_UI_HELPER


@pytest.mark.parametrize("name", ["chrome", "Google Chrome", "chrome.exe", "browser", "Chrome browser"])
def test_open_app_chrome_opens_the_agents_window(monkeypatch, name):
    from app.browser import registry

    brought_front = []

    async def fake_connect():
        return True

    async def open_agent_tab():
        async def bring_to_front():
            brought_front.append(True)
        return SimpleNamespace(bring_to_front=bring_to_front)

    monkeypatch.setitem(sys.modules, "app.main", SimpleNamespace(ensure_browser_connection=fake_connect))
    monkeypatch.setattr(registry, "get_browser_controller", lambda: SimpleNamespace(open_agent_tab=open_agent_tab))
    monkeypatch.setattr(local.subprocess, "run", _no_powershell)
    monkeypatch.setattr(local.subprocess, "Popen", _no_powershell)

    out = local.OpenAppTool().run(name=name)
    assert out.startswith("Launched: the agent's own Chrome window")
    assert "not used" in out and brought_front == [True]
    assert output_indicates_failure(out, "open_app") is False


def test_open_app_reports_when_the_agents_chrome_cannot_start(monkeypatch):
    async def fake_connect():
        return False

    monkeypatch.setitem(sys.modules, "app.main", SimpleNamespace(ensure_browser_connection=fake_connect))
    out = local.OpenAppTool().run(name="chrome")
    assert output_indicates_failure(out, "open_app") is True


def test_open_app_chrome_is_not_reported_open_when_no_window_came_up(monkeypatch):
    """Found 2026-09-17 in the log: 'No browser context available', yet 'Launched:'."""
    from app.browser import registry

    async def fake_connect():
        return True

    async def open_agent_tab():
        raise RuntimeError("No browser context available to open an agent tab.")

    monkeypatch.setitem(sys.modules, "app.main", SimpleNamespace(ensure_browser_connection=fake_connect))
    monkeypatch.setattr(registry, "get_browser_controller", lambda: SimpleNamespace(open_agent_tab=open_agent_tab))
    out = local.OpenAppTool().run(name="chrome")
    assert out.startswith("Could not open the agent's Chrome window") and "No browser context" in out
    assert output_indicates_failure(out, "open_app") is True


async def test_agent_tab_reconnects_when_the_window_was_closed(monkeypatch):
    from app.browser.controller import BrowserController

    opened = []

    async def new_page():
        page = SimpleNamespace(is_closed=lambda: False, bring_to_front=_async_none, on=lambda *a: None)
        opened.append(page)
        return page

    fresh = SimpleNamespace(contexts=[SimpleNamespace(new_page=new_page, pages=[], on=lambda *a: None)],
                            is_connected=lambda: True)
    controller = BrowserController()
    controller._connected = True
    controller._browser = SimpleNamespace(contexts=[], is_connected=lambda: False)  # the user closed it

    async def connect(endpoint, allow_launch=True):
        controller._browser, controller._connected = fresh, True

    monkeypatch.setattr(controller, "connect", connect)
    monkeypatch.setattr(controller, "_watch_downloads", lambda: None)
    page = await controller.open_agent_tab()
    assert page is opened[0] and controller._browser is fresh


async def _async_none(*args, **kwargs):
    return None


def test_web_addresses_are_not_opened_in_the_users_browser(monkeypatch):
    monkeypatch.setattr(local.os, "startfile", _no_powershell, raising=False)
    for address in ("https://gemini.google.com", "www.youtube.com", "http://example.com/a.pdf"):
        out = local.OpenPathTool().run(path=address)
        assert out.startswith("Not opened:") and "navigate_browser" in out
        assert output_indicates_failure(out, "open_file_or_folder") is True


@pytest.mark.parametrize("command", [
    "Start-Process chrome",
    "start chrome https://gemini.google.com",
    'Start-Process "https://gemini.google.com"',
    'cmd /c start "" "https://example.com"',
    '& "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --new-window',
    "Stop-Process -Name chrome -Force",
    "taskkill /im chrome.exe /f",
    "explorer.exe https://youtube.com",
    '[System.Diagnostics.Process]::Start("https://example.com")',
    "Start-Process www.google.com",
])
def test_commands_that_touch_the_users_chrome_are_not_run(monkeypatch, command):
    monkeypatch.setattr(local, "_run_powershell", _no_powershell)
    out = local.RunCommandTool().run(command=command)
    assert out.startswith("Not run:") and "navigate_browser" in out


@pytest.mark.parametrize("command", [
    "Get-Process chrome | Select-Object -First 3",
    "Start-Process notepad",
    'Get-Date -Format "yyyy-MM-dd"',
    'Start-Process "ms-settings:sound"',
    "Invoke-WebRequest https://example.com -OutFile page.html",
    'Get-ChildItem "C:\\Users\\me\\AppData\\Local\\Google\\Chrome"',
    "start msedge",
])
def test_everyday_commands_still_run(command):
    assert local._TOUCHES_USERS_CHROME.search(command) is None


# =============================================================================
# download_file and add_to_cart in a real (headless) browser
# =============================================================================

PAGES = {
    "/files": '<html><head><title>Files</title></head><body><a id="dl" href="/file/report.pdf">Download report</a>'
              '<a id="page" href="/files">Not a file</a></body></html>',
    "/list": '<html><head><title>Shoes</title></head><body>'
             '<a id="p1" href="/product/shoe" target="_blank">Running Shoe</a></body></html>',
    "/product/shoe": """<html><head><title>Running Shoe</title></head><body>
<a href="/cart">Bag <span class="cart-count">0</span></a>
<div class="size-buttons"><button class="size">S</button><button class="size">M</button><button class="size">L</button></div>
<p id="msg"></p>
<button class="buy-now">BUY NOW</button>
<button class="pdp-add-to-bag">ADD TO BAG</button>
<script>
let size = null;
document.querySelectorAll('.size').forEach(b => b.onclick = () => { size = b.innerText; });
document.querySelector('.pdp-add-to-bag').onclick = () => {
  if (!size) { document.getElementById('msg').innerText = 'Please select a size'; return; }
  const c = document.querySelector('.cart-count');
  c.innerText = String(Number(c.innerText) + 1);
  document.getElementById('msg').innerText = 'Added to bag';
};
document.querySelector('.buy-now').onclick = () => { document.title = 'BOUGHT'; };
</script></body></html>""",
}
FILE_BODY = b"%PDF-1.4 quarterly report"


class _Site(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/file/"):
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Disposition", f'attachment; filename="{self.path.rsplit("/", 1)[-1]}"')
            self.send_header("Content-Length", str(len(FILE_BODY)))
            self.end_headers()
            self.wfile.write(FILE_BODY)
            return
        body = PAGES.get(self.path, "<html><body>missing</body></html>").encode()
        self.send_response(200 if self.path in PAGES else 404)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
async def agent_browser(monkeypatch, tmp_path):
    from playwright.async_api import async_playwright

    from app.browser import actions
    from app.browser import controller as ctl
    from app.utils import loop as loop_module

    downloads = tmp_path / "Downloads"
    monkeypatch.setattr(local, "_known_folder", lambda name: downloads if name == "downloads" else tmp_path / name)
    monkeypatch.setattr(loop_module, "_main_loop", asyncio.get_running_loop())

    async def no_delay():
        return None

    monkeypatch.setattr(actions, "_human_delay", no_delay)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Site)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(accept_downloads=True)
        controller = ctl.BrowserController()
        controller._browser = browser
        controller._connected = True
        controller._agent_page = await context.new_page()
        monkeypatch.setattr(browser_tools, "get_browser_controller", lambda: controller)
        yield controller, base, downloads
        await browser.close()
    server.shutdown()


async def _run(tool, **kwargs) -> str:
    """Like the engine: the tool runs in a worker thread, its body on this loop."""
    return await asyncio.to_thread(tool.run, **kwargs)


async def test_a_download_lands_in_the_downloads_folder(agent_browser):
    controller, base, downloads = agent_browser
    page = await controller.get_active_page()
    await page.goto(f"{base}/files")

    first = await _run(browser_tools.DownloadFileTool(), selector="#dl")
    assert first.startswith("Downloaded: ") and output_indicates_failure(first, "download_file") is False
    assert (downloads / "report.pdf").read_bytes() == FILE_BODY

    second = await _run(browser_tools.DownloadFileTool(), url=f"{base}/file/report.pdf")
    assert second.startswith("Downloaded: ")
    assert (downloads / "report (1).pdf").read_bytes() == FILE_BODY


async def test_a_download_started_by_a_plain_click_is_saved_too(agent_browser):
    controller, base, downloads = agent_browser
    page = await controller.get_active_page()
    await page.goto(f"{base}/files")

    out = await _run(browser_tools.ClickElementTool(), selector="#dl")
    assert "started a download" in out and "report.pdf" in out
    deadline = time.monotonic() + 10
    while controller.downloads[-1]["status"] == "downloading" and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    assert controller.downloads[-1]["status"] == "saved"
    assert (downloads / "report.pdf").read_bytes() == FILE_BODY


async def test_a_link_that_is_not_a_file_is_reported(agent_browser, monkeypatch):
    controller, base, downloads = agent_browser
    monkeypatch.setattr(browser_tools, "_DOWNLOAD_START_SECONDS", 1.0)
    page = await controller.get_active_page()
    await page.goto(f"{base}/files")
    out = await _run(browser_tools.DownloadFileTool(), selector="#page")
    assert out.startswith("No download started") and output_indicates_failure(out, "download_file") is True
    assert not downloads.exists() or not any(downloads.iterdir())


async def test_add_to_cart_asks_for_a_size_then_adds(agent_browser):
    controller, base, _ = agent_browser
    page = await controller.get_active_page()
    await page.goto(f"{base}/list")

    # The product opens in a new tab; the agent must keep working there.
    first = await _run(browser_tools.AddToCartTool(), product_selector="#p1")
    assert first.startswith("Not added yet") and "size" in first
    assert "S, M, L" in first
    assert output_indicates_failure(first, "add_to_cart") is True
    active = await controller.get_active_page()
    assert active.url.endswith("/product/shoe")

    second = await _run(browser_tools.AddToCartTool(), option="M")
    assert second.startswith("Added to cart: 'Running Shoe'"), second
    assert "0 → 1" in second
    assert output_indicates_failure(second, "add_to_cart") is False
    assert await active.title() == "Running Shoe"  # Buy Now was never pressed
    assert await active.locator(".cart-count").inner_text() == "1"


async def test_add_to_cart_without_a_button_adds_nothing(agent_browser):
    controller, base, _ = agent_browser
    page = await controller.get_active_page()
    await page.goto(f"{base}/files")
    out = await _run(browser_tools.AddToCartTool())
    assert out.startswith("No 'Add to Cart'") and output_indicates_failure(out, "add_to_cart") is True


def test_downloaded_file_names_are_safe_and_unique(monkeypatch, tmp_path):
    from app.browser import controller as ctl

    monkeypatch.setattr(local, "_known_folder", lambda name: tmp_path)
    (tmp_path / "report.pdf").write_bytes(b"x")
    assert ctl._download_target("report.pdf", set()) == tmp_path / "report (1).pdf"
    assert ctl._download_target("report.pdf", {str(tmp_path / "report (1).pdf")}) == tmp_path / "report (2).pdf"
    assert ctl._download_target("..\\..\\evil<name>.exe", set()) == tmp_path / "evil_name_.exe"
    assert ctl._download_target("", set()) == tmp_path / "download"


# =============================================================================
# install_app (winget)
# =============================================================================

WINGET_SEARCH = (
    "   - \r   \\ \r"
    "Name                            Id                            Version         Source\r\n"
    "--------------------------------------------------------------------------------------\r\n"
    "WhatsApp                        9NKSQGP7F2NH                  Unknown         msstore\r\n"
    "WhatsApp Beta                   9NBDXK71NK08                  Unknown         msstore\r\n"
    "WhatsappTray                    D4koon.WhatsappTray           1.9.0.0         winget\r\n"
)
WINGET_STORE_ONLY = (
    "Name                         Id           Version\n"
    "-------------------------------------------------\n"
    "Spotify - Music and Podcasts 9NCBCSZSJRSB Unknown\n"
)
WINGET_LIST = (
    "Name     Id           Version      Source\n"
    "-------------------------------------------\n"
    "WhatsApp 9NKSQGP7F2NH 2.2635.100.0 msstore\n"
)


def test_winget_tables_are_read():
    from app.tools import apps

    rows = apps.parse_winget_table(WINGET_SEARCH)
    assert [r["id"] for r in rows] == ["9NKSQGP7F2NH", "9NBDXK71NK08", "D4koon.WhatsappTray"]
    assert rows[0] == {"name": "WhatsApp", "id": "9NKSQGP7F2NH", "version": "Unknown", "source": "msstore"}
    assert apps.parse_winget_table(WINGET_STORE_ONLY)[0]["name"] == "Spotify - Music and Podcasts"
    assert apps.parse_winget_table("No package found matching input criteria.") == []


def test_the_best_matching_package_is_picked():
    from app.tools import apps

    rows = apps.parse_winget_table(WINGET_SEARCH)
    assert apps.pick_package(rows, "whatsapp")["id"] == "9NKSQGP7F2NH"
    assert apps.pick_package(rows, "WhatsApp B")["id"] == "9NBDXK71NK08"
    assert apps.pick_package(apps.parse_winget_table(WINGET_STORE_ONLY), "spotify")["id"] == "9NCBCSZSJRSB"
    assert apps.pick_package([], "x") is None


def _fake_winget(installed: set[str]):
    def run(args, timeout=90):
        if args[0] == "search":
            return 0, WINGET_SEARCH
        if args[0] == "list":
            package_id = args[args.index("--id") + 1]
            return (0, WINGET_LIST) if package_id in installed else (1, "No installed package found.")
        raise AssertionError(args)
    return run


def test_an_installed_app_is_reported_not_reinstalled(monkeypatch):
    from app.tools import apps

    monkeypatch.setattr(apps, "_winget", _fake_winget({"9NKSQGP7F2NH"}))
    monkeypatch.setattr(apps, "_install", lambda *a: (_ for _ in ()).throw(AssertionError("no install")))
    out = apps.InstallAppTool().run(name="WhatsApp")
    assert out.startswith("Already installed: WhatsApp (9NKSQGP7F2NH)")
    assert output_indicates_failure(out, "install_app") is False


def test_install_asks_first_and_respects_no(monkeypatch):
    from app.tools import apps
    from app.utils import confirmation

    asked = []

    class _Confirmer:
        async def request(self, description, details):
            asked.append(description)
            return False

    monkeypatch.setattr(confirmation, "get_confirmer", lambda: _Confirmer())
    monkeypatch.setattr(apps, "_winget", _fake_winget(set()))
    monkeypatch.setattr(apps, "_install", lambda *a: (_ for _ in ()).throw(AssertionError("no install")))
    out = apps.InstallAppTool().run(name="WhatsApp", source="store")
    assert asked == ["Install 'WhatsApp' from the Microsoft Store on this computer"]
    assert "declined" in out and output_indicates_failure(out, "install_app") is True


def test_install_runs_winget_and_confirms(monkeypatch):
    from app.tools import apps

    installed: set[str] = set()
    calls = []

    def fake_install(package_id, source):
        calls.append((package_id, source))
        installed.add(package_id)
        return 0, "Successfully installed"

    monkeypatch.setattr(apps, "_winget", _fake_winget(installed))
    monkeypatch.setattr(apps, "_install", fake_install)
    out = apps.InstallAppTool().run(name="WhatsApp")
    assert calls == [("9NKSQGP7F2NH", "msstore")]
    assert out.startswith("Installed: WhatsApp (9NKSQGP7F2NH) from the Microsoft Store")
    assert output_indicates_failure(out, "install_app") is False


def test_install_failures_are_reported(monkeypatch):
    from app.tools import apps

    monkeypatch.setattr(apps, "_winget", _fake_winget(set()))
    monkeypatch.setattr(apps, "_install", lambda package_id, source: (1, "Installer failed with exit code: 1603"))
    failed = apps.InstallAppTool().run(name="WhatsApp")
    assert "exit code 1" in failed and output_indicates_failure(failed, "install_app") is True

    monkeypatch.setattr(apps, "_install", lambda package_id, source: (None, ""))
    running = apps.InstallAppTool().run(name="WhatsApp")
    assert running.startswith("Still installing") and output_indicates_failure(running, "install_app") is True

    def no_winget(args, timeout=90):
        raise FileNotFoundError("winget")

    monkeypatch.setattr(apps, "_winget", no_winget)
    missing = apps.InstallAppTool().run(name="WhatsApp")
    assert "winget" in missing and output_indicates_failure(missing, "install_app") is True

    monkeypatch.setattr(apps, "_winget", lambda args, timeout=90: (1, "No package found matching input criteria."))
    unknown = apps.InstallAppTool().run(name="Nonexistent App")
    assert unknown.startswith("Could not find") and output_indicates_failure(unknown, "install_app") is True


# =============================================================================
# The engine knows the new tools
# =============================================================================

def test_new_tools_are_registered_and_classified():
    from app.agent import toolkit
    from app.state.brain import local_skill_type_for

    local_scope = toolkit.build_toolset("local")
    browser_scope = toolkit.build_toolset("browser")
    assert {"install_app", "download_file", "add_to_cart"} <= set(local_scope)
    assert {"download_file", "add_to_cart"} <= set(browser_scope) and "install_app" not in browser_scope
    assert set(toolkit.BROWSER_TOOL_NAMES) == {t.name for t in toolkit.browser_tools()}
    assert toolkit.instruction_needs_web("add the black shoes to my cart")
    assert local_skill_type_for(["navigate_browser", "add_to_cart"]) == "add_to_cart"
    assert local_skill_type_for(["install_app", "open_app"]) == "install_app"
    assert local_skill_type_for(["navigate_browser", "download_file"]) == "file_download"
    for name in ("install_app", "download_file", "add_to_cart"):
        assert name not in toolkit.SAFE_TO_REPEAT_TOOLS  # they change something: no blind retry


def test_prompts_teach_the_new_tools():
    from app.agent import prompts

    local_prompt = prompts.actor_system_prompt("local")
    for phrase in ("install_app", "download_file", "add_to_cart", "CHROME IS THE AGENT'S OWN WINDOW"):
        assert phrase in local_prompt
    browser_prompt = prompts.actor_system_prompt("browser")
    assert "download_file" in browser_prompt and "add_to_cart" in browser_prompt
