"""
From the 2026-09-15 run "open gemini and send him the last downloads ppt ...
and ask him to explain that ppt":

  * there was no tool to attach a local file to a web page -> upload_file
  * the report "... is not available ... I cannot attach ... cannot be
    completed" was shown as COMPLETED -> honest failure detection
  * see_window('Gemini') found no window, captured the whole screen and the
    agent clicked on it blindly -> a missing window is reported, not captured
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agent.toolkit import (
    BROWSER_TOOL_NAMES,
    build_toolset,
    describe_action,
    output_indicates_failure,
    report_indicates_failure,
)
from app.tools import browser as browser_tools


# =============================================================================
# Honest reports
# =============================================================================

def test_a_report_that_says_it_cannot_is_a_failure():
    report = (
        "I attempted to follow the plan, but the required tool to upload a file to Gemini "
        "(upload_file_to_element) is not available in the current toolset. Without that capability, "
        "I cannot attach the PPT file to the Gemini chat. Therefore, the task cannot be completed as specified."
    )
    assert report_indicates_failure(report) is True
    assert report_indicates_failure("It didn't work: the button was disabled.") is True
    assert report_indicates_failure("I wasn't able to open the chat.") is True
    assert report_indicates_failure("Uploaded the file and sent the message. No errors occurred.") is False


def test_action_output_with_cannot_is_a_failure():
    assert output_indicates_failure("Cannot find path 'C:\\nope' because it does not exist.", "run_command") is True


# =============================================================================
# upload_file: registration and safety
# =============================================================================

def test_upload_file_is_a_browser_tool_everywhere():
    assert "upload_file" in BROWSER_TOOL_NAMES
    assert "upload_file" in build_toolset("browser")
    assert "upload_file" in build_toolset("local")
    assert describe_action("upload_file", {"path": r"C:\Users\x\Downloads\Deck.pptx"}) == "📎 Attaching 'Deck.pptx' to the page"


@pytest.mark.parametrize("path,private", [
    (r"C:\Users\x\Downloads\System Design.pptx", False),
    (r"C:\Users\x\Documents\report.pdf", False),
    (r"C:\Users\x\.ssh\id_rsa", True),
    (r"C:\project\.env", True),
    (r"C:\Users\x\Documents\passwords.xlsx", True),
    (r"C:\Users\x\certs\server.pem", True),
    (r"C:\Users\x\AppData\Local\Google\Chrome\User Data\Default\Login Data", True),
    (r"C:\Users\x\.aws\config", True),
    # Ordinary files in AppData / Temp (chat exports, attachments) still upload.
    (r"C:\Users\x\AppData\Local\Temp\WhatsApp Chat.zip", False),
])
def test_private_files_are_recognised(path, private):
    assert browser_tools._looks_private(Path(path)) is private


def test_missing_file_is_reported(tmp_path):
    out = browser_tools.UploadFileTool().run(path=str(tmp_path / "nope.pptx"))
    assert out.startswith("File not found") and output_indicates_failure(out, "upload_file")


def test_private_file_is_never_uploaded(tmp_path, monkeypatch):
    secret = tmp_path / ".env"
    secret.write_text("GROQ_API_KEY=x")
    monkeypatch.setattr(browser_tools, "get_browser_controller", lambda: pytest.fail("must not touch the browser"))
    out = browser_tools.UploadFileTool().run(path=str(secret))
    assert out.startswith("Upload blocked") and output_indicates_failure(out, "upload_file")


def test_user_can_decline_the_upload(tmp_path, monkeypatch):
    deck = tmp_path / "deck.pptx"
    deck.write_bytes(b"x" * 2048)

    class _Confirmer:
        def __init__(self):
            self.asked = []

        async def request(self, description, details):
            self.asked.append(description)
            return False

    confirmer = _Confirmer()
    monkeypatch.setattr("app.utils.confirmation.get_confirmer", lambda: confirmer)
    page = SimpleNamespace(url="https://gemini.google.com/app")

    async def get_page():
        return page

    monkeypatch.setattr(browser_tools, "get_browser_controller", lambda: SimpleNamespace(get_active_page=get_page))
    out = browser_tools.UploadFileTool().run(path=str(deck))
    assert confirmer.asked and "deck.pptx" in confirmer.asked[0] and "gemini.google.com" in confirmer.asked[0]
    assert "declined" in out and output_indicates_failure(out, "upload_file")


# =============================================================================
# upload_file against a real (headless) browser
# =============================================================================

PLAIN_INPUT = """<input type="file" id="f" onchange="document.title = 'got ' + this.files[0].name">"""

BUTTON_OPENS_PICKER = """
<button aria-label="Attach files" onclick="
  const i = document.createElement('input'); i.type = 'file';
  i.onchange = () => document.title = 'got ' + i.files[0].name; i.click();">📎</button>"""

MENU_THEN_PICKER = """
<button aria-label="Open upload file menu" onclick="document.getElementById('menu').style.display='block'">+</button>
<div id="menu" style="display:none">
  <div role="menuitem" onclick="
    const i = document.createElement('input'); i.type = 'file';
    i.onchange = () => document.title = 'got ' + i.files[0].name; i.click();">Upload files</div>
</div>"""


@pytest.fixture
async def page():
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        yield await browser.new_page()
        await browser.close()


@pytest.mark.parametrize("html", [PLAIN_INPUT, BUTTON_OPENS_PICKER, MENU_THEN_PICKER],
                         ids=["plain-file-input", "button-opens-picker", "menu-then-picker"])
async def test_file_is_attached(page, tmp_path, html):
    deck = tmp_path / "System Design.pptx"
    deck.write_bytes(b"pptx")
    await page.set_content(html)
    how = await browser_tools._attach_file(page, str(deck))
    assert how is not None
    await page.wait_for_function("document.title.startsWith('got ')", timeout=5000)
    assert await page.title() == "got System Design.pptx"


async def test_page_without_any_upload_control(page, tmp_path):
    deck = tmp_path / "deck.pptx"
    deck.write_bytes(b"pptx")
    await page.set_content("<p>Nothing to upload here</p><button>Send</button>")
    assert await browser_tools._attach_file(page, str(deck)) is None


# =============================================================================
# see_window never turns a missing window into a whole-screen capture
# =============================================================================

def test_missing_window_is_reported_not_captured(monkeypatch):
    from app.tools import desktop

    monkeypatch.setattr(desktop, "_check_enabled", lambda: None)
    monkeypatch.setattr(desktop, "_ps", lambda script, timeout=60: "NOWINDOW|\n")
    out = desktop.SeeWindowTool().run(window="Gemini")
    assert out.startswith("No visible window matching 'Gemini'")
    assert "navigate_browser" in out
    assert output_indicates_failure(out, "see_window") is True
