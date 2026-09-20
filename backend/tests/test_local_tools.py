"""Tests for local computer access tools (app/tools/local.py).

Pure-stdlib paths are tested directly; shell/app-launch tools are only
exercised where the platform supports them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from unittest.mock import patch, MagicMock

from app.tools.local import (
    FindFilesTool,
    ListFolderTool,
    ListRecentFilesTool,
    OpenAppTool,
    ReadFileTool,
    RunCommandTool,
    SendKeysTool,
    WriteFileTool,
    _downloads_dir,
    _resolve_location,
)


@pytest.fixture()
def scratch(tmp_path: Path) -> Path:
    (tmp_path / "report 2026.pdf").write_text("annual report", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("hello world", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "invoice.pdf").write_text("invoice", encoding="utf-8")
    return tmp_path


def test_list_recent_files_newest_first(scratch: Path):
    tool = ListRecentFilesTool()
    out = tool._run(location=str(scratch), count=5)
    # _run is sync wrapper around async; app.tools.base.BaseTool._run executes directly
    assert "Recent files" in out
    assert "notes.txt" in out and "report 2026.pdf" in out


def test_find_files_recursive(scratch: Path):
    tool = FindFilesTool()
    out = tool._run(pattern="invoice", location=str(scratch))
    assert "invoice.pdf" in out


def test_read_write_file_roundtrip(scratch: Path):
    target = scratch / "out.txt"
    w = WriteFileTool()
    r = ReadFileTool()
    assert "Wrote" in w._run(path=str(target), content="line1\nline2")
    read = r._run(path=str(target))
    assert "line1" in read and "line2" in read
    assert "Appended" in w._run(path=str(target), content="line3", append=True)
    assert "line3" in r._run(path=str(target))


def test_list_folder(scratch: Path):
    tool = ListFolderTool()
    out = tool._run(path=str(scratch))
    assert "[DIR ] sub" in out and "[FILE] notes.txt" in out


def test_resolve_location_known_folders():
    d = _resolve_location("downloads")
    assert d == _downloads_dir()
    assert _resolve_location(str(Path.home())) == Path.home()


def test_downloads_dir_exists():
    # Sanity: the resolved Downloads folder is a real directory
    assert _downloads_dir().exists()


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell only on Windows")
def test_run_command_echo():
    tool = RunCommandTool()
    out = tool._run(command="Write-Output 'agent-smoke-ok'", timeout_seconds=30)
    assert "agent-smoke-ok" in out


# =============================================================================
# open_app: regression coverage for the false-positive bug — the tool used
# to always return "Launched: X" via a bare `cmd /c start` even when Windows
# could not resolve the app (e.g. Store/UWP apps like WhatsApp), producing a
# genuine Windows error dialog while reporting success back to the agent AND
# to the learning system. Verified live against this machine (see session
# notes) that Get-StartApps + `explorer.exe shell:AppsFolder\<id>` is the
# mechanism that actually works; these tests mock subprocess so they run
# without touching real windows/processes.
# =============================================================================

def test_open_app_resolves_via_start_apps_when_found():
    tool = OpenAppTool()
    lookup_result = MagicMock(stdout="FOUND|5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App\n", returncode=0)
    with patch("app.tools.local.subprocess.run", return_value=lookup_result) as mock_run, \
         patch("app.tools.local.subprocess.Popen") as mock_popen, \
         patch("app.tools.local._wait_for_app_window", return_value=("whatsapp.root :: WhatsApp", "ready")):
        out = tool._run(name="WhatsApp")

    assert "Launched: WhatsApp" in out
    assert "5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App" in out
    # Must launch via explorer.exe shell:AppsFolder, not the old bare `start`.
    popen_args = mock_popen.call_args[0][0]
    assert popen_args[0] == "explorer.exe"
    assert "shell:AppsFolder" in popen_args[1]
    assert "5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App" in popen_args[1]


def test_open_app_does_not_launch_an_unknown_name():
    """An unknown name used to be run blindly with `cmd /c start`, which popped
    Windows' "cannot find" error and was reported as a launch attempt (found
    2026-09-15 with "open gemini", a website). Now nothing is started and the
    agent is told to use the browser for web apps."""
    tool = OpenAppTool()
    lookup_result = MagicMock(stdout="NOTFOUND\n", returncode=0)
    with patch("app.tools.local.subprocess.run", return_value=lookup_result), \
         patch("app.tools.local.subprocess.Popen") as mock_popen:
        out = tool._run(name="TotallyMadeUpAppXYZ")

    # Must NOT claim a confident "Launched:" — that was the false-positive bug.
    assert "Launched:" not in out
    assert out.startswith("Could not open 'TotallyMadeUpAppXYZ'")
    assert "navigate_browser" in out
    mock_popen.assert_not_called()


def test_open_app_rejects_empty_name():
    tool = OpenAppTool()
    assert "No app name given" in tool._run(name="   ")


# =============================================================================
# send_keys: escaping, required window_hint, and force-focus-before-send.
#
# Regression coverage for the false-positive bug found live in production:
# the first version of this tool sent keys to "whatever has focus" without
# forcing focus itself, so a real task ("open WhatsApp and search X and
# send hello") opened WhatsApp but every subsequent send_keys call typed
# into the wrong window (most likely this backend's own console), while the
# agent's final report narrated each step as if it had worked. Verified
# live (see session notes) that this requires: (1) window_hint to identify
# the target, (2) actually forcing that window to the foreground via
# AttachThreadInput + SetForegroundWindow (a bare SetForegroundWindow call
# from a background subprocess is silently refused by Windows), and (3)
# refusing to send anything and saying so plainly if no matching window is
# found or it can't be focused — never a blind "Sent keys" claim.
# =============================================================================

def test_send_keys_rejects_empty_input():
    tool = SendKeysTool()
    out = tool._run(window_hint="WhatsApp", text="", keys="", delay_ms=0)
    assert "Nothing to send" in out


def test_send_keys_requires_window_hint():
    tool = SendKeysTool()
    out = tool._run(window_hint="   ", text="hello", delay_ms=0)
    assert "window_hint is required" in out


def test_send_keys_uses_sendinput_and_verified_focus_when_window_found():
    """Keys must go through SendInput after the window is found AND focused.

    .NET SendKeys drives input via legacy journal-playback hooks, which
    Chromium/WebView2 content silently ignores - that is why typing into
    WhatsApp reported success while its search box stayed empty. The payload
    must therefore be injected with SendInput, and focus must be obtained
    (and verified) rather than assumed.
    """
    tool = SendKeysTool()
    completed = MagicMock(returncode=0, stdout="SENT|whatsapp.root|WhatsApp\n", stderr="")
    with patch("app.tools.desktop.subprocess.run", return_value=completed) as mock_run:
        out = tool._run(window_hint="WhatsApp", text="Vavish", keys="^f", delay_ms=100)

    assert "Sent to" in out
    assert "WhatsApp" in out
    ps_command = mock_run.call_args[0][0][-1]
    assert "SendInput" in ps_command
    assert "SetForegroundWindow" in ps_command
    assert "AttachThreadInput" in ps_command
    # WebView2 apps (WhatsApp) swallow keys unless focus is handed to the web host child
    assert "Chrome_WidgetWin_0" in ps_command
    # keys before text, each dispatched through the SendInput helper
    assert "[OaskUI]::Keys('^f')" in ps_command
    assert "[OaskUI]::Type('Vavish')" in ps_command
    assert ps_command.index("::Keys(") < ps_command.index("::Type(")


def test_send_keys_waits_for_a_window_instead_of_failing_instantly():
    """A freshly launched app needs seconds before its window exists."""
    tool = SendKeysTool()
    completed = MagicMock(returncode=0, stdout="SENT|whatsapp.root|WhatsApp\n", stderr="")
    with patch("app.tools.desktop.subprocess.run", return_value=completed) as mock_run:
        tool._run(window_hint="WhatsApp", text="hi", delay_ms=0)

    ps_command = mock_run.call_args[0][0][-1]
    assert "WaitFor(" in ps_command


def test_send_keys_reports_window_not_found_without_sending():
    tool = SendKeysTool()
    completed = MagicMock(returncode=0, stdout="WINDOW_NOT_FOUND\n", stderr="")
    with patch("app.tools.desktop.subprocess.run", return_value=completed):
        out = tool._run(window_hint="NoSuchApp", text="hello", delay_ms=0)

    assert "No visible window matching" in out
    assert "Sent to" not in out


def test_send_keys_reports_focus_failure_without_claiming_success():
    tool = SendKeysTool()
    completed = MagicMock(returncode=0, stdout="FOCUS_FAILED\n", stderr="")
    with patch("app.tools.desktop.subprocess.run", return_value=completed):
        out = tool._run(window_hint="WhatsApp", text="hello", delay_ms=0)

    assert "refused to bring it to the foreground" in out
    assert "NOTHING was sent" in out
    assert "Sent to" not in out


# =============================================================================
# App readiness: a cold WhatsApp start shows a near-empty splash (logo, progress
# bar, "End-to-end encrypted") for ~8s after its window appears. The agent
# screenshotted that splash, was told it was a login screen, and stopped.
# open_app now waits until the window stops looking like a splash.
# =============================================================================

from PIL import Image, ImageDraw  # noqa: E402

from app.tools.desktop import _frame_change, _looks_like_splash, _reduce_frame  # noqa: E402


def _splash_like() -> Image.Image:
    im = Image.new("RGB", (800, 600), (245, 244, 242))
    d = ImageDraw.Draw(im)
    d.ellipse((465, 238, 530, 300), outline=(210, 210, 210), width=4)
    d.text((450, 330), "WhatsApp", fill=(0, 0, 0))
    d.line((375, 386, 625, 386), fill=(200, 200, 200), width=4)
    return im


def _loaded_ui() -> Image.Image:
    """A chat-list style window: sidebar icons, search box, avatars, text lines."""
    im = Image.new("RGB", (800, 600), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 90, 600), fill=(240, 240, 240))
    for i in range(6):
        d.rectangle((30, 70 + i * 60, 60, 100 + i * 60), fill=(60, 60, 60))
    d.rounded_rectangle((115, 130, 395, 180), radius=20, fill=(225, 225, 225))
    d.rectangle((140, 148, 330, 162), fill=(120, 120, 120))
    for row in range(5):
        y = 210 + row * 78
        d.ellipse((115, y, 170, y + 55), fill=(40, 110, 80))
        d.rectangle((190, y + 6, 190 + 120 + row * 15, y + 22), fill=(20, 20, 20))
        d.rectangle((190, y + 32, 380 - row * 10, y + 44), fill=(110, 110, 110))
        d.rectangle((340, y + 6, 390, y + 18), fill=(0, 150, 90))
    return im


def test_splash_screen_is_recognised_as_still_loading():
    assert _looks_like_splash(_reduce_frame(_splash_like())) is True


def test_loaded_interface_is_not_mistaken_for_a_splash():
    assert _looks_like_splash(_reduce_frame(_loaded_ui())) is False


def test_frame_change_detects_splash_to_ui_transition():
    splash = _reduce_frame(_splash_like())
    assert _frame_change(splash, splash) == 0.0
    assert _frame_change(splash, _reduce_frame(_loaded_ui())) > 5.0


def test_open_app_reports_loading_screen_instead_of_claiming_ready():
    tool = OpenAppTool()
    lookup_result = MagicMock(stdout="FOUND|5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App\n", returncode=0)
    with patch("app.tools.local.subprocess.run", return_value=lookup_result), \
         patch("app.tools.local.subprocess.Popen"), \
         patch("app.tools.local._wait_for_app_window", return_value=("whatsapp.root :: WhatsApp", "loading")):
        out = tool._run(name="WhatsApp")

    assert "still showing its loading screen" in out
    assert "not a login screen" in out
    assert "is ready" not in out
