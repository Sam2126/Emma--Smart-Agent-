"""
Everyday local-computer tools (2026-09-16).

The agent used to write PowerShell by hand for ordinary work, and a run that
created a folder and a file needed three failed attempts first:
"The string is missing the terminator". Commands now run from a script file,
and each everyday action has its own tool that cannot be mis-quoted.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agent.toolkit import build_toolset, describe_action
from app.tools import local


@pytest.fixture(autouse=True)
def local_tools_enabled(monkeypatch):
    monkeypatch.setattr(local, "_check_enabled", lambda: None)


# =============================================================================
# Commands run from a script file, not a quoted -Command string
# =============================================================================

def test_powershell_runs_from_a_script_file(monkeypatch):
    seen = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        seen["script"] = Path(args[-1]).read_text(encoding="utf-8-sig")
        return SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(local.subprocess, "run", fake_run)
    # A script naming no redirected folder reaches PowerShell unchanged; the
    # rewriting of "$env:USERPROFILE\Desktop" lives in test_real_folders_and_noise.py.
    local._run_powershell('Get-ChildItem -Path "D:\\work" -Recurse')
    assert "-File" in seen["args"] and "-Command" not in seen["args"]
    assert seen["script"] == 'Get-ChildItem -Path "D:\\work" -Recurse'


def test_quoted_command_that_used_to_fail_now_runs(tmp_path):
    """The exact shape that failed three times in the 2026-09-15 run."""
    target = tmp_path / "MAYANK"
    out = local.RunCommandTool().run(
        command=f'New-Item -ItemType Directory -Path "{target}" | Out-Null; Test-Path "{target}"'
    )
    assert "Exit code 0" in out and "True" in out
    assert target.is_dir()


def test_catastrophic_commands_are_still_blocked():
    assert "BLOCKED for safety" in local.RunCommandTool().run(command="Stop-Computer; shutdown /s")


# =============================================================================
# Folders and files
# =============================================================================

def test_create_folder_makes_parents_and_is_repeatable(tmp_path):
    target = tmp_path / "Desktop" / "MAYANK"
    first = local.CreateFolderTool().run(path=str(target))
    second = local.CreateFolderTool().run(path=str(target))
    assert target.is_dir()
    assert first.startswith("Folder ready") and "already existed" in second


def test_move_rename_and_into_a_folder(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("hello")
    renamed = tmp_path / "renamed.txt"
    assert "Moved" in local.MovePathTool().run(source=str(source), destination=str(renamed))
    assert renamed.read_text() == "hello"

    folder = tmp_path / "archive"
    folder.mkdir()
    local.MovePathTool().run(source=str(renamed), destination=str(folder))
    assert (folder / "renamed.txt").is_file()

    missing = local.MovePathTool().run(source=str(tmp_path / "gone.txt"), destination=str(folder))
    assert missing.startswith("Not found")


def test_copy_file_and_folder(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.txt").write_text("a")
    local.CopyPathTool().run(source=str(tmp_path / "src"), destination=str(tmp_path / "copy"))
    assert (tmp_path / "copy" / "a.txt").read_text() == "a"

    target_dir = tmp_path / "into"
    target_dir.mkdir()
    local.CopyPathTool().run(source=str(tmp_path / "src" / "a.txt"), destination=str(target_dir))
    assert (target_dir / "a.txt").is_file()


def test_delete_goes_to_the_recycle_bin(tmp_path, monkeypatch):
    victim = tmp_path / "old.txt"
    victim.write_text("x")
    recycled = []
    monkeypatch.setattr(local, "_recycle", lambda path: recycled.append(path) or None)
    out = local.DeletePathTool().run(path=str(victim))
    assert "Recycle Bin" in out and recycled == [victim.resolve()]


def test_delete_refuses_system_and_home_folders(monkeypatch):
    monkeypatch.setattr(local, "_recycle", lambda path: pytest.fail("must not delete"))
    out = local.DeletePathTool().run(path=str(Path.home()))
    assert out.startswith("Refused")


def test_delete_reports_a_missing_path(tmp_path):
    assert local.DeletePathTool().run(path=str(tmp_path / "nothing")).startswith("Not found")


# =============================================================================
# Search inside files
# =============================================================================

def test_search_in_files_finds_the_line(tmp_path):
    (tmp_path / "code.py").write_text("def is_prime(n):\n    return n > 1\n")
    (tmp_path / "notes.txt").write_text("nothing here")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "more.py").write_text("# is_prime again\n")

    out = local.SearchInFilesTool().run(text="is_prime", location=str(tmp_path))
    assert "code.py:1" in out and "more.py:1" in out and "notes.txt" not in out

    only_txt = local.SearchInFilesTool().run(text="is_prime", location=str(tmp_path), file_pattern="*.txt")
    assert only_txt.startswith("No files under")


# =============================================================================
# Reading documents
# =============================================================================

def test_read_word_document(tmp_path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "report.docx"
    document = docx.Document()
    document.add_paragraph("System design is worth learning")
    document.save(str(path))
    out = local.ReadFileTool().run(path=str(path))
    assert "System design is worth learning" in out


def test_read_powerpoint_document(tmp_path):
    pptx = pytest.importorskip("pptx")
    path = tmp_path / "deck.pptx"
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = "Scaling a web app"
    presentation.save(str(path))
    out = local.ReadFileTool().run(path=str(path))
    assert "slide 1" in out and "Scaling a web app" in out


def test_read_excel_document(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "marks.xlsx"
    workbook = openpyxl.Workbook()
    workbook.active.append(["Subject", "Marks"])
    workbook.active.append(["DBMS", 91])
    workbook.save(str(path))
    out = local.ReadFileTool().run(path=str(path))
    assert "Subject | Marks" in out and "DBMS | 91" in out


def test_read_plain_text_is_unchanged(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("plain content")
    assert "plain content" in local.ReadFileTool().run(path=str(path))


# =============================================================================
# Clipboard
# =============================================================================

def test_clipboard_read(monkeypatch):
    monkeypatch.setattr(local, "_run_powershell", lambda script, timeout_seconds=60: SimpleNamespace(stdout="copied text", stderr="", returncode=0))
    assert "copied text" in local.ClipboardTool().run()


def test_clipboard_write_sends_text_through_a_file(monkeypatch):
    scripts = []

    def fake(script, timeout_seconds=60):
        scripts.append(script)
        payload = script.split("'")[1]
        assert Path(payload).read_text(encoding="utf-8") == 'He said "hi" & left'
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(local, "_run_powershell", fake)
    out = local.ClipboardTool().run(text='He said "hi" & left')
    assert "Copied 19 characters" in out and "Set-Clipboard" in scripts[0]


# =============================================================================
# The agent can actually reach these tools
# =============================================================================

def test_find_files_returns_the_newest_matches_first(tmp_path):
    """"add the last ppt from downloads" must reach the newest one. Found
    2026-09-16: matches came back in folder order and the scan stopped at
    max_results, so the newest .pptx was usually not even in the list."""
    import os
    import time as _time

    names = ["old deck.pptx", "mid deck.pptx", "new deck.pptx"]
    for offset, name in zip((3000, 1500, 0), names):
        path = tmp_path / name
        path.write_text(name)
        stamp = _time.time() - offset
        os.utime(path, (stamp, stamp))

    out = local.FindFilesTool().run(pattern=".pptx", location=str(tmp_path), max_results=2)
    assert "newest first" in out
    listed = [line for line in out.splitlines() if "deck.pptx" in line]
    assert len(listed) == 2, out
    assert "new deck.pptx" in listed[0] and "mid deck.pptx" in listed[1]


def test_new_tools_are_available_to_the_agent():
    names = set(build_toolset("local"))
    assert {"create_folder", "move_path", "copy_path", "delete_path", "search_in_files", "clipboard"} <= names


def test_new_tools_have_progress_lines():
    assert describe_action("create_folder", {"path": r"C:\x\MAYANK"}) == "📁 Creating folder 'C:\\x\\MAYANK'"
    assert describe_action("delete_path", {"path": r"C:\x\old.txt"}).startswith("🗑️ Sending")
    assert describe_action("search_in_files", {"text": "is_prime"}) == "🔎 Searching files for 'is_prime'"
    assert describe_action("clipboard", {"text": "hi"}) == "📋 Copying to the clipboard"
    assert describe_action("clipboard", {}) == "📋 Reading the clipboard"
