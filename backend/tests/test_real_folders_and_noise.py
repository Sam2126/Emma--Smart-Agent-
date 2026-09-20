"""
From the 2026-09-16 run "create a new folder in desktop named MAYANK ...":

  * the folder was created at C:\\Users\\<user>\\Desktop\\MAYANK while the user's
    real desktop was C:\\Users\\<user>\\OneDrive\\Desktop, so nothing appeared on
    screen and the task looked like it had done nothing
  * a stray microphone recording ran as a task with no tool calls, reported
    success, and was stored as a good experience
  * the offline wake word printed every decoy it heard ("dune", "dawn", "sun")
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.toolkit import local_completion_shortfall
from app.tools import local


@pytest.fixture
def onedrive_desktop(tmp_path, monkeypatch):
    """Windows reporting Desktop and Documents inside OneDrive, as on this machine."""
    folders = {
        "desktop": tmp_path / "OneDrive" / "Desktop",
        "documents": tmp_path / "OneDrive" / "Documents",
        "downloads": tmp_path / "Downloads",
        "pictures": tmp_path / "Pictures",
        "videos": tmp_path / "Videos",
        "music": tmp_path / "Music",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(local, "_known_folder", lambda name: folders[name])
    monkeypatch.setattr(local, "_LOCATION_MAP", {
        **{name: (lambda name=name: folders[name]) for name in folders},
        "home": lambda: tmp_path,
    })
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(local, "_check_enabled", lambda: None)
    return folders


# =============================================================================
# The user's real folders
# =============================================================================

def test_desktop_resolves_to_the_real_desktop(onedrive_desktop):
    assert local._resolve_user_path(r"Desktop\MAYANK") == onedrive_desktop["desktop"] / "MAYANK"
    assert local._resolve_user_path("desktop") == onedrive_desktop["desktop"]
    assert local._resolve_user_path(r"Documents\notes\todo.txt") == onedrive_desktop["documents"] / "notes" / "todo.txt"
    assert local._resolve_user_path(r"Downloads\deck.pptx") == onedrive_desktop["downloads"] / "deck.pptx"


def test_absolute_paths_are_left_alone(onedrive_desktop):
    assert local._resolve_user_path(r"D:\work\file.txt") == Path(r"D:\work\file.txt")


def test_a_hand_built_userprofile_desktop_path_is_redirected(onedrive_desktop, tmp_path):
    """Found 2026-09-16, after the folder lookup was fixed: the model wrote an
    absolute '%USERPROFILE%\\Desktop\\MAYANK\\prime_code.txt' of its own, so the
    folder landed on the real desktop and the file inside it did not."""
    assert local._resolve_user_path(str(tmp_path / "Desktop" / "MAYANK" / "prime_code.txt")) == (
        onedrive_desktop["desktop"] / "MAYANK" / "prime_code.txt"
    )
    assert local._resolve_user_path(str(tmp_path / "Documents" / "a.txt")) == onedrive_desktop["documents"] / "a.txt"
    assert local._resolve_user_path(str(tmp_path / "My Documents" / "a.txt")) == onedrive_desktop["documents"] / "a.txt"


def test_paths_already_in_the_real_folder_are_untouched(onedrive_desktop):
    target = onedrive_desktop["desktop"] / "MAYANK"
    assert local._resolve_user_path(str(target)) == target


def test_a_folder_windows_does_not_redirect_keeps_its_literal_path(onedrive_desktop, tmp_path):
    assert local._resolve_user_path(str(tmp_path / "Downloads" / "x.zip")) == tmp_path / "Downloads" / "x.zip"
    assert local._resolve_user_path(str(tmp_path / "projects" / "x.txt")) == tmp_path / "projects" / "x.txt"


def test_writing_to_a_hand_built_desktop_path_lands_on_the_real_desktop(onedrive_desktop, tmp_path):
    local.CreateFolderTool().run(path=r"Desktop\MAYANK")
    local.WriteFileTool().run(path=str(tmp_path / "Desktop" / "MAYANK" / "prime_code.txt"), content="def is_prime(n): ...")
    assert (onedrive_desktop["desktop"] / "MAYANK" / "prime_code.txt").read_text() == "def is_prime(n): ..."
    assert not (tmp_path / "Desktop").exists(), "nothing may be written to the invisible leftover folder"


def test_other_relative_paths_stay_in_the_home_folder(onedrive_desktop, tmp_path):
    assert local._resolve_user_path(r"projects\x.txt") == tmp_path / "projects" / "x.txt"


def test_create_folder_and_write_file_land_on_the_real_desktop(onedrive_desktop):
    out = local.CreateFolderTool().run(path=r"Desktop\MAYANK")
    assert str(onedrive_desktop["desktop"] / "MAYANK") in out
    assert (onedrive_desktop["desktop"] / "MAYANK").is_dir()

    local.WriteFileTool().run(path=r"Desktop\MAYANK\prime_code.txt", content="def is_prime(n): ...")
    written = onedrive_desktop["desktop"] / "MAYANK" / "prime_code.txt"
    assert written.read_text() == "def is_prime(n): ..."
    assert not (Path.home() / "Desktop" / "MAYANK").exists(), "must not use the non-redirected Desktop"


def test_search_locations_use_the_real_folders(onedrive_desktop, tmp_path):
    assert local._resolve_location("desktop") == onedrive_desktop["desktop"]
    assert local._resolve_location(str(tmp_path / "Desktop" / "notes")) == onedrive_desktop["desktop"] / "notes"
    assert local._resolve_location(r"D:\work") == Path(r"D:\work")


def test_a_powershell_script_is_pointed_at_the_real_desktop(onedrive_desktop):
    """The strategy recalled for the MAYANK task on 2026-09-16 was exactly this
    command, and a script's own paths never pass through _resolve_user_path."""
    script = 'New-Item -ItemType Directory -Path "$env:USERPROFILE\\Desktop\\MAYANK" -Force'
    out = local._redirect_shell_script(script)
    assert str(onedrive_desktop["desktop"] / "MAYANK") in out
    assert "USERPROFILE" not in out


def test_other_spellings_of_the_home_folder_are_covered(onedrive_desktop):
    assert str(onedrive_desktop["desktop"]) in local._redirect_shell_script("cd $HOME\\Desktop")
    assert str(onedrive_desktop["desktop"]) in local._redirect_shell_script("cd ~/Desktop/notes")
    assert str(onedrive_desktop["documents"]) in local._redirect_shell_script('ls "%USERPROFILE%\\Documents"')


def test_scripts_that_name_no_redirected_folder_are_left_alone(onedrive_desktop):
    for script in (
        "Get-Process | Select-Object -First 3",
        'Set-Content -Path "$env:USERPROFILE\\Downloads\\a.txt" -Value x',
        'echo "$env:USERPROFILE\\projects\\x"',
        'echo "$env:USERPROFILE\\DesktopShortcuts\\x"',
    ):
        assert local._redirect_shell_script(script) == script


def test_read_and_list_use_the_real_folders(onedrive_desktop):
    note = onedrive_desktop["documents"] / "note.txt"
    note.write_text("remember this")
    assert "remember this" in local.ReadFileTool().run(path=r"Documents\note.txt")
    assert "note.txt" in local.ListFolderTool().run(path="documents")


def test_known_folder_falls_back_when_windows_says_nothing(monkeypatch):
    monkeypatch.setattr(local.os.path, "expandvars", lambda value: "")
    assert local._known_folder("desktop") == Path.home() / "Desktop"
    assert local._known_folder("documents") == Path.home() / "Documents"


# =============================================================================
# A run that did nothing is not a success
# =============================================================================

def test_a_task_with_no_steps_is_not_complete():
    assert local_completion_shortfall("create a folder on the desktop", []) == (
        "no tool ran at all, so nothing was actually done on the computer"
    )
    assert local_completion_shortfall("open whatsapp and message rakesh", [])


def test_a_question_without_an_action_is_still_fine():
    assert local_completion_shortfall("how are you doing today", []) is None
    assert local_completion_shortfall("तुझे लिसन करेगा", []) is None


# =============================================================================
# The wake word only reports near-misses
# =============================================================================

def test_only_wake_like_noises_are_printed():
    from app.wake_listener import WakeWordListener

    listener = WakeWordListener(wake_word="hello", stop_word="done")
    assert listener._sounds_like_wake_attempt("hello") is True
    assert listener._sounds_like_wake_attempt("halo [unk]") is True
    for noise in ("dune", "dawn", "sun", "we're", "dumb dune done", "[unk]"):
        assert listener._sounds_like_wake_attempt(noise) is False
