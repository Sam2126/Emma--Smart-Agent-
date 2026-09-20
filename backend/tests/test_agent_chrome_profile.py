"""
The agent's Chrome runs on its own profile and never on one of the user's.

Found 2026-09-17: the agent ran on a copy of the user's "Person 1" profile,
Google sign-in included, and the user's own Person 1 kept being signed out; a
launch on the real (migrated) profile folder also kept the user's other Chrome
profiles from opening while the agent's window was up.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from app.browser import controller as ctl

BACKEND = Path(__file__).resolve().parent.parent


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_agent_chrome_command_uses_only_the_agent_profile(monkeypatch, tmp_path):
    agent_dir = tmp_path / "agent_profile"
    monkeypatch.setattr(ctl, "AGENT_PROFILE_DIR", agent_dir)
    cmd = ctl._agent_chrome_command("chrome.exe", 9222)
    assert cmd[0] == "chrome.exe"
    assert f"--user-data-dir={agent_dir}" in cmd
    assert "--profile-directory=Default" in cmd
    assert "--remote-debugging-port=9222" in cmd
    assert "--window-name=Self-Improving Agent" in cmd
    joined = " ".join(cmd)
    for users_chrome in ("Google\\Chrome\\User Data", "ChromeProfile", "chrome_shadow"):
        assert users_chrome not in joined


def test_old_chrome_settings_in_env_are_ignored(monkeypatch, tmp_path):
    from app.config import Settings

    monkeypatch.setenv("CHROME_USER_DATA_DIR", str(tmp_path / "ChromeProfile"))
    monkeypatch.setenv("CHROME_PROFILE_DIR", "Default")
    settings = Settings(_env_file=None)
    assert not hasattr(settings, "chrome_user_data_dir")
    assert not hasattr(settings, "chrome_profile_dir")


async def test_connect_launches_the_agent_profile_not_the_users(monkeypatch, tmp_path):
    agent_dir = tmp_path / "agent_profile"
    real_dir = tmp_path / "ChromeProfile"
    (real_dir / "Default").mkdir(parents=True)
    monkeypatch.setenv("CHROME_USER_DATA_DIR", str(real_dir))
    monkeypatch.setattr(ctl, "AGENT_PROFILE_DIR", agent_dir)

    launched: list[list[str]] = []
    attached: list[str] = []

    class _Driver:
        def __init__(self):
            self.chromium = SimpleNamespace(connect_over_cdp=self._connect)

        async def _connect(self, endpoint, timeout=None):
            if not launched:
                raise ConnectionError("nothing on the debug port yet")
            attached.append(endpoint)
            return SimpleNamespace(contexts=[], is_connected=lambda: True)

        async def stop(self):
            pass

    class _Starter:
        async def start(self):
            return _Driver()

    async def _port_ready(host, port, timeout=6.0):
        return True

    def _popen(args, **kwargs):
        launched.append(list(args))
        return SimpleNamespace(pid=4242, terminate=lambda: None)

    monkeypatch.setattr(ctl, "async_playwright", lambda: _Starter())
    monkeypatch.setattr(ctl, "_port_open", lambda host, port, timeout=0.5: False)
    monkeypatch.setattr(ctl, "_find_chrome_exe", lambda: "chrome.exe")
    monkeypatch.setattr(ctl, "_wait_for_cdp_port", _port_ready)
    monkeypatch.setattr(ctl.subprocess, "Popen", _popen)

    browser_controller = ctl.BrowserController()
    await browser_controller.connect("http://127.0.0.1:9222")

    assert browser_controller.is_connected
    assert attached == ["http://127.0.0.1:9222"]
    assert len(launched) == 1
    assert f"--user-data-dir={agent_dir}" in launched[0]
    assert not any(str(real_dir) in arg for arg in launched[0])
    assert list(real_dir.iterdir()) == [real_dir / "Default"]  # the user's folder is untouched
    assert _read(agent_dir / "Local State")["profile"]["info_cache"]["Default"]["name"] == "Self-Improving Agent"


def test_new_agent_profile_gets_its_own_name(tmp_path):
    ctl._prepare_agent_profile(tmp_path)
    entry = _read(tmp_path / "Local State")["profile"]["info_cache"]["Default"]
    assert entry == {"name": "Self-Improving Agent", "is_using_default_name": False}
    prefs = _read(tmp_path / "Default" / "Preferences")["profile"]
    assert prefs == {"name": "Self-Improving Agent", "using_default_name": False}


def test_default_named_agent_profile_is_renamed_and_other_settings_kept(tmp_path):
    (tmp_path / "Default").mkdir()
    (tmp_path / "Local State").write_text(json.dumps({
        "os_crypt": {"encrypted_key": "abc"},
        "profile": {"info_cache": {"Default": {"name": "Your Chrome", "is_using_default_name": True, "avatar_icon": "x"}}},
    }), encoding="utf-8")
    (tmp_path / "Default" / "Preferences").write_text(json.dumps({
        "profile": {"name": "Person 1", "exit_type": "Normal"}, "session": {"restore_on_startup": 1},
    }), encoding="utf-8")

    ctl._prepare_agent_profile(tmp_path)

    state = _read(tmp_path / "Local State")
    assert state["os_crypt"] == {"encrypted_key": "abc"}
    assert state["profile"]["info_cache"]["Default"] == {
        "name": "Self-Improving Agent", "is_using_default_name": False, "avatar_icon": "x",
    }
    prefs = _read(tmp_path / "Default" / "Preferences")
    assert prefs["profile"] == {"name": "Self-Improving Agent", "exit_type": "Normal", "using_default_name": False}
    assert prefs["session"] == {"restore_on_startup": 1}


def test_a_name_chosen_in_chrome_is_kept(tmp_path):
    (tmp_path / "Default").mkdir()
    chosen = {"profile": {"info_cache": {"Default": {"name": "Shopping bot", "is_using_default_name": False}}}}
    (tmp_path / "Local State").write_text(json.dumps(chosen), encoding="utf-8")
    ctl._prepare_agent_profile(tmp_path)
    assert _read(tmp_path / "Local State") == chosen


def test_unreadable_profile_files_never_block_the_launch(tmp_path):
    (tmp_path / "Default").mkdir()
    (tmp_path / "Local State").write_text("{not json", encoding="utf-8")
    ctl._prepare_agent_profile(tmp_path)
    assert (tmp_path / "Local State").read_text(encoding="utf-8") == "{not json"
    assert _read(tmp_path / "Default" / "Preferences")["profile"]["name"] == "Self-Improving Agent"


def test_manual_launcher_opens_the_agent_window(monkeypatch, tmp_path, capsys):
    spec = importlib.util.spec_from_file_location(
        "launch_linked_chrome", BACKEND / "scripts" / "launch_linked_chrome.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    launched: list[list[str]] = []
    monkeypatch.setattr(ctl, "AGENT_PROFILE_DIR", tmp_path / "agent_profile")
    monkeypatch.setattr(script, "AGENT_PROFILE_DIR", tmp_path / "agent_profile")
    monkeypatch.setattr(script, "_port_open", lambda host, port, timeout=0.5: False)
    monkeypatch.setattr(script, "_find_chrome_exe", lambda: "chrome.exe")
    monkeypatch.setattr(script.subprocess, "Popen", lambda args, **kw: launched.append(list(args)))

    assert script.main() == 0
    assert f"--user-data-dir={tmp_path / 'agent_profile'}" in launched[0]
    assert (tmp_path / "agent_profile" / "Local State").exists()
    assert "Your own Chrome" in capsys.readouterr().out
