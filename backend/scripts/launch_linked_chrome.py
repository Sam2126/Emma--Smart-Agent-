"""
Open the agent's own Chrome window, with the debug port the backend attaches to.

Use it to sign in, once, to the sites you want the agent to use (Gemini,
WhatsApp Web, shopping sites). The agent's Chrome keeps those logins in its own
profile, apart from your Chrome profiles, which stay open and signed in as
usual. A running backend attaches to this window; otherwise the backend opens
the same window itself on the first browser task.

    backend\\.venv\\Scripts\\python.exe backend\\scripts\\launch_linked_chrome.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.browser.controller import (  # noqa: E402
    AGENT_PROFILE_DIR,
    AGENT_PROFILE_NAME,
    _agent_chrome_command,
    _find_chrome_exe,
    _port_open,
    _prepare_agent_profile,
)

DEBUG_PORT = 9222


def main() -> int:
    if _port_open("127.0.0.1", DEBUG_PORT):
        print(f"[OK] A Chrome window with the debug port is already open (port {DEBUG_PORT}).")
        return 0
    chrome = _find_chrome_exe()
    if not chrome:
        print("[X] Chrome not found on this machine.")
        return 1
    _prepare_agent_profile(AGENT_PROFILE_DIR)
    subprocess.Popen(
        _agent_chrome_command(chrome, DEBUG_PORT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"[OK] Opened the agent's Chrome window ({AGENT_PROFILE_NAME}).")
    print(f"     Profile: {AGENT_PROFILE_DIR}")
    print("     Sign in there to the sites the agent should use. Your own Chrome")
    print("     profiles are separate and are not touched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
