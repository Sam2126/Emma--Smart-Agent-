"""
Install the Self-Improving Agent desktop app.

Run once, from the project folder:
    backend\\.venv\\Scripts\\python.exe backend\\scripts\\install_desktop_app.py

It:
  1. installs the two small packages the app needs (pywebview, pystray) into
     the backend's virtual environment, if they are missing
  2. checks that Microsoft Edge WebView2 is present (it ships with Windows 11)
  3. draws the app icon
  4. creates "Self-Improving Agent" shortcuts on the Desktop and in the Start menu
  5. removes the old "Local Agent" shortcuts, which opened a Chrome window on
     http://localhost:8000/local but never started the backend

Opening the shortcut starts the backend in the background, shows the agent
window, and keeps listening for the wake word from the system tray. Tray menu >
"Start with Windows" keeps it listening right after you log in.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
DESKTOP_APP = PROJECT_ROOT / "desktop"
APP_SCRIPT = DESKTOP_APP / "agent_desktop.pyw"
ASSETS = DESKTOP_APP / "assets"
VENV_SCRIPTS = BACKEND_DIR / ".venv" / "Scripts"
PYTHON = VENV_SCRIPTS / "python.exe"
PYTHONW = VENV_SCRIPTS / "pythonw.exe"
NAME = "Self-Improving Agent"
OLD_NAME = "Local Agent"
PACKAGES = {"webview": "pywebview>=5.0", "pystray": "pystray>=0.19"}
WEBVIEW2_CLIENT_ID = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"


def _ps_quote(value: object) -> str:
    return str(value).replace("'", "''")


def run_powershell(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True,
    )


def install_packages() -> bool:
    missing = [spec for module, spec in PACKAGES.items() if importlib.util.find_spec(module) is None]
    if not missing:
        print("[OK] pywebview and pystray are installed")
        return True
    print(f"Installing {', '.join(missing)} ...")
    result = subprocess.run([str(PYTHON), "-m", "pip", "install", *missing])
    if result.returncode != 0:
        print("[X] pip install failed; see the output above.")
        return False
    print("[OK] Packages installed")
    return True


def webview2_installed() -> bool:
    import winreg

    locations = [
        (winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{WEBVIEW2_CLIENT_ID}"),
        (winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{WEBVIEW2_CLIENT_ID}"),
        (winreg.HKEY_CURRENT_USER, rf"Software\Microsoft\EdgeUpdate\Clients\{WEBVIEW2_CLIENT_ID}"),
    ]
    for hive, path in locations:
        try:
            with winreg.OpenKey(hive, path) as key:
                version, _ = winreg.QueryValueEx(key, "pv")
                if version and version != "0.0.0.0":
                    return True
        except OSError:
            continue
    return False


def draw_icon() -> None:
    """A microphone on a blue-violet rounded square, as .png (tray) and .ico (shortcut, window)."""
    from PIL import Image, ImageDraw

    ASSETS.mkdir(parents=True, exist_ok=True)
    size = 256
    gradient = Image.new("RGBA", (size, size))
    shade = ImageDraw.Draw(gradient)
    top, bottom = (79, 140, 255), (143, 92, 255)
    for y in range(size):
        t = y / (size - 1)
        shade.line((0, y, size, y), fill=tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)) + (255,))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((8, 8, size - 8, size - 8), radius=56, fill=255)

    icon = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    icon.paste(gradient, (0, 0), mask)
    draw = ImageDraw.Draw(icon)
    white = (255, 255, 255, 255)
    draw.rounded_rectangle((98, 44, 158, 146), radius=30, fill=white)      # capsule
    draw.arc((70, 86, 186, 186), start=0, end=180, fill=white, width=12)   # cradle
    draw.rectangle((122, 184, 134, 206), fill=white)                        # stem
    draw.rounded_rectangle((90, 202, 166, 216), radius=7, fill=white)       # base

    icon.save(ASSETS / "icon.png")
    icon.save(ASSETS / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print(f"[OK] Icon drawn: {ASSETS / 'icon.ico'}")


def shortcut_folders() -> tuple[Path, Path]:
    result = run_powershell("[Environment]::GetFolderPath('Desktop'); [Environment]::GetFolderPath('Programs')")
    folders = [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if len(folders) < 2:
        return Path.home() / "Desktop", Path.home() / r"AppData\Roaming\Microsoft\Windows\Start Menu\Programs"
    return folders[0], folders[1]


def create_shortcut(link: Path) -> bool:
    script = (
        "$ws = New-Object -ComObject WScript.Shell; "
        f"$s = $ws.CreateShortcut('{_ps_quote(link)}'); "
        f"$s.TargetPath = '{_ps_quote(PYTHONW)}'; "
        f"$s.Arguments = '\"{_ps_quote(APP_SCRIPT)}\"'; "
        f"$s.WorkingDirectory = '{_ps_quote(DESKTOP_APP)}'; "
        f"$s.IconLocation = '{_ps_quote(ASSETS / 'icon.ico')},0'; "
        "$s.Description = 'Starts the agent and listens for the wake word'; "
        "$s.Save()"
    )
    result = run_powershell(script)
    if result.returncode != 0 or not link.exists():
        print(f"[X] Could not create {link}: {result.stderr.strip()[:200]}")
        return False
    print(f"[OK] Shortcut: {link}")
    return True


def remove_old_shortcuts(desktop: Path) -> None:
    for link in {desktop / f"{OLD_NAME}.lnk", Path.home() / "Desktop" / f"{OLD_NAME}.lnk"}:
        if not link.exists():
            continue
        result = run_powershell(
            f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{_ps_quote(link)}'); $s.Arguments"
        )
        if "localhost:8000/local" in result.stdout:
            link.unlink()
            print(f"[OK] Removed the old shortcut {link} (it never started the backend)")
        else:
            print(f"[i] Left {link} alone: it does not open the old Local Agent page")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-shortcuts", action="store_true", help="install packages and the icon only")
    args = parser.parse_args()

    if sys.platform != "win32":
        print("[X] The desktop app targets Windows.")
        return 1
    if not PYTHONW.exists():
        print(f"[X] Backend virtual environment not found: {PYTHONW}")
        return 1
    if not install_packages():
        return 1
    if not webview2_installed():
        print("[!] Microsoft Edge WebView2 was not found. Install it from "
              "https://developer.microsoft.com/microsoft-edge/webview2/ and run this again.")
        return 1
    print("[OK] Microsoft Edge WebView2 is present")
    draw_icon()
    if args.no_shortcuts:
        return 0

    desktop, programs = shortcut_folders()
    ok = create_shortcut(desktop / f"{NAME}.lnk")
    ok = create_shortcut(programs / f"{NAME}.lnk") and ok
    remove_old_shortcuts(desktop)

    print()
    print(f'Open "{NAME}" from your Desktop or Start menu.')
    print("  - It starts the agent in the background and opens its window.")
    print('  - Say "hello", your task, then "done".')
    print("  - Closing the window keeps it listening in the tray; right-click the tray icon to quit.")
    print('  - Tray menu > "Start with Windows" keeps it listening right after you log in.')
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
