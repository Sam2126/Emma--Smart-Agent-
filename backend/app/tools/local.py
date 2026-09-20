"""
Local computer access tools for the agent engine (Windows).

Lets the agent operate the user's laptop alongside the browser:
find/open/read/write local files, list recent downloads, launch apps, and run
shell commands. Pure stdlib + PowerShell — no extra dependencies.

Safety: gated by settings.local_tools_enabled / shell_commands_enabled (both
default True per user request for full local access). Commands run as the
logged-in user with normal user permissions.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Type

import structlog
from app.tools.base import BaseTool
from pydantic import BaseModel, Field

from app.tools.runtime import on_main_loop, run_in_worker as _run_in_worker
from app.config import get_settings

logger = structlog.get_logger(__name__)

# Where Windows really keeps these folders. Desktop and Documents are commonly
# redirected into OneDrive: found 2026-09-16, "create a folder on the Desktop"
# wrote to C:\Users\samar\Desktop while the user's real desktop was
# C:\Users\samar\OneDrive\Desktop, so nothing appeared on screen and the task
# looked like it had done nothing.
_KNOWN_FOLDER_GUIDS = {
    "desktop": "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}",
    "documents": "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}",
    "downloads": "{374DE290-123F-4565-9164-39C4925E467B}",
    "pictures": "{33E28130-4E1E-4676-835A-98395C3BC3BB}",
    "videos": "{18989B1D-99B5-455B-841C-AB7C74E4DDFC}",
    "music": "{4BD8D571-6D19-48D3-BE97-422220080E43}",
}
# The same registry key also carries the older names for some of them.
_KNOWN_FOLDER_LEGACY_NAMES = {
    "desktop": "Desktop", "documents": "Personal", "pictures": "My Pictures",
    "videos": "My Video", "music": "My Music",
}
_SHELL_FOLDERS_KEY = r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders"
_DEFAULT_FOLDER_NAMES = {
    "desktop": "Desktop", "documents": "Documents", "downloads": "Downloads",
    "pictures": "Pictures", "videos": "Videos", "music": "Music",
}


def _known_folder(name: str) -> Path:
    """The user's real Desktop/Documents/Downloads/... folder, redirection included."""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _SHELL_FOLDERS_KEY) as key:
            for value_name in (_KNOWN_FOLDER_GUIDS.get(name), _KNOWN_FOLDER_LEGACY_NAMES.get(name)):
                if not value_name:
                    continue
                try:
                    raw, _ = winreg.QueryValueEx(key, value_name)
                except OSError:
                    continue
                expanded = os.path.expandvars(raw or "")
                if expanded:
                    return Path(expanded)
    except Exception:
        pass
    return Path.home() / _DEFAULT_FOLDER_NAMES.get(name, name.capitalize())


def _downloads_dir() -> Path:
    """Downloads folder — from Windows itself, fallback to the default location."""
    return _known_folder("downloads")


_LOCATION_MAP = {
    "downloads": _downloads_dir,
    "desktop": lambda: _known_folder("desktop"),
    "documents": lambda: _known_folder("documents"),
    "pictures": lambda: _known_folder("pictures"),
    "videos": lambda: _known_folder("videos"),
    "music": lambda: _known_folder("music"),
    "home": lambda: Path.home(),
}


def _resolve_location(location: str) -> Path:
    loc = (location or "downloads").strip().lower()
    if loc in _LOCATION_MAP:
        return Path(_LOCATION_MAP[loc]())
    return _resolve_user_path(location)


_AGENT_TEMP = Path(tempfile.gettempdir()) / "self_improving_agent"


# "Desktop" written as a plain sub-folder of the user's profile. Windows keeps a
# leftover empty C:\Users\<name>\Desktop on a redirected machine, so such a path
# is valid, writable, and invisible to the user.
_HOME_FOLDER_ALIASES = {
    "desktop": "desktop",
    "documents": "documents", "my documents": "documents",
    "downloads": "downloads",
    "pictures": "pictures", "my pictures": "pictures",
    "videos": "videos", "my videos": "videos",
    "music": "music", "my music": "music",
}


def _redirect_known_folder(path: Path) -> Path:
    """Send C:\\Users\\<name>\\Desktop\\x to the real Desktop when Windows redirects it.

    Found 2026-09-16, after the folder lookup below was fixed: the model still
    built an absolute '%USERPROFILE%\\Desktop\\MAYANK\\prime_code.txt' itself, so
    the folder was created on the real desktop while the file inside it went to
    the invisible leftover folder. An absolute path is only rewritten when it
    names a folder Windows actually keeps somewhere else.
    """
    try:
        relative = path.relative_to(Path.home())
    except ValueError:
        return path
    parts = relative.parts
    if not parts:
        return path
    key = _HOME_FOLDER_ALIASES.get(parts[0].lower())
    if key is None:
        return path
    real = Path(_known_folder(key))
    if real == Path.home() / parts[0]:
        return path  # not redirected: the literal path is the right one
    return real.joinpath(*parts[1:])


def _resolve_user_path(raw: str) -> Path:
    """Expand %VARS% and ~, and read 'Desktop\\MAYANK' as the user's REAL Desktop.

    A path that starts with a known folder name (Desktop, Documents, Downloads,
    Pictures, Videos, Music) is resolved through Windows, so OneDrive-redirected
    folders end up where the user actually sees them.
    """
    text = os.path.expandvars((raw or "").strip().strip('"'))
    path = Path(text).expanduser()
    if path.is_absolute():
        return _redirect_known_folder(path)
    parts = path.parts
    if parts and parts[0].lower() in _LOCATION_MAP:
        return Path(_LOCATION_MAP[parts[0].lower()]()).joinpath(*parts[1:])
    return Path.home() / path


# How a script spells the user's folder before it builds a path of its own.
_SHELL_HOME_PREFIXES = ("$env:USERPROFILE", "%USERPROFILE%", "$HOME", "~")


def _redirect_shell_script(script: str) -> str:
    """Point '$env:USERPROFILE\\Desktop' in a script at the user's real Desktop.

    A script builds its own paths, so the path handling above never sees them.
    Found 2026-09-16: the strategy the agent recalled for this very task was
    'New-Item -ItemType Directory -Path "$env:USERPROFILE\\Desktop\\MAYANK"',
    which on a redirected machine creates a folder the user cannot see.
    """
    for key in ("desktop", "documents", "pictures", "videos", "music"):
        name = _DEFAULT_FOLDER_NAMES[key]
        real = _known_folder(key)
        if real == Path.home() / name:
            continue  # Windows keeps it where the script expects it
        for prefix in _SHELL_HOME_PREFIXES:
            for separator in ("\\", "/"):
                script = re.sub(
                    re.escape(f"{prefix}{separator}{name}") + r"(?=[\\/\"'\s]|$)",
                    str(real).replace("\\", "\\\\"),
                    script,
                    flags=re.IGNORECASE,
                )
    return script


def _run_powershell(script: str, timeout_seconds: int = 60) -> subprocess.CompletedProcess:
    """Run a PowerShell script from a file, never from a -Command string.

    Found 2026-09-15: the agent's commands were passed as -Command strings and
    three attempts in a row died on quoting ("The string is missing the
    terminator") before a folder could be created. A script file has no
    quoting layer at all.
    """
    _AGENT_TEMP.mkdir(parents=True, exist_ok=True)
    script_file = _AGENT_TEMP / f"cmd_{os.getpid()}_{int(time.time() * 1000)}.ps1"
    script_file.write_text(_redirect_shell_script(script), encoding="utf-8-sig")
    try:
        return subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script_file)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout_seconds,
        )
    finally:
        try:
            script_file.unlink()
        except OSError:
            pass


def _check_enabled() -> str | None:
    if not get_settings().local_tools_enabled:
        return "Local computer tools are disabled (local_tools_enabled=false)."
    return None


# A file can vanish, or refuse to be read, between listing a folder and asking
# for its details; one such file must not fail the whole listing.
def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _stamp(path: Path) -> str:
    moment = _mtime(path)
    return datetime.fromtimestamp(moment).strftime("%Y-%m-%d %H:%M") if moment else "(unknown date)"


def _is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


# Folders full of generated files that nobody means by "search my files":
# environments, dependencies, caches and build output. Only folders BELOW the
# search root are skipped; searching inside one directly still works.
_SKIP_DIRS = {
    ".venv", "venv", "env", "node_modules", "bower_components", ".git", "__pycache__", "site-packages",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".cache", ".turbo",
    ".next", ".nuxt", "dist", "build", "target", "obj", "coverage", ".gradle", ".idea", ".vs", ".pio",
    "$recycle.bin", "appdata",
}


def _walk_files(root: Path, deadline: float):
    """Files under `root`, nearest first, skipping generated-file folders, until `deadline`.

    Breadth-first on purpose. Found 2026-09-16: a Desktop search went depth-first
    into a project tree (30,000 files of ML datasets and a .next build) and spent
    its whole 20 s budget there, never reaching shallow folders — a first run found
    a match in "Virtual Machine setup", a second run with a different order found
    nothing and the agent reported "no matches". "Search my Desktop" means the
    files near the top first.
    """
    pending = deque([root])
    while pending:
        folder = pending.popleft()
        try:
            with os.scandir(folder) as listing:
                entries = sorted(listing, key=lambda entry: entry.name.lower())
        except OSError:
            continue
        for entry in entries:
            if time.monotonic() > deadline:
                return
            try:
                if entry.is_dir(follow_symlinks=False):
                    if entry.name.lower() not in _SKIP_DIRS:
                        pending.append(Path(entry.path))
                elif entry.is_file():
                    yield Path(entry.path)
            except OSError:
                continue


# =============================================================================
# 1. List recent files (Downloads / Desktop / Documents / ...)
# =============================================================================

class ListRecentFilesInput(BaseModel):
    location: str = Field(default="downloads", description="Which folder: downloads, desktop, documents, pictures, videos, music, home — or a full path.")
    count: int = Field(default=10, description="How many files to list (newest first).")


class ListRecentFilesTool(BaseTool):
    name: str = "list_recent_files"
    description: str = (
        "Lists the most recent files in a local folder (newest first). "
        "Use for requests like 'open my latest download' or 'what did I download today'. "
        "Returns full paths with modified timestamps."
    )
    args_schema: Type[BaseModel] = ListRecentFilesInput

    def _run(self, location: str = "downloads", count: int = 10) -> str:
        async def _async_run():
            blocked = _check_enabled()
            if blocked:
                return blocked
            folder = _resolve_location(location)
            if not folder.exists():
                return f"Folder not found: {folder}"
            files = [f for f in folder.iterdir() if _is_file(f)]
            files.sort(key=_mtime, reverse=True)
            if not files:
                return f"No files found in {folder}"
            lines = [f"Recent files in {folder} (newest first):"]
            for f in files[: max(1, min(count, 50))]:
                lines.append(f"  {_stamp(f)}  {f}")
            return "\n".join(lines)

        return _run_in_worker(_async_run)


# =============================================================================
# 2. Find files by name pattern
# =============================================================================

_FIND_TIME_BUDGET_S = 6.0


class FindFilesInput(BaseModel):
    pattern: str = Field(..., description="File name or partial name to search for (e.g. 'invoice', 'report.pdf').")
    location: str = Field(default="home", description="Where to search: downloads, desktop, documents, home (recursive), or a full path.")
    max_results: int = Field(default=15, description="Max matches to return.")


class FindFilesTool(BaseTool):
    name: str = "find_files"
    description: str = (
        "Searches the local computer for files whose NAME contains the pattern. "
        "Searches recursively under the chosen location. Use when the user asks "
        "to find/open a specific file and you don't know its path."
    )
    args_schema: Type[BaseModel] = FindFilesInput

    def _run(self, pattern: str, location: str = "home", max_results: int = 15) -> str:
        async def _async_run():
            blocked = _check_enabled()
            if blocked:
                return blocked
            folder = _resolve_location(location)
            if not folder.exists():
                return f"Folder not found: {folder}"
            needle = pattern.lower().replace("*", "")
            limit = max(1, min(max_results, 50))
            matches = []
            deadline = time.monotonic() + _FIND_TIME_BUDGET_S  # a deep folder must not hang the task
            for f in _walk_files(folder, deadline):
                if needle in f.name.lower():
                    matches.append(f)
                    if len(matches) >= 2000:
                        break
            cut_off = time.monotonic() > deadline
            unfinished = (
                f"\n(The search stopped after {int(_FIND_TIME_BUDGET_S)} s before covering all of {folder}; "
                "more files may match. Search a narrower folder to be sure.)" if cut_off else ""
            )
            if not matches:
                return f"No files matching '{pattern}' under {folder}{unfinished}"
            # Newest first, because "the last ppt from downloads" means the most
            # recent one. Found 2026-09-16: matches were returned in folder order
            # and the scan stopped at max_results, so the newest .pptx was
            # usually not among them and the agent would attach the wrong file.
            matches.sort(key=_mtime, reverse=True)
            matches = matches[:limit]
            lines = [f"Found {len(matches)} file(s) matching '{pattern}' (newest first):"]
            for f in matches:
                lines.append(f"  {_stamp(f)}  {f}")
            return "\n".join(lines) + unfinished

        return _run_in_worker(_async_run)


# =============================================================================
# 3. Open a file or folder with its default Windows app
# =============================================================================

class OpenPathInput(BaseModel):
    path: str = Field(..., description="Full path of the file or folder to open.")


class OpenPathTool(BaseTool):
    name: str = "open_file_or_folder"
    description: str = (
        "Opens a local file or folder on the user's computer with its default "
        "application (PDF in a PDF viewer, folder in Explorer, etc.). "
        "If you only know the name, find the full path first with find_files "
        "or list_recent_files."
    )
    args_schema: Type[BaseModel] = OpenPathInput

    def _run(self, path: str) -> str:
        async def _async_run():
            blocked = _check_enabled()
            if blocked:
                return blocked
            if _WEB_ADDRESS.match(path or ""):
                return (
                    f"Not opened: '{path}' is a web address. Open it with navigate_browser, in the agent's own "
                    "Chrome window; opening it here would use the user's own Chrome."
                )
            p = _resolve_user_path(path)
            if _EMAIL_SETUP.search(p.name):
                return _EMAIL_SETUP_REFUSAL
            if not p.exists():
                return f"Path not found: {p}. Use find_files or list_recent_files to locate it."
            os.startfile(str(p))  # noqa: S606 - intentional local action
            logger.info("local_path_opened", path=str(p))
            return f"Opened: {p}"

        return _run_in_worker(_async_run)


# =============================================================================
# 4. Read a text file
# =============================================================================

class ReadFileInput(BaseModel):
    path: str = Field(..., description="Full path of the file to read.")
    max_chars: int = Field(default=4000, description="Max characters to return.")


_DOCUMENT_SUFFIXES = {".pdf", ".docx", ".pptx", ".xlsx"}
_MAX_TEXT_READ_BYTES = 2 * 1024 * 1024


def _document_text(path: Path) -> tuple[str, str | None]:
    """Text of a PDF, Word, PowerPoint or Excel file. Returns (text, problem)."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            pages = [f"--- page {i + 1} ---\n{(page.extract_text() or '').strip()}" for i, page in enumerate(reader.pages[:50])]
            return "\n".join(pages), None
        if suffix == ".docx":
            import docx

            document = docx.Document(str(path))
            lines = [para.text for para in document.paragraphs]
            for table in document.tables:
                for row in table.rows:
                    lines.append(" | ".join(cell.text.strip() for cell in row.cells))
            return "\n".join(line for line in lines if line.strip()), None
        if suffix == ".pptx":
            from pptx import Presentation

            slides = []
            for i, slide in enumerate(Presentation(str(path)).slides, start=1):
                texts = [shape.text.strip() for shape in slide.shapes if getattr(shape, "has_text_frame", False) and shape.text.strip()]
                slides.append(f"--- slide {i} ---\n" + "\n".join(texts))
            return "\n".join(slides), None
        if suffix == ".xlsx":
            import openpyxl

            workbook = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
            parts = []
            for sheet in workbook.worksheets[:10]:
                parts.append(f"--- sheet {sheet.title} ---")
                for row in sheet.iter_rows(max_row=200, values_only=True):
                    cells = [str(value) for value in row if value is not None]
                    if cells:
                        parts.append(" | ".join(cells))
            workbook.close()
            return "\n".join(parts), None
    except ImportError as e:
        return "", f"Could not read {path}: the reader for {suffix} files is not installed ({e})."
    except Exception as e:
        return "", f"Could not read {path}: {e}"
    return "", f"Could not read {path}: unsupported document type."


class ReadFileTool(BaseTool):
    name: str = "read_file"
    description: str = (
        "Reads a local file and returns its text: plain text and code files, and also PDF, Word (.docx), "
        "PowerPoint (.pptx) and Excel (.xlsx) documents. Use it to inspect or summarise a document."
    )
    args_schema: Type[BaseModel] = ReadFileInput

    def _run(self, path: str, max_chars: int = 4000) -> str:
        async def _async_run():
            blocked = _check_enabled()
            if blocked:
                return blocked
            p = _resolve_user_path(path)
            if not p.exists():
                return f"File not found: {p}"
            cut_short = False
            if p.suffix.lower() in _DOCUMENT_SUFFIXES:
                text, problem = _document_text(p)
                if problem:
                    return problem
            elif p.is_dir():
                return f"Could not read {p}: it is a folder. Use list_folder to see what is inside."
            else:
                # Only the beginning is read: asked about a large video or archive,
                # the whole file used to be loaded into memory to return 4,000 characters.
                try:
                    with open(p, "rb") as handle:
                        raw = handle.read(_MAX_TEXT_READ_BYTES + 1)
                except Exception as e:
                    return f"Could not read {p}: {e}"
                cut_short = len(raw) > _MAX_TEXT_READ_BYTES
                text = raw[:_MAX_TEXT_READ_BYTES].decode("utf-8", errors="replace")
            head = text[: max(200, min(max_chars, 20000))]
            if cut_short:
                more = "\n... (a large file: only its beginning was read)"
            elif len(text) > len(head):
                more = f"\n... ({len(text)} chars total, truncated)"
            else:
                more = ""
            return f"Contents of {p}:\n{head}{more}"

        return _run_in_worker(_async_run)


# =============================================================================
# 5. Write / append a text file
# =============================================================================

class WriteFileInput(BaseModel):
    path: str = Field(..., description="Full path of the file to write.")
    content: str = Field(..., description="Text content to write.")
    append: bool = Field(default=False, description="Append instead of overwrite.")


class WriteFileTool(BaseTool):
    name: str = "write_file"
    description: str = "Writes (or appends) text to a local file. Creates parent folders if needed."
    args_schema: Type[BaseModel] = WriteFileInput

    def _run(self, path: str, content: str, append: bool = False) -> str:
        async def _async_run():
            blocked = _check_enabled()
            if blocked:
                return blocked
            p = _resolve_user_path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with open(p, mode, encoding="utf-8") as f:
                f.write(content)
            logger.info("local_file_written", path=str(p), append=append)
            return f"{'Appended to' if append else 'Wrote'} {p} ({len(content)} chars)."

        return _run_in_worker(_async_run)


# =============================================================================
# 6. List a folder's contents
# =============================================================================

class ListFolderInput(BaseModel):
    path: str = Field(..., description="Full path of the folder to list.")


class ListFolderTool(BaseTool):
    name: str = "list_folder"
    description: str = "Lists files and subfolders inside a local folder."
    args_schema: Type[BaseModel] = ListFolderInput

    def _run(self, path: str) -> str:
        async def _async_run():
            blocked = _check_enabled()
            if blocked:
                return blocked
            p = _resolve_user_path(path)
            if not p.exists():
                return f"Folder not found: {p}"
            entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
            lines = [f"Contents of {p}:"]
            for e in entries[:100]:
                kind = "DIR " if e.is_dir() else "FILE"
                lines.append(f"  [{kind}] {e.name}")
            return "\n".join(lines)

        return _run_in_worker(_async_run)


# =============================================================================
# 7. Open an application
# =============================================================================

def _wait_for_app_window(hint: str, timeout_ms: int = 25000) -> tuple[str | None, str]:
    """Block until a window matching `hint` exists AND the app has finished starting.

    Waiting for the window alone was not enough: a cold WhatsApp start shows a
    splash screen for ~8s after its window appears, and the agent acted on the
    splash (see tools/desktop.wait_until_app_ready for the measurements).

    Uses the same scored window resolver the desktop GUI tools use, so "the
    window open_app waited for" and "the window see_window/click_window/
    send_keys act on" are always the same one.

    Returns (description, status) — status is "ready", "loading" or "no_window".
    """
    from app.tools.desktop import wait_until_app_ready

    try:
        return wait_until_app_ready(hint, window_timeout_ms=timeout_ms)
    except Exception as e:
        logger.warning("wait_for_app_ready_failed", hint=hint, error=str(e)[:150])
        return None, "no_window"


def _launchable_command(target: str) -> bool:
    """A file path, a program on PATH, a registered App Path (e.g. 'excel') or a URI like 'ms-settings:'."""
    import re
    import shutil
    import winreg

    t = (target or "").strip().strip('"')
    if not t:
        return False
    if re.match(r"^[A-Za-z][\w+.-]+:", t) and not re.match(r"^[A-Za-z]:[\\/]", t):
        return True
    if Path(os.path.expandvars(t)).exists():
        return True
    if shutil.which(t) or shutil.which(f"{t}.exe"):
        return True
    exe = t if t.lower().endswith(".exe") else f"{t}.exe"
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(hive, rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{exe}"):
                return True
        except OSError:
            continue
    return False


# The agent never opens or closes the user's own Chrome: it works in its own
# Chrome window (app/browser/controller.py). Found 2026-09-17: open_app('Chrome')
# started the user's Chrome on its last-used profile, so a second task landed
# in a different profile from the first.
_CHROME_APP_NAME = re.compile(
    r"^\s*(?:google\s+)?chrome(?:\.exe)?(?:\s+browser)?\s*$|^\s*(?:the\s+|web\s+)?browser\s*$",
    re.IGNORECASE,
)
_WEB_ADDRESS = re.compile(r"^\s*(?:https?://|www\.)", re.IGNORECASE)
# PowerShell that would open a page in the user's default browser (their Chrome),
# start their Chrome, or close it.
_TOUCHES_USERS_CHROME = re.compile(
    r"(?:^|[\s;&|(])(?:start-process|saps|start|invoke-item|ii|explorer(?:\.exe)?)\s+"
    r"(?:-filepath\s+)?(?:[\"']{2}\s+)?[\"']?(?:[^\"'\s;|]*[\\/])?(?:chrome(?:\.exe)?(?=[\"'\s;|]|$)|https?://|www\.)"
    r"|chrome\.exe"
    r"|(?:stop-process|spps|kill|taskkill)\b[^;|]*\bchrome\b"
    r"|::start\(\s*[\"'](?:https?://|www\.)",
    re.IGNORECASE,
)


# The email setup asks for the user's own password: only the user runs it.
# Found 2026-09-17: told that email was not set up, the agent found
# setup_email.bat and ran it itself.
_EMAIL_SETUP = re.compile(r"setup_email", re.IGNORECASE)
_EMAIL_SETUP_REFUSAL = (
    "Not run: the email setup asks for the user's own email password, so only the user can run it "
    "(setup_email.bat in the project folder). Tell the user that; do not start it."
)


async def _open_agent_chrome() -> str:
    """Open the agent's own Chrome window, or bring it to the front."""
    from app.browser.registry import get_browser_controller
    from app.main import ensure_browser_connection
    from app.tools.desktop import CHROME_IS_BROWSER_TOOLS

    if not await ensure_browser_connection():
        return (
            "Could not open the agent's Chrome window. Try navigate_browser, which opens it too, "
            "and report the problem if that fails as well."
        )
    try:
        page = await get_browser_controller().open_agent_tab()
    except Exception as e:
        logger.warning("agent_chrome_open_failed", error=str(e)[:150])
        return (
            f"Could not open the agent's Chrome window ({str(e)[:150]}). Try navigate_browser, which "
            "opens it too, and report the problem if that fails as well."
        )
    try:
        await page.bring_to_front()
    except Exception as e:
        logger.warning("agent_chrome_front_failed", error=str(e)[:150])
    logger.info("local_app_opened", app="chrome", method="agent_window")
    return (
        "Launched: the agent's own Chrome window ('Self-Improving Agent'); the user's own Chrome "
        f"profiles are not used. {CHROME_IS_BROWSER_TOOLS}"
    )


class OpenAppInput(BaseModel):
    name: str = Field(..., description="App or command to launch (e.g. 'notepad', 'calc', 'mspaint', 'explorer', 'WhatsApp', or an .exe path).")


class OpenAppTool(BaseTool):
    name: str = "open_app"
    description: str = (
        "Launches a Windows application by name (notepad, calc, mspaint, explorer, code, WhatsApp, etc. — "
        "Store/UWP apps included). 'chrome' opens the agent's own Chrome window, which is then driven with the "
        "browser tools. To get an app that is not installed yet, use install_app."
    )
    args_schema: Type[BaseModel] = OpenAppInput

    def _run(self, name: str) -> str:
        async def _async_run():
            blocked = _check_enabled()
            if blocked:
                return blocked
            target = name.strip()
            if not target:
                return "No app name given."
            if _CHROME_APP_NAME.match(target):
                return await on_main_loop(_open_agent_chrome())

            # Resolve via Get-StartApps first — the reliable, universal way
            # to launch ANY Start-Menu-registered app. On current Windows
            # builds this covers Store/UWP apps (WhatsApp) AND classic ones
            # (Notepad, Calculator, Paint are UWP-packaged here too; Chrome
            # and VS Code also resolve through it) — verified directly
            # against this machine. The plain `cmd /c start AppName` this
            # tool used before silently "succeeded" (always returned
            # "Launched: X") even when Windows popped an error dialog and
            # nothing actually opened, e.g. for 'WhatsApp' — that false
            # positive was also getting recorded as a learned success.
            from app.tools.desktop import _PS_UTF8

            safe_name = target.replace("'", "''")
            ps_lookup = (
                _PS_UTF8
                + f"$m = Get-StartApps | Where-Object {{ $_.Name -like '*{safe_name}*' }} "
                "| Select-Object -First 1; "
                "if ($m) { Write-Output ('FOUND|' + $m.AppID) } else { Write-Output 'NOTFOUND' }"
            )
            try:
                lookup = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps_lookup],
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                    timeout=15,
                )
                output = (lookup.stdout or "").strip()
            except Exception as e:
                output = ""
                logger.warning("local_app_lookup_failed", app=name, error=str(e)[:150])

            if output.startswith("FOUND|"):
                app_id = output.split("|", 1)[1].strip()
                subprocess.Popen(
                    ["explorer.exe", f"shell:AppsFolder\\{app_id}"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                logger.info("local_app_opened", app=name, app_id=app_id, method="start_apps")
                # Launching is asynchronous: explorer returns instantly while
                # the app takes seconds to map a window (WhatsApp measured at
                # ~6s before its window handle exists). Previously this tool
                # returned immediately, so the very next step — see_window or
                # send_keys — found no window and gave up, which is exactly why
                # tasks stalled right after the app opened. Wait for the window
                # here so the next step always has something to act on.
                appeared, status = _wait_for_app_window(target, timeout_ms=25000)
                if appeared and status == "ready":
                    return (
                        f"Launched: {name} (Start Menu app id {app_id}). "
                        f"It has finished starting and is ready: {appeared}. "
                        "Continue with the rest of the task now."
                    )
                if appeared:
                    return (
                        f"Launched: {name} (Start Menu app id {app_id}). Its window "
                        f"({appeared}) is open but was still showing its loading screen "
                        "after 45s. Wait a few seconds, then call see_window before typing. "
                        "A logo or progress bar is a loading screen, not a login screen."
                    )
                return (
                    f"Launched: {name} (Start Menu app id {app_id}), but no window appeared "
                    "within 25s. It may still be loading or may have opened minimised — "
                    "call list_windows to check before sending any keys."
                )

            # Fallback: not a Start-Menu app. Only a real file path, a program
            # on PATH, a registered program (App Paths, e.g. 'excel') or a URI
            # such as 'ms-settings:' is started. Found 2026-09-15: "open gemini"
            # ran `cmd /c start Gemini`, which cannot work (Gemini is a
            # website), can pop Windows' "cannot find" error on screen, and was
            # recorded as a successful launch.
            if not _launchable_command(target):
                logger.info("local_app_not_found", app=name)
                return (
                    f"Could not open '{name}': no installed app or program with that name was found. "
                    "If it is a website or web app (for example Gemini, ChatGPT, YouTube, Gmail), open it "
                    "with navigate_browser instead. Otherwise use find_files to locate the program."
                )
            subprocess.Popen(
                ["cmd", "/c", "start", "", target],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            logger.info("local_app_opened", app=name, method="cmd_start_fallback")
            appeared, status = _wait_for_app_window(target, timeout_ms=8000)
            if appeared:
                return (
                    f"Launched: {name}. Its window is open: {appeared}. "
                    + ("It is still loading; wait a moment before typing." if status == "loading"
                       else "Continue with the rest of the task now.")
                )
            return (
                f"Started '{name}' with the Windows start command, but no window matching it was seen "
                "within 8s. Call list_windows to check before continuing."
            )

        return _run_in_worker(_async_run)


# =============================================================================
# 8. Run a shell command (full local control)
# =============================================================================

class RunCommandInput(BaseModel):
    command: str = Field(..., description="PowerShell command to run (e.g. 'Get-Volume', 'Get-Process | Select -First 5').")
    timeout_seconds: int = Field(default=60, description="Max seconds to wait for the command.")


class RunCommandTool(BaseTool):
    name: str = "run_command"
    description: str = (
        "Runs a WINDOWS POWERSHELL command ONLY on the user's computer and returns "
        "its output. This is Windows PowerShell — never bash, sh, AppleScript/osascript, "
        "xdotool, or any Linux/macOS syntax; those will always fail here. "
        "Use for local device actions: system settings, volume, network info, "
        "processes, moving/copying files, anything not covered by other tools. "
        "To type text or send keystrokes into whatever window currently has focus "
        "(e.g. to search inside an app you just opened), use the dedicated "
        "send_keys tool instead of trying to script this with PowerShell. "
        "Prefer the dedicated file tools (find/open/read/write) for simple file work."
    )
    args_schema: Type[BaseModel] = RunCommandInput

    def _run(self, command: str, timeout_seconds: int = 60) -> str:
        async def _async_run():
            blocked = _check_enabled()
            if blocked:
                return blocked
            if not get_settings().shell_commands_enabled:
                return "Shell commands are disabled (shell_commands_enabled=false)."
            # Security rail: block obviously destructive system-level commands.
            # Everything else runs with normal user permissions (no admin).
            lowered = command.lower()
            catastrophic = (
                "del /f /s /q c:", "rd /s /q c:", "remove-item -recurse c:",
                "remove-item -recurse -force c:", "rm -rf /", "diskpart", "bcdedit",
                "cipher /w", "vssadmin delete", "shutdown", "restart-computer", "stop-computer",
                "format-volume", "clear-disk",
                "-encodedcommand", "reg delete hklm", "reg delete hku",
            )
            # "format" only as the disk command (format C:, format.com d: /q).
            # Found 2026-09-16: the plain substring "format " also blocked
            # everyday commands such as Get-Date -Format "yyyy-MM-dd".
            formats_a_disk = re.search(r"(?:^|[\s;&|(])format(?:\.com)?\s+[a-z]:", lowered) is not None
            if _TOUCHES_USERS_CHROME.search(command):
                logger.warning("local_command_blocked_users_chrome", command=command[:120])
                return (
                    "Not run: this command would open, start or close the user's own Chrome. Web pages are "
                    "opened with navigate_browser, in the agent's own Chrome window."
                )
            if _EMAIL_SETUP.search(command):
                logger.warning("local_command_blocked_email_setup", command=command[:120])
                return _EMAIL_SETUP_REFUSAL
            if formats_a_disk or any(c in lowered for c in catastrophic):
                logger.warning("local_command_blocked_catastrophic", command=command[:120])
                return (
                    "BLOCKED for safety: this command could destroy system data or "
                    "your OS install and will not be executed. If it is truly needed, "
                    "run it yourself in an admin terminal."
                )
            try:
                proc = _run_powershell(command, timeout_seconds=max(5, min(timeout_seconds, 300)))
                out = (proc.stdout or "").strip()
                err = (proc.stderr or "").strip()
                result = out if out else ("(no output)" if not err else "")
                if err:
                    result += f"\n[stderr] {err[:1500]}"
                logger.info("local_command_run", command=command[:80], code=proc.returncode)
                return f"Exit code {proc.returncode}:\n{result[:8000]}"
            except subprocess.TimeoutExpired:
                return f"Command timed out after {timeout_seconds}s."
            except Exception as e:
                return f"Command failed: {e}"

        return _run_in_worker(_async_run)


# =============================================================================
# 9. Send keystrokes to a specific window (force-focused first)
# =============================================================================
# Added because the model, lacking any real way to "type into the app I just
# opened", was inventing bash/xdotool/AppleScript commands via run_command —
# which is Windows PowerShell only, so those always failed instantly and
# burned iterations on a task before timing out.
#
# The first version sent keys to "whatever window currently has focus" without
# forcing focus itself, which is not reliable: this backend runs as a
# background process and Windows does not hand it the foreground just because
# it recently launched an app. That produced tasks reporting "message sent" in
# full narrative detail while nothing after the app launch had actually
# happened. So `window_hint` is required, and no keys are sent unless the
# target window was found AND confirmed focused.
#
# The second version still failed for WhatsApp specifically, for two reasons
# found by driving it live on this machine:
#
#   * It selected the target with `Get-Process | ... | Select-Object -First 1`,
#     filtered on MainWindowHandle. WhatsApp is a WebView2 app, so BOTH
#     `WhatsApp.Root` and the `msedgewebview2` child expose a window titled
#     "WhatsApp"; the child usually won, and focusing it did not give keyboard
#     focus to the real app. It also meant this tool and the see/click tools
#     could each pick a different window.
#   * It typed via .NET `SendKeys.SendWait`, which drives input through legacy
#     journal-playback hooks. Chromium/WebView2 content ignores those, so the
#     call reported success while WhatsApp received nothing (confirmed by
#     screenshotting the window afterwards: the search box stayed empty).
#
# Both are now handled by the shared helper in tools/desktop.py: `Find` scores
# candidate windows so the real application window wins and every desktop tool
# resolves to the same one, and `Type`/`Keys` inject through `SendInput`
# (KEYEVENTF_UNICODE for literal text), which Chromium-based UI does accept.

class SendKeysInput(BaseModel):
    window_hint: str = Field(..., description="Process name or window title substring identifying the target window to focus before typing (e.g. 'WhatsApp', 'notepad'). Required — use the app name you passed to open_app.")
    text: str = Field(default="", description="Literal text to type into the target window, typed exactly as given (no escaping needed).")
    keys: str = Field(default="", description="Shortcut/special-key notation, e.g. '^f' for Ctrl+F, '{ENTER}', '{TAB}', '{ESC}', '{DOWN}'.")
    delay_ms: int = Field(default=600, description="Milliseconds to wait after focusing before sending, giving a just-opened window time to finish rendering.")


class SendKeysTool(BaseTool):
    name: str = "send_keys"
    description: str = (
        "Types into or sends keyboard shortcuts to a specific app's window on Windows — "
        "use this to type a search term, a chat message, or press Enter. "
        "REQUIRES 'window_hint' (the app/process name, e.g. 'WhatsApp') so the target "
        "window can be found and forced to the foreground before anything is sent — "
        "this tool does NOT type into whatever happens to have focus, and it refuses to "
        "send anything if it cannot confirm the window is focused. Pass 'text' for "
        "literal text and/or 'keys' for shortcuts ('^f' = Ctrl+F, '{ENTER}', '{TAB}', "
        "'{ESC}', '{DOWN}'). "
        "IMPORTANT ORDER NOTE: if both are given in ONE call, 'keys' is sent BEFORE "
        "'text', not after — so to type a message and then press Enter to send it, make "
        "TWO separate calls: first send_keys(text=<message>), then send_keys(keys='{ENTER}'). "
        "Typing only goes where the caret already is, so click the target box with "
        "click_window first. Do NOT attempt this via run_command; PowerShell has no "
        "bash/xdotool/AppleScript."
    )
    args_schema: Type[BaseModel] = SendKeysInput

    def _run(self, window_hint: str, text: str = "", keys: str = "", delay_ms: int = 600) -> str:
        async def _async_run():
            blocked = _check_enabled()
            if blocked:
                return blocked
            if not get_settings().shell_commands_enabled:
                return "Shell commands are disabled (shell_commands_enabled=false)."
            if not window_hint or not window_hint.strip():
                return "window_hint is required: which app's window should receive these keys?"
            if not text and not keys:
                return "Nothing to send: provide 'text' and/or 'keys'."

            from app.tools.desktop import _PS_UI_HELPER, _ps, _ps_quote, chrome_window_refusal

            refused = chrome_window_refusal(window_hint)
            if refused:
                return refused

            hint = _ps_quote(window_hint.strip())
            wait = max(0, min(delay_ms, 5000))

            # Keys first, then literal text — matches the documented order and
            # lets a caller open a search box with a shortcut and type into it
            # in a single call.
            send_lines = []
            if keys:
                send_lines.append(f"[OaskUI]::Keys('{_ps_quote(keys)}')")
            if text:
                send_lines.append(f"[OaskUI]::Type('{_ps_quote(text)}')")
            send_block = "\n".join(send_lines)

            script = _PS_UI_HELPER + f"""
$w = [OaskUI]::WaitFor('{hint}', 15000)
if (-not $w) {{ Write-Output 'WINDOW_NOT_FOUND'; exit 0 }}
if (-not [OaskUI]::Focus($w.H)) {{ Write-Output 'FOCUS_FAILED'; exit 0 }}
Start-Sleep -Milliseconds {wait}
{send_block}
Write-Output ("SENT|" + $w.Proc + "|" + $w.Title)
"""
            try:
                out = _ps(script, timeout=90)
            except Exception as e:
                return f"send_keys failed: {e}"

            if "WINDOW_NOT_FOUND" in out:
                return (
                    f"No visible window matching '{window_hint}' after waiting 15s — nothing "
                    "was sent. Check that open_app actually reported the window as open, and "
                    "call list_windows to see the exact name Windows uses."
                )
            if "FOCUS_FAILED" in out:
                return (
                    f"Found a window matching '{window_hint}' but Windows refused to bring it "
                    "to the foreground, so NOTHING was sent (sending anyway would have typed "
                    "into whatever app is actually in front). Retry, or call focus_window first."
                )
            line = next((l for l in out.splitlines() if l.startswith("SENT|")), "")
            if line:
                _, proc_name, window_title = line.split("|", 2)
                logger.info("local_send_keys", window=proc_name, title=window_title[:80], text=text[:50], keys=keys[:50])
                return (
                    f"Sent to '{window_title}' ({proc_name}): {(keys + ' ' + text).strip()[:90]}. "
                    "Call see_window to confirm it actually appeared where you expected."
                )
            return f"send_keys failed (unexpected output): {out[:500]}"

        return _run_in_worker(_async_run)


# =============================================================================
# 10. Everyday file management: create / move / copy / delete
# =============================================================================
# Added 2026-09-16: these were only possible through hand-written PowerShell,
# which kept failing on quoting. A dedicated tool cannot be mis-quoted.

class CreateFolderInput(BaseModel):
    path: str = Field(..., description=r"Folder to create, e.g. 'C:\Users\me\Desktop\MAYANK' or 'Desktop\MAYANK'. Parent folders are created too.")


class CreateFolderTool(BaseTool):
    name: str = "create_folder"
    description: str = "Creates a folder on this computer (with any missing parent folders). Use this instead of a PowerShell command."
    args_schema: Type[BaseModel] = CreateFolderInput

    def _run(self, path: str) -> str:
        async def _async_run():
            blocked = _check_enabled()
            if blocked:
                return blocked
            target = _resolve_user_path(path)
            try:
                existed = target.is_dir()
                target.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                return f"Could not create the folder {target}: {e}"
            logger.info("local_folder_created", path=str(target), existed=existed)
            return f"Folder ready: {target}" + (" (it already existed)" if existed else "")

        return _run_in_worker(_async_run)


class MovePathInput(BaseModel):
    source: str = Field(..., description="File or folder to move or rename.")
    destination: str = Field(..., description="New full path, or an existing folder to move it into.")


class MovePathTool(BaseTool):
    name: str = "move_path"
    description: str = "Moves or renames a file or folder on this computer."
    args_schema: Type[BaseModel] = MovePathInput

    def _run(self, source: str, destination: str) -> str:
        async def _async_run():
            blocked = _check_enabled()
            if blocked:
                return blocked
            src, dst = _resolve_user_path(source), _resolve_user_path(destination)
            if not src.exists():
                return f"Not found: {src}. Use find_files or list_folder to get the exact path."
            if dst.is_dir():
                dst = dst / src.name
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
            except Exception as e:
                return f"Could not move {src} to {dst}: {e}"
            logger.info("local_path_moved", source=str(src), destination=str(dst))
            return f"Moved {src} to {dst}"

        return _run_in_worker(_async_run)


class CopyPathInput(BaseModel):
    source: str = Field(..., description="File or folder to copy.")
    destination: str = Field(..., description="Full path of the copy, or an existing folder to copy it into.")


class CopyPathTool(BaseTool):
    name: str = "copy_path"
    description: str = "Copies a file or a whole folder to another place on this computer."
    args_schema: Type[BaseModel] = CopyPathInput

    def _run(self, source: str, destination: str) -> str:
        async def _async_run():
            blocked = _check_enabled()
            if blocked:
                return blocked
            src, dst = _resolve_user_path(source), _resolve_user_path(destination)
            if not src.exists():
                return f"Not found: {src}. Use find_files or list_folder to get the exact path."
            if dst.is_dir():
                # Into an existing folder, as the description and move_path say.
                # A folder used to be merged into the destination's own contents.
                dst = dst / src.name
            if src.is_dir():
                real_src, real_dst = src.resolve(), dst.resolve()
                if real_dst == real_src or real_src in real_dst.parents:
                    return f"Could not copy {src} to {dst}: a folder cannot be copied into itself."
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.is_dir():
                    shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
                else:
                    shutil.copy2(str(src), str(dst))
            except Exception as e:
                return f"Could not copy {src} to {dst}: {e}"
            logger.info("local_path_copied", source=str(src), destination=str(dst))
            return f"Copied {src} to {dst}"

        return _run_in_worker(_async_run)


# Never delete these, whatever is asked.
_PROTECTED_PATHS = {Path.home(), Path(os.environ.get("SystemRoot", r"C:\Windows")), Path(r"C:\Program Files"), Path(r"C:\Program Files (x86)")}


def _protected_paths() -> set[Path]:
    """Folders delete_path refuses: system folders, the home folder, and the
    user's own top-level folders (Desktop, Documents, Downloads, OneDrive ...).

    Found 2026-09-16: "delete desktop" resolved to the user's whole OneDrive
    Desktop and would have sent all of it to the Recycle Bin. Only the home
    folder itself was protected, not the folders the user keeps their work in.
    Files and folders INSIDE them can still be deleted.
    """
    paths = set(_PROTECTED_PATHS) | {Path.home().parent}
    for key, name in _DEFAULT_FOLDER_NAMES.items():
        paths.add(Path.home() / name)
        paths.add(Path(_known_folder(key)))
    for variable in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        if os.environ.get(variable):
            paths.add(Path(os.environ[variable]))
    resolved: set[Path] = set()
    for path in paths:
        try:
            resolved.add(path.resolve())
        except OSError:
            continue
    return resolved


def _recycle(path: Path) -> str | None:
    """Send a path to the Recycle Bin. Returns None on success, else the problem."""
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [
            ("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", wintypes.LPCWSTR),
            ("pTo", wintypes.LPCWSTR), ("fFlags", ctypes.c_uint16), ("fAnyOperationsAborted", wintypes.BOOL),
            ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR),
        ]

    FO_DELETE, FOF_SILENT, FOF_NOCONFIRMATION, FOF_ALLOWUNDO, FOF_NOERRORUI = 3, 0x0004, 0x0010, 0x0040, 0x0400
    operation = SHFILEOPSTRUCTW(
        None, FO_DELETE, f"{path}\0\0", None,
        FOF_SILENT | FOF_NOCONFIRMATION | FOF_ALLOWUNDO | FOF_NOERRORUI, False, None, None,
    )
    code = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(operation))
    if code != 0:
        return f"Windows returned error code {code}"
    if operation.fAnyOperationsAborted:
        return "Windows cancelled the operation, so nothing was deleted"
    return None


class DeletePathInput(BaseModel):
    path: str = Field(..., description="File or folder to delete. It goes to the Recycle Bin, so it can be restored.")


class DeletePathTool(BaseTool):
    name: str = "delete_path"
    description: str = (
        "Deletes a file or folder by sending it to the Recycle Bin, so the user can restore it. "
        "System folders and the user's home folder itself are refused."
    )
    args_schema: Type[BaseModel] = DeletePathInput

    def _run(self, path: str) -> str:
        async def _async_run():
            blocked = _check_enabled()
            if blocked:
                return blocked
            target = _resolve_user_path(path)
            if not target.exists():
                return f"Not found: {target}. Nothing was deleted."
            resolved = target.resolve()
            if resolved in _protected_paths() or resolved.parent == resolved:
                return f"Refused: {resolved} is a system or home folder and will not be deleted."
            problem = _recycle(resolved)
            if problem:
                return f"Could not delete {resolved}: {problem}"
            logger.info("local_path_recycled", path=str(resolved))
            return f"Moved to the Recycle Bin: {resolved} (restore it from there if this was wrong)."

        return _run_in_worker(_async_run)


# =============================================================================
# 11. Search inside files (content search)
# =============================================================================

_SEARCHABLE_SUFFIXES = {
    ".txt", ".md", ".py", ".js", ".ts", ".json", ".csv", ".log", ".html", ".css", ".yml", ".yaml",
    ".ini", ".cfg", ".ps1", ".bat", ".sql", ".xml", ".java", ".c", ".cpp", ".cs", ".go", ".rs", ".sh", ".env",
}
_SEARCH_TIME_BUDGET_S = 20.0
_SEARCH_MAX_FILE_BYTES = 5 * 1024 * 1024


class SearchInFilesInput(BaseModel):
    text: str = Field(..., description="Text to look for inside files (case-insensitive).")
    location: str = Field(default="desktop", description="Where to search: a folder path, or 'desktop', 'downloads', 'documents', 'home'.")
    file_pattern: str = Field(default="*", description="Only files matching this, e.g. '*.py' or '*.txt'.")
    max_results: int = Field(default=20, description="Maximum matching lines to return.")


class SearchInFilesTool(BaseTool):
    name: str = "search_in_files"
    description: str = (
        "Searches INSIDE files on this computer for a piece of text and returns the matching lines with their "
        "file and line number. Use it to find where something is written (code, notes, logs, config). "
        "To find files by NAME instead, use find_files."
    )
    args_schema: Type[BaseModel] = SearchInFilesInput

    def _run(self, text: str, location: str = "desktop", file_pattern: str = "*", max_results: int = 20) -> str:
        async def _async_run():
            blocked = _check_enabled()
            if blocked:
                return blocked
            needle = (text or "").strip()
            if not needle:
                return "No search text given."
            folder = _resolve_location(location)
            if not folder.exists():
                return f"Folder not found: {folder}"
            deadline = time.monotonic() + _SEARCH_TIME_BUDGET_S
            matches: list[str] = []
            scanned = 0
            name_pattern = (file_pattern or "*").lower()
            lowered_needle = needle.lower()
            for file in _walk_files(folder, deadline):
                if len(matches) >= max_results:
                    break
                try:
                    if file.suffix.lower() not in _SEARCHABLE_SUFFIXES or not fnmatch.fnmatch(file.name.lower(), name_pattern):
                        continue
                    if file.stat().st_size > _SEARCH_MAX_FILE_BYTES:
                        continue
                    scanned += 1
                    with open(file, "r", encoding="utf-8", errors="replace") as handle:
                        content = handle.read()
                    # One check on the whole file; lines are only walked for a
                    # file that contains the text (most files do not).
                    if lowered_needle not in content.lower():
                        continue
                    for number, line in enumerate(content.splitlines(), start=1):
                        if lowered_needle in line.lower():
                            matches.append(f"{file}:{number}: {line.strip()[:160]}")
                            if len(matches) >= max_results:
                                break
                except OSError:
                    continue
            # Stated first, where it cannot be overlooked. Found 2026-09-16: with
            # the note at the end, the agent reported "no matches in the 2,106
            # files examined" as if the whole Desktop had been searched.
            partial = ""
            if time.monotonic() > deadline and len(matches) < max_results:
                partial = (
                    f"PARTIAL RESULT — the search stopped at its {int(_SEARCH_TIME_BUDGET_S)} s limit before "
                    f"covering all of {folder}. Tell the user the search was incomplete, or search a narrower folder.\n"
                )
            if not matches:
                return (
                    f"{partial}No files under {folder} matching '{file_pattern}' contain '{needle}' "
                    f"({scanned} files searched)."
                )
            logger.info("local_content_search", text=needle[:40], folder=str(folder), matches=len(matches))
            return (
                f"{partial}Found '{needle}' in {len(matches)} place(s) ({scanned} files searched):\n"
                + "\n".join(matches)
            )

        return _run_in_worker(_async_run)


# =============================================================================
# 12. Clipboard
# =============================================================================

class ClipboardInput(BaseModel):
    text: str = Field(default="", description="Text to copy to the clipboard. Leave empty to read the clipboard instead.")


class ClipboardTool(BaseTool):
    name: str = "clipboard"
    description: str = "Reads the Windows clipboard, or copies text into it (leave `text` empty to read)."
    args_schema: Type[BaseModel] = ClipboardInput

    def _run(self, text: str = "") -> str:
        async def _async_run():
            blocked = _check_enabled()
            if blocked:
                return blocked
            if not text:
                proc = _run_powershell("Get-Clipboard -Raw", timeout_seconds=20)
                content = (proc.stdout or "").strip()
                return f"Clipboard contains:\n{content[:4000]}" if content else "The clipboard is empty."
            # The text goes through a file, so quotes and newlines in it cannot break the script.
            payload = _AGENT_TEMP / f"clip_{os.getpid()}_{int(time.time() * 1000)}.txt"
            payload.parent.mkdir(parents=True, exist_ok=True)
            payload.write_text(text, encoding="utf-8")
            try:
                # -Encoding UTF8: Windows PowerShell otherwise reads the file as
                # ANSI, and Hindi or accented text reached the clipboard as
                # mojibake (verified 2026-09-16: "नमस्ते" became "à¤¨à¤®...").
                literal = str(payload).replace("'", "''")
                proc = _run_powershell(
                    f"Get-Content -Raw -Encoding UTF8 -LiteralPath '{literal}' | Set-Clipboard", timeout_seconds=20
                )
            finally:
                try:
                    payload.unlink()
                except OSError:
                    pass
            if proc.returncode != 0:
                return f"Could not copy to the clipboard: {(proc.stderr or '').strip()[:200]}"
            logger.info("local_clipboard_set", chars=len(text))
            return f"Copied {len(text)} characters to the clipboard."

        return _run_in_worker(_async_run)


from app.tools.desktop import DESKTOP_TOOLS  # noqa: E402  (windows GUI: list/focus/see/click)
from app.tools.apps import InstallAppTool  # noqa: E402  (winget: Microsoft Store and winget installs)
from app.tools.mail import EMAIL_TOOLS  # noqa: E402  (the user's mailbox: read, draft, send)

ALL_LOCAL_TOOLS = [
    *[T for T in DESKTOP_TOOLS],
    ListRecentFilesTool,
    FindFilesTool,
    SearchInFilesTool,
    OpenPathTool,
    ReadFileTool,
    WriteFileTool,
    CreateFolderTool,
    MovePathTool,
    CopyPathTool,
    DeletePathTool,
    ListFolderTool,
    ClipboardTool,
    OpenAppTool,
    InstallAppTool,
    RunCommandTool,
    SendKeysTool,
    *EMAIL_TOOLS,
]
