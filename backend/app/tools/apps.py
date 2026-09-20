"""
Installing apps with winget: the Microsoft Store and the winget catalogue.

Added 2026-09-17: asked to "download X from the Microsoft Store", the agent had
no tool for it and gave up, or tried to click through the Store app. winget
ships with Windows 11 (App Installer) and installs Store apps (source msstore)
and ordinary programs (source winget) without opening any window.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Type

import structlog
from pydantic import BaseModel, Field

from app.tools.base import BaseTool
from app.tools.runtime import on_main_loop, run_in_worker

logger = structlog.get_logger(__name__)

# Installs can be slow; the tool itself is stopped after 300 s.
INSTALL_WAIT_SECONDS = 270
_AGREE = ["--accept-source-agreements", "--disable-interactivity"]
_SOURCES = {"": "", "any": "", "store": "msstore", "microsoft store": "msstore", "msstore": "msstore", "winget": "winget"}
_SOURCE_LABEL = {"msstore": "from the Microsoft Store", "winget": "from the winget catalogue"}
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _winget(args: list[str], timeout: float = 90) -> tuple[int, str]:
    proc = subprocess.run(
        ["winget", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, creationflags=_NO_WINDOW,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def parse_winget_table(output: str) -> list[dict[str, str]]:
    """Rows of a winget search/list table, keyed by lower-case column name."""
    lines = [line.rstrip() for line in re.split(r"\r\n|\r|\n", output or "")]
    header_at = next(
        (
            i for i in range(len(lines) - 1)
            if re.match(r"^\s*Name\s+Id\s", lines[i]) and lines[i + 1].strip() and set(lines[i + 1].strip()) == {"-"}
        ),
        None,
    )
    if header_at is None:
        return []
    header = lines[header_at]
    columns: list[tuple[str, int]] = []
    pos = 0
    for name in header.split():
        pos = header.index(name, pos)
        columns.append((name.lower(), pos))
        pos += len(name)
    rows = []
    for line in lines[header_at + 2:]:
        if not line.strip() or len(line) <= columns[1][1]:
            continue
        row = {}
        for k, (name, start) in enumerate(columns):
            end = columns[k + 1][1] if k + 1 < len(columns) else None
            row[name] = line[start:end].strip()
        if row.get("id"):
            rows.append(row)
    return rows


def pick_package(rows: list[dict[str, str]], name: str) -> dict[str, str] | None:
    """The row whose name matches `name` best: exact, then starts with, then winget's own order."""
    if not rows:
        return None
    wanted = name.strip().casefold()
    exact = [r for r in rows if r.get("name", "").casefold() == wanted]
    if exact:
        return exact[0]
    starts = [r for r in rows if r.get("name", "").casefold().startswith(wanted)]
    if starts:
        return min(starts, key=lambda r: len(r.get("name", "")))
    return rows[0]


def _find_package(name: str, source: str) -> dict[str, str] | None:
    source_args = ["--source", source] if source else []
    for query in (["--name", name], [name]):
        code, out = _winget(["search", *query, *source_args, *_AGREE])
        rows = parse_winget_table(out)
        for row in rows:
            row.setdefault("source", source)
        best = pick_package(rows, name)
        if best:
            return best
    return None


def _is_installed(package_id: str) -> bool:
    code, out = _winget(["list", "--id", package_id, "--exact", *_AGREE])
    return any(row.get("id", "").casefold() == package_id.casefold() for row in parse_winget_table(out))


def _install(package_id: str, source: str) -> tuple[int | None, str]:
    """Run winget install. Returns (exit code, output); exit code None when it is still running."""
    args = ["install", "--id", package_id, "--exact", "--accept-package-agreements", *_AGREE]
    if source:
        args += ["--source", source]
    if source == "winget":
        args.append("--silent")
    # Output goes to a file, not a pipe: a still-running install must not block on a full pipe.
    log_path = Path(tempfile.gettempdir()) / "self_improving_agent" / f"winget_{re.sub(r'[^A-Za-z0-9.]', '_', package_id)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(["winget", *args], stdout=log, stderr=subprocess.STDOUT, creationflags=_NO_WINDOW)
        try:
            code = proc.wait(timeout=INSTALL_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            code = None
    return code, log_path.read_text(encoding="utf-8", errors="replace")


class InstallAppInput(BaseModel):
    name: str = Field(..., description="The app to install, as people call it (e.g. 'WhatsApp', 'Spotify', 'VLC', 'Zoom').")
    source: str = Field(default="any", description="'store' for the Microsoft Store only, 'winget' for the winget catalogue only, or 'any'.")


class InstallAppTool(BaseTool):
    name: str = "install_app"
    description: str = (
        "Installs an app on this computer from the Microsoft Store or the winget catalogue (winget), without opening "
        "the Store: use it for 'download/install X from the Microsoft Store' and 'install X'. It finds the app, "
        "checks whether it is already installed, asks the user when a window is open, installs it and confirms. "
        "Then open it with open_app. For a file from a website (PDF, image, installer link), use download_file."
    )
    args_schema: Type[BaseModel] = InstallAppInput

    def _run(self, name: str, source: str = "any") -> str:
        # Read in this worker thread: the engine's context variable is copied here.
        from app.utils.confirmation import get_confirmer
        confirmer = get_confirmer()

        async def _async_run():
            from app.tools.local import _check_enabled

            blocked = _check_enabled()
            if blocked:
                return blocked
            wanted = (name or "").strip()
            if not wanted:
                return "No app name given."
            src = _SOURCES.get((source or "").strip().lower(), "")
            try:
                package = _find_package(wanted, src)
            except FileNotFoundError:
                return (
                    "Could not install: winget (App Installer) is not available on this computer. Open the Store "
                    "with open_app('Microsoft Store') and search there instead."
                )
            except subprocess.TimeoutExpired:
                return "Could not install: searching with winget took too long. Try again in a moment."
            if package is None:
                return f"Could not find an app called '{wanted}' in the Microsoft Store or the winget catalogue."

            package_id = package["id"]
            package_name = package.get("name") or wanted
            package_source = package.get("source") or src
            label = _SOURCE_LABEL.get(package_source, "with winget")
            if _is_installed(package_id):
                return f"Already installed: {package_name} ({package_id}). Open it with open_app('{package_name}')."

            if confirmer is not None:
                approved = await on_main_loop(confirmer.request(
                    f"Install '{package_name}' {label} on this computer",
                    {"id": package_id, "source": package_source, "version": package.get("version", "")},
                ))
                if not approved:
                    return f"The user declined, so '{package_name}' was not installed. Do not retry; report it."

            logger.info("app_install_started", app=package_name, id=package_id, source=package_source)
            code, output = _install(package_id, package_source)
            tail = " ".join(output.split())[-300:]
            if code is None:
                return (
                    f"Still installing {package_name}: winget is still working after {INSTALL_WAIT_SECONDS // 60} "
                    "minutes and continues in the background. Call install_app again later; it reports "
                    "'Already installed' once it has finished."
                )
            if code == 0 or _is_installed(package_id):
                logger.info("app_installed", app=package_name, id=package_id)
                return f"Installed: {package_name} ({package_id}) {label}. Open it with open_app('{package_name}')."
            logger.warning("app_install_failed", app=package_name, id=package_id, code=code, output=tail)
            return f"Could not install {package_name} ({package_id}): winget exit code {code}. {tail}"

        return run_in_worker(_async_run)
