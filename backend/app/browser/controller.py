"""
Playwright CDP connection manager.

Connection strategy (in order):
  1. Attach to a Chrome already listening on 127.0.0.1:9222 (normally the
     agent's own window from an earlier run).
  2. Launch the installed Chrome on the agent's OWN persistent profile,
     ~/.self_improving_agent/chrome_profile, with the debug port open.
  3. If that fails: a Playwright-managed window on the same profile, then
     Playwright's bundled Chromium.

The agent never launches, copies or reads the user's own Chrome profiles.
Found 2026-09-17: the agent used to run on a copy of the user's last-used
profile ("Person 1"), including its Google sign-in, so one Google session was
in use in two browsers at once, and the user's own Person 1 kept being signed
out. After scripts/migrate_chrome_profile.py it could also start Chrome on the
user's real profile folder under a second path; Chrome allows one browser per
profile folder, so the user's own profiles would not open while that window
was up. The user signs in once, in the agent's window, to the sites the agent
needs; those logins stay in the agent's profile.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import weakref
from pathlib import Path
from typing import Any, Optional

import structlog
from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
    Playwright,
)

logger = structlog.get_logger(__name__)

# Seconds to wait for Chrome to start up and open the CDP port
_CHROME_STARTUP_WAIT = 4.0
# Seconds to wait between polling attempts for the port
_CHROME_POLL_INTERVAL = 0.3


def _find_chrome_exe() -> Optional[str]:
    """
    Locate the Google Chrome executable on Windows.

    Checks the standard installation paths.  Returns None if not found.
    """
    import os
    import platform

    system = platform.system()

    if system == "Windows":
        candidates = [
            os.path.expandvars(r"%PROGRAMFILES%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%PROGRAMFILES(X86)%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        ]
    elif system == "Darwin":
        candidates = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    else:  # Linux
        candidates = [
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/chromium-browser",
        ]

    for path in candidates:
        if Path(path).exists():
            return path

    return None


# The agent's own Chrome profile: persistent, so sites signed into in the
# agent's window stay signed in across restarts.
AGENT_PROFILE_DIR = Path.home() / ".self_improving_agent" / "chrome_profile"
AGENT_PROFILE_NAME = "Self-Improving Agent"


def _extension_args() -> list[str]:
    """
    Arguments that load the project's extension into the agent's Chrome.

    Google Chrome 137 and later ignore --load-extension (checked on Chrome 152,
    2026-09-17); Chromium still honours it. In Chrome the side panel is added
    once via chrome://extensions → Load unpacked. The agent works without it.
    """
    ext_dir = Path(__file__).resolve().parent.parent.parent.parent / "extension"
    if (ext_dir / "manifest.json").exists():
        return [f"--disable-extensions-except={ext_dir}", f"--load-extension={ext_dir}"]
    return []


def _agent_chrome_command(chrome_exe: str, port: int) -> list[str]:
    """The command that starts the agent's own Chrome window with the debug port."""
    return [
        chrome_exe,
        f"--remote-debugging-port={port}",
        "--remote-allow-origins=*",
        f"--user-data-dir={AGENT_PROFILE_DIR}",
        "--profile-directory=Default",
        # Title bar and taskbar say whose window this is.
        f"--window-name={AGENT_PROFILE_NAME}",
        "--no-first-run",
        "--no-default-browser-check",
        "--hide-crash-restore-bubble",
        "--start-maximized",
        *_extension_args(),
    ]


def _prepare_agent_profile(user_data_dir: Path) -> None:
    """
    Create the agent's profile folder and give the profile a name of its own.

    A new Chrome profile is called "Person 1" or "Your Chrome", just like the
    user's own profiles, which makes the agent's window easy to mistake for
    theirs. A name someone chose in Chrome is left alone. Called only right
    before Chrome is started on this folder, so Chrome is not running on it.
    """
    import json

    profile = user_data_dir / "Default"
    profile.mkdir(parents=True, exist_ok=True)
    targets = (
        (user_data_dir / "Local State", ("profile", "info_cache", "Default"), "is_using_default_name"),
        (profile / "Preferences", ("profile",), "using_default_name"),
    )
    for path, keys, default_flag in targets:
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            node = data
            for key in keys:
                node = node.setdefault(key, {})
            if node.get("name") == AGENT_PROFILE_NAME or node.get(default_flag) is False:
                continue
            node["name"] = AGENT_PROFILE_NAME
            node[default_flag] = False
            path.write_text(json.dumps(data), encoding="utf-8")
        except Exception as e:  # a missing label must never stop the launch
            logger.info("agent_profile_name_skipped", file=path.name, error=str(e)[:80])


_DOWNLOADS_KEPT = 20
_UNSAFE_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _download_target(suggested: str, reserved: set[str]) -> Path:
    """A free path in the user's real Downloads folder for a downloaded file."""
    from app.tools.local import _known_folder

    folder = _known_folder("downloads")
    folder.mkdir(parents=True, exist_ok=True)
    name = _UNSAFE_NAME_CHARS.sub("_", Path(suggested or "").name).strip(" .") or "download"
    target = folder / name
    n = 1
    while target.exists() or str(target) in reserved:
        target = folder / f"{Path(name).stem} ({n}){Path(name).suffix}"
        n += 1
    return target


def _agent_chrome_on_port(port: int) -> bool:
    """True when the Chrome listening on `port` runs on the agent's own profile."""
    script = (
        f"$c = Get-NetTCPConnection -LocalPort {int(port)} -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; "
        "if ($c) { (Get-CimInstance Win32_Process -Filter \"ProcessId=$($c.OwningProcess)\").CommandLine }"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        return False
    return str(AGENT_PROFILE_DIR).lower() in (out.stdout or "").lower()


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """True when something accepts TCP connections on host:port."""
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


async def _wait_for_cdp_port(host: str, port: int, timeout: float = 6.0) -> bool:
    """
    Poll until the CDP HTTP endpoint at host:port responds, or until timeout.

    Returns True if the port becomes available within timeout, False otherwise.
    """
    import httpx
    deadline = asyncio.get_event_loop().time() + timeout
    url = f"http://{host}:{port}/json/version"

    while asyncio.get_event_loop().time() < deadline:
        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return True
        except Exception:
            pass
        await asyncio.sleep(_CHROME_POLL_INTERVAL)

    return False


class BrowserController:
    """
    Manages the Playwright connection to a Chrome browser.

    Lifecycle::

        controller = BrowserController()
        await controller.connect("http://127.0.0.1:9222")
        page = await controller.get_active_page()
        # ... use page ...
        await controller.disconnect()
    """

    def __init__(self) -> None:
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._connected: bool = False
        # Dedicated tab the agent works in (so we never hijack the user's tab)
        self._agent_page: Optional[Page] = None
        # The agent's Chrome, when this controller started it (closed on disconnect)
        self._chrome_proc: Optional[subprocess.Popen] = None  # type: ignore[type-arg]
        # The agent's own Chrome, left open by an earlier run and attached to now:
        # closed on disconnect too, so stopping the agent leaves no window behind.
        self._close_on_disconnect = False
        # Files downloaded in the agent's window, newest last (see _watch_downloads).
        self.downloads: list[dict[str, Any]] = []
        self.download_seq = 0
        self._download_tasks: set[asyncio.Task] = set()
        self._watched: "weakref.WeakSet[Any]" = weakref.WeakSet()

    @property
    def is_connected(self) -> bool:
        return self._connected and (self._browser is not None or self._context is not None)

    async def connect(self, cdp_endpoint: str, allow_launch: bool = True) -> None:
        """
        Connect to Chrome via CDP, using the best available strategy.

        Args:
            cdp_endpoint: CDP base URL, e.g. "http://127.0.0.1:9222".
            allow_launch: When False (backend startup default), only ATTACH to
                an already-running Chrome on the debug port — never launch a
                browser window. The launch strategies run later, lazily, when
                the user actually opens the extension / Local Agent UI
                (see ensure_browser_connection in app.main).
        """
        if self._connected:
            logger.warning("browser_already_connected")
            return

        # Normalize localhost → 127.0.0.1 (avoids IPv6 ::1 on Windows)
        endpoint = cdp_endpoint.replace("localhost", "127.0.0.1")
        host = endpoint.split("//")[-1].split(":")[0]
        port = int(endpoint.split(":")[-1])

        # A reconnect used to start a second Playwright driver and leave the
        # first one running: one leaked driver process per reconnect.
        await self._cleanup_playwright()
        self._playwright = await async_playwright().start()

        # ── Strategy 1: Attach to an already-running Chrome on the CDP port ───
        # Retry a few times only while something is listening: Chrome may still
        # be initialising after scripts/launch_linked_chrome.py. A refused port stays
        # refused — retrying it cost 6-8 s before every browser task (found
        # 2026-09-15). At backend startup one retry is kept, in case Chrome is
        # being launched at the same moment.
        port_open = await asyncio.to_thread(_port_open, host, port)
        max_attach_attempts = 4 if port_open else (1 if allow_launch else 2)
        for attempt in range(1, max_attach_attempts + 1):
            logger.info(
                "browser_strategy1_cdp_attach",
                endpoint=endpoint,
                attempt=f"{attempt}/{max_attach_attempts}",
            )
            try:
                self._browser = await self._playwright.chromium.connect_over_cdp(
                    endpoint, timeout=3000
                )
                self._connected = True
                self._close_on_disconnect = await asyncio.to_thread(_agent_chrome_on_port, port)
                # Work in a NEW tab — never hijack the user's current one.
                await self.open_agent_tab()
                ctx_count = len(self._browser.contexts)
                page_count = sum(len(c.pages) for c in self._browser.contexts)
                logger.info(
                    "browser_connected_to_existing_chrome",
                    mode="cdp_attach",
                    contexts=ctx_count,
                    pages=page_count,
                    note="Attached to the Chrome already listening on the debug port.",
                )
                return
            except Exception as e:
                logger.info(
                    "browser_strategy1_failed",
                    attempt=attempt,
                    reason=str(e)[:80],
                )
                if attempt < max_attach_attempts:
                    await asyncio.sleep(2)

        if not allow_launch:
            logger.info(
                "browser_deferred_attach_only",
                note="No Chrome with debug port running. NOT launching a browser "
                     "at startup — it will start when you open the extension "
                     "or the Local Agent.",
            )
            await self._cleanup_playwright()
            return

        # ── Strategy 2: Launch the agent's own Chrome window ─────────────────
        # On the agent's own profile folder, never one of the user's: their
        # Chrome and all of its profiles keep working side by side, and no
        # Google sign-in is shared between the two browsers (module docstring).
        chrome_exe = _find_chrome_exe()
        if chrome_exe:
            if self._chrome_proc is not None:
                try:
                    self._chrome_proc.terminate()
                except Exception:
                    pass
                self._chrome_proc = None
            try:
                await asyncio.to_thread(_prepare_agent_profile, AGENT_PROFILE_DIR)
                self._chrome_proc = subprocess.Popen(
                    _agent_chrome_command(chrome_exe, port),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logger.info(
                    "agent_chrome_launched",
                    pid=self._chrome_proc.pid,
                    user_data=str(AGENT_PROFILE_DIR),
                    note="The agent's own profile; your Chrome profiles are not touched.",
                )
                if await _wait_for_cdp_port(host, port, timeout=15.0):
                    self._browser = await self._playwright.chromium.connect_over_cdp(
                        endpoint, timeout=5000
                    )
                    self._connected = True
                    logger.info(
                        "browser_connected_to_chrome",
                        mode="agent_profile_cdp",
                        user_data=str(AGENT_PROFILE_DIR),
                    )
                    return
                logger.warning("chrome_cdp_port_did_not_open", port=port)
            except Exception as e:
                logger.warning("browser_strategy2_failed", error=str(e)[:120])
            if self._chrome_proc:
                try:
                    self._chrome_proc.terminate()
                except Exception:
                    pass
                self._chrome_proc = None
        else:
            logger.info("browser_strategy2_skipped", chrome_found=False)

        # ── Strategy 3: Playwright-managed window on the agent's profile ─────
        logger.warning(
            "browser_fallback_to_playwright_window",
            notice="Chrome did not open its debug port; using a Playwright-managed "
                   "window on the agent's profile instead.",
        )
        try:
            await asyncio.to_thread(_prepare_agent_profile, AGENT_PROFILE_DIR)
            self._context = await self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(AGENT_PROFILE_DIR),
                headless=False,
                channel="chrome",
                args=[
                    "--start-maximized",
                    f"--remote-debugging-port={port}",
                    f"--window-name={AGENT_PROFILE_NAME}",
                    "--hide-crash-restore-bubble",
                    "--no-first-run",
                    *_extension_args(),
                ],
            )
            self._browser = self._context.browser or self._context  # type: ignore[assignment]
            self._connected = True
            logger.info(
                "browser_launched_standalone_profile",
                profile_path=str(AGENT_PROFILE_DIR),
                mode="standalone_persistent_context",
            )
            return

        except Exception:
            # Final fallback: bundled Chromium (no channel="chrome")
            try:
                profile_dir = Path.home() / ".self_improving_agent" / "chromium_profile"
                profile_dir.mkdir(parents=True, exist_ok=True)

                self._context = await self._playwright.chromium.launch_persistent_context(
                    user_data_dir=str(profile_dir),
                    headless=False,
                    args=[
                        "--start-maximized",
                        f"--remote-debugging-port={port}",
                    ],
                )
                self._browser = self._context.browser or self._context  # type: ignore[assignment]
                self._connected = True
                logger.info("browser_launched_bundled_chromium", profile_path=str(profile_dir))
                return

            except Exception as final_err:
                logger.error("browser_all_strategies_failed", error=str(final_err))
                await self._cleanup_playwright()
                raise ConnectionError(
                    f"Could not connect to or launch any browser: {final_err}"
                ) from final_err

    async def _ensure_alive(self) -> None:
        """
        Check the current connection is still responsive; reconnect if not.

        Logs the reason for reconnection so browser context death is visible
        in the log stream (fixes Fix #6 — silent context death).
        """
        need_reconnect = False
        reason = ""

        if not self._connected:
            need_reconnect = True
            reason = "not_connected"
        elif self._context is not None:
            try:
                _ = self._context.pages  # Will raise if context closed
            except Exception as e:
                need_reconnect = True
                reason = f"context_closed: {e}"
        elif self._browser is not None:
            try:
                if hasattr(self._browser, "is_connected") and not self._browser.is_connected():
                    need_reconnect = True
                    reason = "browser_disconnected"
            except Exception as e:
                need_reconnect = True
                reason = f"browser_check_failed: {e}"
        else:
            need_reconnect = True
            reason = "no_browser_or_context"

        if need_reconnect:
            logger.warning("browser_reconnecting", reason=reason)
            self._connected = False
            self._browser = None
            self._context = None
            from app.config import get_settings
            await self.connect(get_settings().cdp_endpoint)

    def _watch_downloads(self) -> None:
        """
        Save every file downloaded in the agent's window to the user's Downloads.

        Found 2026-09-17: with Playwright attached, Chrome hands each download
        to Playwright, which keeps it in a temporary folder deleted when it
        disconnects, so a file downloaded with a click never reached Downloads.
        """
        contexts = []
        if self._context is not None:
            contexts.append(self._context)
        if self._browser is not None and hasattr(self._browser, "contexts"):
            contexts.extend(self._browser.contexts)
        for context in contexts:
            if context in self._watched:
                continue
            self._watched.add(context)
            context.on("page", self._watch_page)
            for page in context.pages:
                self._watch_page(page)

    def _watch_page(self, page: Page) -> None:
        if page in self._watched:
            return
        self._watched.add(page)
        page.on("download", self._on_download)

    def _on_download(self, download: Any) -> None:
        reserved = {d["path"] for d in self.downloads if d["status"] == "downloading"}
        target = _download_target(download.suggested_filename, reserved)
        self.download_seq += 1
        record = {
            "seq": self.download_seq,
            "name": download.suggested_filename,
            "url": download.url,
            "path": str(target),
            "status": "downloading",
            "size": 0,
            "error": "",
        }
        self.downloads.append(record)
        del self.downloads[:-_DOWNLOADS_KEPT]
        logger.info("download_started", name=record["name"], url=record["url"][:150])
        task = asyncio.get_running_loop().create_task(self._save_download(download, target, record))
        self._download_tasks.add(task)
        task.add_done_callback(self._download_tasks.discard)

    async def _save_download(self, download: Any, target: Path, record: dict[str, Any]) -> None:
        try:
            await download.save_as(target)  # waits until the download has finished
            record["size"] = target.stat().st_size
            record["status"] = "saved"
            logger.info("download_saved", path=str(target), bytes=record["size"])
        except Exception as e:
            record["status"] = "failed"
            record["error"] = str(e)[:200]
            logger.warning("download_failed", name=record["name"], error=record["error"])

    async def open_agent_tab(self) -> Page:
        """
        Open a dedicated tab for the agent to work in and remember it.

        When we attach to the user's own Chrome, this guarantees the agent
        navigates a NEW tab instead of hijacking the tab the user is reading.
        """
        if self._agent_page is not None and not self._agent_page.is_closed():
            return self._agent_page

        # Found 2026-09-17: after the user closed the agent's Chrome window, the
        # connection still counted as open, there was no context to open a tab
        # in, and open_app('chrome') reported success anyway.
        await self._ensure_alive()
        if self._agent_page is not None and not self._agent_page.is_closed():
            return self._agent_page  # the reconnect opened it already

        context = None
        if self._browser is not None and self._browser.contexts:
            context = self._browser.contexts[0]
        elif self._context is not None:
            context = self._context
        if context is None:
            raise RuntimeError("No browser context available to open an agent tab.")

        self._agent_page = await context.new_page()
        self._watch_downloads()
        try:
            await self._agent_page.bring_to_front()
        except Exception:
            pass
        logger.info("agent_tab_opened")
        return self._agent_page

    async def follow_new_tab(self, pages_before: int) -> Page | None:
        """
        Make a tab that a click just opened the agent's working tab.

        Found 2026-09-17: shops such as Flipkart open a product in a new tab;
        the agent kept working in the old tab, so "add to cart" never found
        the product page's button.
        """
        pages = await self.get_all_pages()
        if len(pages) <= pages_before:
            return None
        newest = pages[-1]
        self._agent_page = newest
        self._watch_page(newest)
        try:
            await newest.bring_to_front()
            await newest.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            pass
        logger.info("agent_tab_followed", url=newest.url[:150])
        return newest

    async def get_active_page(self) -> Page:
        """
        Return the currently visible, non-closed page in the browser.

        Prefers the dedicated agent tab (see open_agent_tab), then user-facing
        tabs over extension/devtools pages.
        Auto-creates a new page if none exist.
        Retries once after a reconnect on failure.
        """
        for attempt in range(2):
            try:
                await self._ensure_alive()

                self._watch_downloads()
                # Case 0: the agent's own tab (never hijack the user's tab)
                if self._agent_page is not None and not self._agent_page.is_closed():
                    return self._agent_page
                if self._agent_page is not None and self._agent_page.is_closed():
                    self._agent_page = None

                # Case 1: Persistent context (Strategy 3)
                context = self._context
                if context is not None:
                    pages = context.pages
                    if not pages:
                        logger.info("creating_new_page_in_persistent_context")
                        return await context.new_page()
                    # Prefer non-special pages
                    for p in reversed(pages):
                        if not p.is_closed() and not p.url.startswith("chrome-extension://") and not p.url.startswith("devtools://"):
                            return p
                    for p in reversed(pages):
                        if not p.is_closed():
                            return p
                    return await context.new_page()

                # Case 2: Standard Browser (Strategy 1 & 2 — CDP attach)
                if self._browser is not None and hasattr(self._browser, "contexts"):
                    contexts = self._browser.contexts
                    if not contexts:
                        logger.info("creating_new_context_and_page")
                        ctx = await self._browser.new_context()
                        return await ctx.new_page()
                    for ctx in contexts:
                        for p in reversed(ctx.pages):
                            if not p.is_closed() and not p.url.startswith("chrome-extension://") and not p.url.startswith("devtools://"):
                                return p
                        for p in reversed(ctx.pages):
                            if not p.is_closed():
                                return p
                    return await contexts[0].new_page()

                raise RuntimeError("Browser not initialized properly")

            except Exception as e:
                logger.warning(
                    "get_active_page_recovery_attempt",
                    attempt=attempt,
                    error=str(e),
                )
                self._connected = False
                self._browser = None
                self._context = None
                if attempt == 1:
                    raise

    async def get_all_pages(self) -> list[Page]:
        """Return all open, non-closed pages across all contexts."""
        await self._ensure_alive()
        context = self._context
        if context is not None:
            return [p for p in context.pages if not p.is_closed()]

        if self._browser is not None and hasattr(self._browser, "contexts"):
            all_pages: list[Page] = []
            for ctx in self._browser.contexts:
                all_pages.extend(p for p in ctx.pages if not p.is_closed())
            return all_pages

        return []

    async def disconnect(self) -> None:
        """
        Disconnect from the browser.

        If we launched the agent's Chrome ourselves (Strategy 2), shut it down
        GRACEFULLY so cookies/sessions flush to disk — logins made in the
        agent's window survive restarts. A Chrome we only attached to
        (Strategy 1) is left running. A Playwright window (Strategy 3) is closed.
        """
        if not self._connected:
            return

        logger.info("browser_disconnecting")

        # 1. Close the agent's dedicated tab (keeps the rest of the window intact
        #    until we're ready to close the agent's whole Chrome process)
        if self._agent_page is not None:
            try:
                if not self._agent_page.is_closed():
                    await self._agent_page.close()
            except Exception:
                pass
            self._agent_page = None

        # 2. Close standalone Playwright context (Strategy 3 only)
        if self._context is not None:
            try:
                await self._context.close()
            except Exception as e:
                logger.warning("context_close_error", error=str(e))
            self._context = None

        # 3. The agent's own Chrome from an earlier run: close it normally
        #    (Browser.close writes cookies first). Found 2026-09-17: after the
        #    agent was stopped, its window stayed open.
        if self._chrome_proc is None and self._close_on_disconnect and self._browser is not None:
            try:
                session = await self._browser.new_browser_cdp_session()
                await session.send("Browser.close")
                logger.info("agent_chrome_closed", note="The agent's own Chrome window from an earlier run.")
            except Exception as e:
                logger.warning("agent_chrome_close_failed", error=str(e)[:120])
            self._close_on_disconnect = False

        # 4. Detach from CDP browser (does NOT kill Chrome — just drops the
        #    Playwright connection so Chrome can flush its state undisturbed)
        if self._browser is not None:
            try:
                if hasattr(self._browser, "disconnect"):
                    pass  # CDP browsers: left running while we flush
            except Exception as e:
                logger.warning("browser_disconnect_error", error=str(e))
            self._browser = None

        # 5. Gracefully close the agent's Chrome, if we launched it.
        #    On Windows, Popen.terminate() = TerminateProcess = hard kill with
        #    NO cookie flush. taskkill WITHOUT /F posts WM_CLOSE — a real
        #    graceful shutdown where Chrome writes cookies, LocalStorage and
        #    sessions to disk before exiting. Force-kill is the last resort.
        if self._chrome_proc is not None:
            pid = self._chrome_proc.pid
            try:
                graceful = subprocess.run(
                    ["taskkill", "/PID", str(pid)],
                    capture_output=True, text=True, timeout=10,
                )
                exited = False
                try:
                    self._chrome_proc.wait(timeout=8)
                    exited = True
                except subprocess.TimeoutExpired:
                    pass

                if not exited:
                    # Graceful close refused (window prompt, background mode) —
                    # force-kill the whole process tree as a last resort.
                    logger.warning("agent_chrome_force_killed", pid=pid,
                                   note="Did not exit gracefully; force-killed. "
                                        "Some session data may not have been saved.")
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True, text=True, timeout=10,
                    )
                elif graceful.returncode == 0:
                    logger.info(
                        "agent_chrome_exited_gracefully",
                        pid=pid,
                        note="Cookies and sessions flushed to disk.",
                    )
            except Exception as e:
                logger.warning("agent_chrome_cleanup_error", error=str(e))
            self._chrome_proc = None

        await self._cleanup_playwright()
        self._connected = False
        logger.info("browser_disconnected")

    async def _cleanup_playwright(self) -> None:
        """Stop the Playwright driver process."""
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    def _ensure_connected(self) -> None:
        """Raise RuntimeError if not connected."""
        if not self._connected or (self._browser is None and self._context is None):
            raise RuntimeError("Not connected to browser. Call connect() first.")
