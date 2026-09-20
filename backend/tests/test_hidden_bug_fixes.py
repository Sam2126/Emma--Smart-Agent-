"""
Hidden bugs found in the full-project review of 2026-09-16, one group per bug.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app.agent.toolkit import output_indicates_failure
from app.tools import local
from app.tools.base import BaseTool


class _NoArgs(BaseModel):
    pass


@pytest.fixture
def tools_enabled(monkeypatch):
    from app.tools import desktop

    monkeypatch.setattr(local, "_check_enabled", lambda: None)
    monkeypatch.setattr(desktop, "_check_enabled", lambda: None)


@pytest.fixture
def fake_powershell(monkeypatch, tools_enabled):
    ran: list[str] = []

    def fake(script, timeout_seconds=60):
        ran.append(script)
        return SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(local, "_run_powershell", fake)
    return ran


# =============================================================================
# The test suite itself used the agent's real database
# =============================================================================

def test_the_suite_never_touches_the_real_agent_memory():
    from app.config import _BACKEND_DIR, get_settings

    settings = get_settings()
    real_db = (_BACKEND_DIR / "data" / "agent.db").resolve().as_posix()
    assert real_db not in settings.db_url
    assert not Path(settings.chroma_path).resolve().is_relative_to(_BACKEND_DIR.resolve())


# =============================================================================
# Local and desktop tools froze the whole server while they ran
# =============================================================================

def test_local_and_desktop_tools_do_not_run_their_bodies_on_the_main_loop():
    """Their bodies block: PowerShell calls, folder scans, a 45 s window watch."""
    for module in ("local.py", "desktop.py"):
        source = (Path(local.__file__).parent / module).read_text(encoding="utf-8")
        assert "run_sync" not in source, module
        assert "_run_in_worker(_async_run)" in source, module


async def test_a_blocking_tool_leaves_the_event_loop_free():
    from app.tools.runtime import run_in_worker
    from app.utils.loop import set_main_loop

    class _SlowLocalTool(BaseTool):
        name: str = "slow_local"
        description: str = "blocks the way a PowerShell call does"
        args_schema: type[BaseModel] = _NoArgs

        def _run(self) -> str:
            async def body():
                time.sleep(0.6)
                return "done"

            return run_in_worker(body)

    set_main_loop(asyncio.get_running_loop())
    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.05)
            ticks += 1

    beat = asyncio.create_task(heartbeat())
    try:
        out = await asyncio.to_thread(_SlowLocalTool().run)
    finally:
        beat.cancel()
        set_main_loop(None)
    assert out == "done"
    assert ticks >= 6, f"the event loop was blocked ({ticks} heartbeats in 0.6 s)"


async def test_a_worker_tool_sends_its_network_call_to_the_main_loop():
    from app.tools.runtime import on_main_loop, run_in_worker
    from app.utils.loop import set_main_loop

    main = asyncio.get_running_loop()
    set_main_loop(main)

    async def which_loop():
        return asyncio.get_running_loop()

    async def body():
        return await on_main_loop(which_loop())

    try:
        used = await asyncio.to_thread(run_in_worker, body)
    finally:
        set_main_loop(None)
    assert used is main


def test_run_in_worker_also_works_inside_a_running_loop():
    from app.tools.runtime import run_in_worker

    async def body():
        return 42

    async def outer():
        return run_in_worker(body)

    assert asyncio.run(outer()) == 42


async def test_a_hung_tool_gives_the_task_back(monkeypatch):
    from app.agent import toolkit

    release = threading.Event()

    class _Hangs(BaseTool):
        name: str = "hangs"
        description: str = "never returns on its own"
        args_schema: type[BaseModel] = _NoArgs

        def _run(self) -> str:
            release.wait(5)
            return "late"

    monkeypatch.setattr(toolkit, "TOOL_TIMEOUT_SECONDS", 0.3)
    try:
        result = await toolkit.execute_tool_call({"hangs": _Hangs()}, "hangs", "{}")
    finally:
        release.set()
    assert result.record["success"] is False
    assert "did not finish" in result.output


# =============================================================================
# Step results: user text read as failure, quiet refusals read as success
# =============================================================================

@pytest.mark.parametrize("tool,output", [
    ("send_keys", "Sent to 'WhatsApp' (whatsapp.root): sorry, I can't come today, the plan failed"),
    ("write_file", r"Wrote C:\Users\x\Desktop\errors\missing.txt (12 chars)."),
    ("find_files", "Found 1 file(s) matching 'failed' (newest first):\n  2026-09-01 10:00  C:\\x\\failed exams.pdf"),
    ("create_folder", r"Folder ready: C:\Users\x\Desktop\Error Logs"),
    ("open_file_or_folder", r"Opened: C:\Users\x\Documents\cannot-delete.docx"),
    ("move_path", r"Moved C:\a\unable.txt to C:\b\unable.txt"),
])
def test_user_text_inside_a_success_message_is_not_a_failure(tool, output):
    """A WhatsApp message saying "can't" made its own successful send look failed,
    so the duplicate guard never recorded it and a retry could send it again."""
    assert output_indicates_failure(output, tool) is False


@pytest.mark.parametrize("tool,output", [
    ("send_keys", "Found a window matching 'WhatsApp' but Windows refused to bring it to the foreground, so NOTHING was sent."),
    ("send_keys", "window_hint is required: which app's window should receive these keys?"),
    ("click_window", "No visible window matching 'Gemini' (waited 15s) — nothing was clicked. Call list_windows."),
    ("focus_window", "Found 'WhatsApp' (whatsapp.root) but Windows refused to bring it to the foreground."),
    ("delete_path", r"Refused: C:\Users\x\OneDrive\Desktop is a system or home folder and will not be deleted."),
    ("open_app", "No app name given."),
])
def test_quiet_refusals_are_failures(tool, output):
    """These contain none of the usual failure words and used to count as done."""
    assert output_indicates_failure(output, tool) is True


def test_the_real_tools_success_messages_are_recognised(tmp_path, tools_enabled, monkeypatch):
    """If a tool's wording changes, this fails instead of every call silently counting as failed."""
    from app.tools import desktop

    folder = tmp_path / "work"
    checks = {
        "create_folder": local.CreateFolderTool().run(path=str(folder)),
        "write_file": local.WriteFileTool().run(path=str(folder / "a.txt"), content="x"),
        "copy_path": local.CopyPathTool().run(source=str(folder / "a.txt"), destination=str(folder / "b.txt")),
        "move_path": local.MovePathTool().run(source=str(folder / "b.txt"), destination=str(folder / "c.txt")),
        "find_files": local.FindFilesTool().run(pattern="a.txt", location=str(tmp_path)),
    }
    monkeypatch.setattr(local.os, "startfile", lambda path: None, raising=False)
    checks["open_file_or_folder"] = local.OpenPathTool().run(path=str(folder / "a.txt"))
    monkeypatch.setattr(local, "_recycle", lambda path: None)
    checks["delete_path"] = local.DeletePathTool().run(path=str(folder / "c.txt"))

    monkeypatch.setattr(desktop, "_ps", lambda script, timeout=60: "SENT|whatsapp.root|WhatsApp")
    checks["send_keys"] = local.SendKeysTool().run(window_hint="WhatsApp", text="hi")
    monkeypatch.setattr(desktop, "_ps", lambda script, timeout=60: "CLICKED|10|20|whatsapp.root :: WhatsApp")
    checks["click_window"] = desktop.ClickWindowTool().run(x=1, y=2, window="WhatsApp")
    monkeypatch.setattr(desktop, "_ps", lambda script, timeout=60: "RESULT|True|whatsapp.root|WhatsApp")
    checks["focus_window"] = desktop.FocusWindowTool().run(window="WhatsApp")

    monkeypatch.setattr(local.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="FOUND|5319275A.WhatsApp!App\n"))
    monkeypatch.setattr(local.subprocess, "Popen", lambda *a, **k: None)
    monkeypatch.setattr(local, "_wait_for_app_window", lambda hint, timeout_ms=25000: ("whatsapp.root :: WhatsApp", "ready"))
    checks["open_app"] = local.OpenAppTool().run(name="WhatsApp")

    for tool, out in checks.items():
        assert output_indicates_failure(out, tool) is False, f"{tool}: {out}"


# =============================================================================
# Searches spent their time inside generated folders and hid that they stopped
# =============================================================================

def test_searches_skip_virtual_environments_and_similar_folders(tmp_path, tools_enabled):
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "prime.py").write_text("def is_prime(): pass")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "prime.js").write_text("is_prime")
    (tmp_path / "MAYANK").mkdir()
    (tmp_path / "MAYANK" / "prime_code.txt").write_text("def is_prime(n): ...")

    content = local.SearchInFilesTool().run(text="is_prime", location=str(tmp_path))
    assert "MAYANK" in content and ".venv" not in content and "node_modules" not in content
    names = local.FindFilesTool().run(pattern="prime", location=str(tmp_path))
    assert "prime_code.txt" in names and ".venv" not in names

    # Searching INSIDE such a folder on purpose still works.
    inside = local.SearchInFilesTool().run(text="is_prime", location=str(tmp_path / ".venv"))
    assert "prime.py" in inside


def test_a_search_cut_short_by_its_time_limit_says_so(tmp_path, tools_enabled, monkeypatch):
    (tmp_path / "a.txt").write_text("nothing")
    monkeypatch.setattr(local, "_SEARCH_TIME_BUDGET_S", -1.0)
    monkeypatch.setattr(local, "_FIND_TIME_BUDGET_S", -1.0)
    content = local.SearchInFilesTool().run(text="is_prime", location=str(tmp_path))
    assert content.startswith("PARTIAL RESULT") and "search was incomplete" in content
    names = local.FindFilesTool().run(pattern="prime", location=str(tmp_path))
    assert "search stopped after" in names


def test_searches_look_at_the_nearest_files_first(tmp_path):
    """Depth-first, a deep project tree used the whole budget before shallow folders."""
    deep = tmp_path / "Projects" / "model" / "dataset" / "train" / "labels"
    deep.mkdir(parents=True)
    for i in range(5):
        (deep / f"{i}.txt").write_text("0 0.5 0.5")
    (tmp_path / "Zeta notes").mkdir()
    (tmp_path / "Zeta notes" / "plan.txt").write_text("is_prime")
    (tmp_path / "top.txt").write_text("x")

    order = [p.name for p in local._walk_files(tmp_path, time.monotonic() + 10)]
    assert order.index("top.txt") < order.index("plan.txt") < order.index("0.txt")


# =============================================================================
# run_command blocked everyday "-Format" commands
# =============================================================================

@pytest.mark.parametrize("command", ['Get-Date -Format "yyyy-MM-dd"', "Get-ChildItem | Format-Table Name"])
def test_everyday_format_commands_run(fake_powershell, command):
    out = local.RunCommandTool().run(command=command)
    assert out.startswith("Exit code 0")
    assert fake_powershell == [command]


@pytest.mark.parametrize("command", [
    "format c:", "cmd /c format D: /q", "Format-Volume -DriveLetter D", "Stop-Computer -Force", "Clear-Disk -Number 1",
])
def test_disk_wiping_and_power_commands_stay_blocked(fake_powershell, command):
    assert local.RunCommandTool().run(command=command).startswith("BLOCKED")
    assert fake_powershell == []


# =============================================================================
# copy_path merged a folder into the destination instead of copying it into it
# =============================================================================

def test_a_folder_is_copied_into_an_existing_folder(tmp_path, tools_enabled):
    (tmp_path / "photos").mkdir()
    (tmp_path / "photos" / "a.jpg").write_text("a")
    (tmp_path / "backup").mkdir()
    out = local.CopyPathTool().run(source=str(tmp_path / "photos"), destination=str(tmp_path / "backup"))
    assert out.startswith("Copied")
    assert (tmp_path / "backup" / "photos" / "a.jpg").read_text() == "a"
    assert not (tmp_path / "backup" / "a.jpg").exists()


def test_a_folder_is_never_copied_into_itself(tmp_path, tools_enabled):
    (tmp_path / "proj" / "sub").mkdir(parents=True)
    out = local.CopyPathTool().run(source=str(tmp_path / "proj"), destination=str(tmp_path / "proj" / "sub"))
    assert "cannot be copied into itself" in out
    assert not (tmp_path / "proj" / "sub" / "proj").exists()


# =============================================================================
# delete_path would recycle the user's whole Desktop
# =============================================================================

def test_the_users_own_top_level_folders_are_never_deleted(tmp_path, tools_enabled, monkeypatch):
    desktop_dir = tmp_path / "OneDrive" / "Desktop"
    (desktop_dir / "old project").mkdir(parents=True)
    monkeypatch.setattr(local, "_known_folder", lambda name: desktop_dir if name == "desktop" else tmp_path / name)
    recycled: list[Path] = []
    monkeypatch.setattr(local, "_recycle", lambda path: recycled.append(path) or None)

    assert local.DeletePathTool().run(path="desktop").startswith("Refused")
    assert local.DeletePathTool().run(path=str(desktop_dir)).startswith("Refused")
    assert recycled == []

    inside = local.DeletePathTool().run(path=str(desktop_dir / "old project"))
    assert inside.startswith("Moved to the Recycle Bin")
    assert recycled == [(desktop_dir / "old project").resolve()]


# =============================================================================
# read_file loaded whole files into memory
# =============================================================================

def test_a_huge_file_is_only_partly_read(tmp_path, tools_enabled):
    big = tmp_path / "big.log"
    big.write_bytes(b"x" * (local._MAX_TEXT_READ_BYTES + 10))
    out = local.ReadFileTool().run(path=str(big), max_chars=500)
    assert "only its beginning was read" in out


def test_reading_a_folder_says_what_to_do(tmp_path, tools_enabled):
    assert "it is a folder" in local.ReadFileTool().run(path=str(tmp_path))


# =============================================================================
# Non-English text was garbled through PowerShell
# =============================================================================

def test_clipboard_text_is_read_back_as_utf8(fake_powershell):
    local.ClipboardTool().run(text="नमस्ते")
    assert "-Encoding UTF8" in fake_powershell[0]


def test_powershell_command_scripts_ask_for_utf8_output(monkeypatch):
    from app.tools import desktop

    seen = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        return SimpleNamespace(stdout="ok", stderr="")

    monkeypatch.setattr(desktop.subprocess, "run", fake_run)
    desktop._ps("Write-Output 'x'")
    assert seen["args"][-1].startswith(desktop._PS_UTF8)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell")
def test_non_english_text_survives_a_real_powershell_call():
    from app.tools import desktop

    assert desktop._ps("Write-Output 'नमस्ते café'") == "नमस्ते café"


# =============================================================================
# Learning
# =============================================================================

async def test_a_first_attempt_that_ran_no_tool_is_not_stored(monkeypatch):
    """Found 2026-09-16: "I have no tool to find the latest PowerPoint", from an
    attempt that never tried, was saved and would be recalled as a lesson."""
    from app.agent.nodes import replanner
    from app.state.reflection import OutcomeReflection

    scheduled: list[str] = []

    def fake_schedule(coro, name):
        coro.close()
        scheduled.append(name)

    async def reflect(**kwargs):
        return OutcomeReflection("None", "no tool used", "none", "use list_recent_files", from_llm=True)

    async def plan(state, retry_context=""):
        return "1. list_recent_files"

    monkeypatch.setattr(replanner, "schedule_background", fake_schedule)
    monkeypatch.setattr(replanner.brain.reflection, "reflect_on_outcome", reflect)
    monkeypatch.setattr(replanner, "make_plan", plan)

    state = {
        "task_id": "t1", "task_instruction": "attach the last ppt", "trajectory": [],
        "error": "no tool", "final_report": "I don't have a tool", "experiences": [],
    }
    update = await replanner.replan_node(state, {})
    assert scheduled == [] and update["attempt"] == 2

    state["trajectory"] = [{"action_type": "find_files", "success": False}]
    await replanner.replan_node(state, {})
    assert scheduled == ["learn:t1#attempt1"]


async def test_a_rating_on_a_run_that_was_not_stored_is_learned(semantic_memory):
    """A 👎 with a note on a run that used no tool used to wait forever for an
    experience that was never going to be written."""
    from app.state.brain import BrainMemory

    brain = BrainMemory(semantic=semantic_memory, reflection=object())
    brain.note_unstored_run(
        task_id="t-none", instruction="attach the last ppt from downloads to gemini",
        domain="local", scope="local", success=False,
        error="Marked incomplete: no tool ran at all", report="I cannot do that",
    )
    result = await brain.record_feedback("t-none", -1, "it did nothing, use list_recent_files")
    assert result["applied"] is True

    recalled = semantic_memory.recall_sync("attach the last ppt from downloads to gemini", min_similarity=0.0)
    stored = next(e for e in recalled if e.task_id == "t-none")
    assert stored.feedback == -1
    assert stored.feedback_comment == "it did nothing, use list_recent_files"
    assert "no tool was used" in stored.lesson


def test_success_and_failure_name_the_same_skill():
    from app.state.brain import skill_type_for_run

    assert skill_type_for_run({"write_file"}, "search my notes and write a summary") == "file_write"
    assert skill_type_for_run({"click_element"}, "search headphones on amazon") == "search_product"
    assert skill_type_for_run(set(), "open the docs") == "view_content"


async def test_a_failed_local_step_is_not_treated_as_a_page_selector(monkeypatch):
    from app.state.brain import BrainMemory

    brain = BrainMemory(semantic=object(), reflection=object())
    deprecated: list[str] = []

    async def deprecate(domain, selector, threshold=3):
        deprecated.append(selector)

    monkeypatch.setattr(brain._domain_store, "deprecate_failed_selector", deprecate)
    trajectory = [
        {"action_type": "write_file", "selector_used": r"C:\x\notes.txt", "success": False},
        {"action_type": "click_element", "selector_used": "#buy-box", "success": False},
    ]
    await brain._degrade_skills_and_selectors("amazon.in", "search headphones", trajectory)
    assert deprecated == ["#buy-box"]


async def test_page_selectors_are_learned_from_the_current_tool_names(monkeypatch):
    """The checks expected the old engine's "type" / "add_to_cart" steps, so
    no selector had been learned from a run since the engine changed."""
    from app.state.brain import BrainMemory

    brain = BrainMemory(semantic=object(), reflection=object())
    recorded: dict = {}

    async def record(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(brain._domain_store, "record_domain_learning", record)
    trajectory = [
        {"action_type": "type_into_element", "selector_used": 'textarea[name="q"]',
         "args": {"press_enter": True}, "success": True},
        {"action_type": "click_element", "selector_used": 'button:has-text("Close")', "success": True},
    ]
    await brain._update_domain_memory("google.com", trajectory, True, "")
    assert recorded["search_selectors"] == ['textarea[name="q"]']
    assert recorded["popup_dismiss_selectors"] == ['button:has-text("Close")']


async def test_a_local_run_is_only_counted(monkeypatch):
    from app.state.brain import BrainMemory

    brain = BrainMemory(semantic=object(), reflection=object())
    calls: list[dict] = []

    async def record(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(brain._domain_store, "record_domain_learning", record)
    trajectory = [{"action_type": "write_file", "selector_used": r"C:\close\dismiss.txt", "success": True}]
    await brain._update_domain_memory("local", trajectory, True, "Wrote the file")
    assert calls == [{"domain_or_url": "local", "success": True}]


async def test_tips_and_reflections_do_not_count_as_extra_runs():
    """One task used to count as two or three runs of its domain."""
    from app.state.domain_memory import DomainMemoryStore

    store = DomainMemoryStore()
    domain = f"count-test-{uuid.uuid4().hex[:8]}.com"
    await store.record_domain_learning(domain, success=True)
    await store.record_domain_learning(domain, general_tips="[REFLEXION] x", success=True, count_run=False)
    await store.record_domain_learning(domain, general_tips="a tip", success=False, count_run=False)
    memory = await store.get_domain_memory(domain)
    assert (memory["successful_runs"], memory["failed_runs"]) == (1, 0)


async def test_pruning_a_selector_removes_only_that_selector():
    """A failed "button" used to wipe out every stored 'button:has-text(...)'."""
    from app.state.domain_memory import DomainMemoryStore

    store = DomainMemoryStore()
    domain = f"prune-test-{uuid.uuid4().hex[:8]}.com"
    await store.record_domain_learning(domain, search_selectors=["button", 'button:has-text("Search")'], success=False)
    await store.deprecate_failed_selector(domain, "button", threshold=1)
    memory = await store.get_domain_memory(domain)
    assert memory["search_selectors"] == ['button:has-text("Search")']


# =============================================================================
# The model retyped identifiers wrongly
# =============================================================================

def test_identifiers_and_file_names_are_pinned_verbatim():
    from app.agent.prompts import extract_probable_proper_nouns

    pinned = extract_probable_proper_nouns(
        "search my desktop for zz_text_that_is_nowhere_qq and is_prime, then open prime_code.txt"
    )
    assert pinned == ["zz_text_that_is_nowhere_qq", "is_prime", "prime_code.txt"]
    # Ordinary words, sentence ends and contractions are left alone.
    assert extract_probable_proper_nouns("write today's date. then stop.") == []


# =============================================================================
# Paths that depended on the folder the backend was started from
# =============================================================================

def test_a_relative_database_path_means_the_backend_folder():
    from app.config import _BACKEND_DIR, Settings

    expected = "sqlite+aiosqlite:///" + (_BACKEND_DIR / "data" / "agent.db").resolve().as_posix()
    assert Settings(db_url="sqlite+aiosqlite:///./data/agent.db").db_url == expected
    assert Settings(db_url="sqlite+aiosqlite:///C:/x/agent.db").db_url == "sqlite+aiosqlite:///C:/x/agent.db"
    assert Settings(db_url="sqlite+aiosqlite:///:memory:").db_url == "sqlite+aiosqlite:///:memory:"
    assert Settings(db_url="postgresql+asyncpg://u@h/db").db_url == "postgresql+asyncpg://u@h/db"


def test_the_adapter_registry_is_found_from_any_folder(tmp_path, monkeypatch):
    from app.rl.registry import AdapterRegistry

    monkeypatch.chdir(tmp_path)
    registry = AdapterRegistry()
    assert registry.registry_path.is_absolute()
    assert registry.registry_path.parts[-4:] == ("backend", "data", "rl", "adapter_registry.json")
    assert not (tmp_path / "data").exists()


# =============================================================================
# Voice: rejected audio was re-sent with every key
# =============================================================================

async def test_rejected_audio_is_not_resent_with_every_key(monkeypatch):
    from app import voice
    from app.utils.key_pool import KeyPool

    posts: list[str] = []

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, *args, **kwargs):
            posts.append(kwargs["headers"]["Authorization"])
            return SimpleNamespace(status_code=400, text="could not process file", json=lambda: {})

    monkeypatch.setattr(voice, "get_key_pool", lambda: KeyPool(["k1", "k2", "k3", "k4"]))
    monkeypatch.setattr(voice.httpx, "AsyncClient", _Client)
    with pytest.raises(RuntimeError, match="rejected the audio"):
        await voice.transcribe_audio(b"not audio")
    assert len(posts) == 1


# =============================================================================
# Browser: the code-editor typing fallback never ran
# =============================================================================

async def test_the_editor_fallback_passes_the_text_to_the_page(monkeypatch):
    from app.browser import actions

    monkeypatch.setattr(actions.asyncio, "sleep", _no_sleep)
    calls: dict = {}

    async def fail(*args, **kwargs):
        raise RuntimeError("not a plain input")

    async def ok(*args, **kwargs):
        return None

    async def count():
        return 1

    async def evaluate(expression, arg=None, timeout=None):
        calls["expression"], calls["arg"] = expression, arg
        return True

    locator = SimpleNamespace(
        count=count, wait_for=ok, scroll_into_view_if_needed=ok, click=ok,
        fill=fail, focus=fail, evaluate=evaluate,
    )
    locator.first = locator
    page = SimpleNamespace(locator=lambda selector: locator, url="https://editor.example")

    result = await actions._type_text(page, ".CodeMirror", "print('hi')", 2000)
    assert result.success is True
    assert calls["arg"] == "print('hi')"
    assert calls["expression"].strip().startswith("(el, val)")


_real_sleep = asyncio.sleep


async def _no_sleep(seconds, *args, **kwargs):
    await _real_sleep(0)


def test_upload_reads_folder_names_like_the_file_tools(tmp_path, monkeypatch):
    from app.tools import browser

    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    monkeypatch.setattr(local, "_known_folder", lambda name: downloads if name == "downloads" else tmp_path / name)
    out = browser.UploadFileTool().run(path=r"Downloads\missing deck.pptx")
    assert out.startswith("File not found")
    assert str(downloads / "missing deck.pptx") in out


async def test_reconnecting_stops_the_previous_playwright_driver(monkeypatch):
    """Each reconnect used to leave one Playwright driver process running."""
    from app.browser import controller as ctl

    stopped: list[str] = []

    class _Driver:
        def __init__(self, name):
            self.name = name
            self.chromium = SimpleNamespace(connect_over_cdp=self._refuse)

        async def _refuse(self, *args, **kwargs):
            raise ConnectionError("no chrome")

        async def stop(self):
            stopped.append(self.name)

    class _Starter:
        async def start(self):
            return _Driver("new")

    monkeypatch.setattr(ctl, "async_playwright", lambda: _Starter())
    monkeypatch.setattr(ctl, "_port_open", lambda host, port, timeout=0.5: False)
    browser_controller = ctl.BrowserController()
    browser_controller._playwright = _Driver("old")
    await browser_controller.connect("http://127.0.0.1:9", allow_launch=False)
    assert stopped[0] == "old"


# =============================================================================
# Server plumbing
# =============================================================================

async def test_progress_sends_are_kept_until_they_have_run():
    from app.websocket.protocol import ErrorMessage
    from app.websocket.server import WebSocketServer

    sent: list[str] = []

    class _Socket:
        async def send(self, data):
            await asyncio.sleep(0)
            sent.append(data)

    server = WebSocketServer()
    server._send_soon(_Socket(), ErrorMessage(message="x"))
    assert len(server._send_tasks) == 1
    await asyncio.sleep(0.05)
    assert sent and not server._send_tasks


def test_the_dashboard_uses_the_engines_brain():
    from app import main
    from app.agent.services import brain

    assert main._brain is brain
