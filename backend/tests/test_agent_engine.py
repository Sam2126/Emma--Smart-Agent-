"""
The unified LangGraph engine end to end, with scripted LLM replies and fake
tools: the actor tool loop, streaming progress, verification of local tasks,
automatic retry with a changed plan, the retry safety block, and the
explanation. No network, no real desktop actions, no database writes.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from pydantic import BaseModel

from app.agent import services
from app.agent.nodes import actor as actor_node
from app.agent.nodes import planner as planner_node
from app.agent.nodes import recall as recall_node
from app.agent.runner import AgentRunner
from app.agent.toolkit import (
    compact_messages,
    describe_action,
    local_completion_shortfall,
    retry_block_reason,
)
from app.config import get_settings
from app.state.reflection import OutcomeReflection, StrategyAdvice
from app.state.semantic_memory import Experience
from app.tools.base import BaseTool
from tests.conftest import llm_message, tool_call


# =============================================================================
# Fakes
# =============================================================================

class _Args(BaseModel):
    query: str = ""
    path: str = ""
    window_hint: str = ""
    text: str = ""
    keys: str = ""


def _make_tool(tool_name: str, outputs: list[str], log: list):
    class _Tool(BaseTool):
        name: str = tool_name
        description: str = f"fake {tool_name}"
        args_schema: type[BaseModel] = _Args

        def _run(self, **kwargs) -> str:
            log.append((tool_name, kwargs))
            return outputs.pop(0) if len(outputs) > 1 else outputs[0]

    return _Tool()


class _ScriptedChat:
    """Replaces app.utils.llm.chat: plans for the planner, a queue for the actor."""

    def __init__(self, actor_replies: list, plans: list[str] | None = None):
        self.actor_replies = list(actor_replies)
        self.plans = list(plans or ["1. do it"])
        self.roles: list[str] = []
        self.actor_messages: list[list] = []
        self.tool_lists: list[list] = []

    async def __call__(self, role, messages, **kwargs):
        self.roles.append(role)
        if role == "planner":
            return llm_message(self.plans.pop(0) if len(self.plans) > 1 else self.plans[0])
        self.actor_messages.append(messages)
        self.tool_lists.append(kwargs.get("tools"))
        item = self.actor_replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeReflection:
    def __init__(self, advice: StrategyAdvice | None = None):
        self.advice = advice
        self.outcomes = 0

    async def plan_strategy(self, instruction, experiences, soft_deadline=None):
        return self.advice

    async def reflect_on_outcome(self, **kwargs):
        self.outcomes += 1
        return OutcomeReflection("None", "searched the wrong folder", "file is in Downloads", "list recent downloads")


@pytest.fixture
def engine(monkeypatch):
    """Isolate the graph from memory, databases and real tools."""
    s = get_settings()
    monkeypatch.setattr(s, "auto_retry_on_failure", True)
    monkeypatch.setattr(s, "task_timeout_seconds", 60)
    monkeypatch.setattr(s, "two_tier_actor", True)

    brain = services.brain
    learned = {"sql": [], "semantic": [], "experiences": []}
    reflection = _FakeReflection()

    async def recall_semantic(instruction, exclude_task_id=None):
        return learned["experiences"]

    async def learn_from_task(**kwargs):
        learned["sql"].append(kwargs)

    async def learn_semantic(**kwargs):
        learned["semantic"].append(kwargs)

    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(brain, "recall_semantic", recall_semantic)
    monkeypatch.setattr(brain, "learn_from_task", learn_from_task)
    monkeypatch.setattr(brain, "learn_semantic", learn_semantic)
    monkeypatch.setattr(brain, "_reflection", reflection)
    monkeypatch.setattr(services.episodic_logger, "log_task_start", noop)
    monkeypatch.setattr(services.episodic_logger, "finalize_task", noop)

    from app.state.domain_memory import DomainMemoryStore

    async def get_domain_memory(self, domain):
        return {}

    async def record_domain_learning(self, *args, **kwargs):
        return None

    monkeypatch.setattr(DomainMemoryStore, "get_domain_memory", get_domain_memory)
    monkeypatch.setattr(DomainMemoryStore, "record_domain_learning", record_domain_learning)

    log: list = []

    def install(tools: dict[str, list[str]], chat: _ScriptedChat):
        toolset = {name: _make_tool(name, list(outputs), log) for name, outputs in tools.items()}
        monkeypatch.setattr(recall_node, "build_toolset", lambda scope: toolset)
        monkeypatch.setattr(actor_node, "build_toolset", lambda scope: toolset)
        monkeypatch.setattr(actor_node, "chat", chat)
        monkeypatch.setattr(planner_node, "chat", chat)
        return log

    return install, learned, reflection


async def _run(instruction: str, scope: str = "local"):
    updates: list[dict] = []
    result = await AgentRunner(status_callback=updates.append).run_task(instruction, task_id="t-engine", scope=scope)
    from app.agent.context import drain_background_tasks

    await drain_background_tasks(timeout=5)
    return result, updates


# =============================================================================
# Full graph runs
# =============================================================================

async def test_successful_local_task(engine):
    install, learned, _ = engine
    chat = _ScriptedChat([
        llm_message(tool_calls=[tool_call("find_files", {"query": "resume"})]),
        llm_message(tool_calls=[tool_call("open_file_or_folder", {"path": "C:/Users/x/resume.pdf"}, "c2")]),
        llm_message("Opened C:/Users/x/resume.pdf"),
    ])
    log = install({
        "find_files": ["Found 1 file(s) matching 'resume' (newest first):\n  2026-09-01 10:00  C:/Users/x/resume.pdf"],
        "open_file_or_folder": ["Opened: C:/Users/x/resume.pdf"],
    }, chat)

    result, updates = await _run("open my resume pdf")

    assert result["success"] is True
    assert result["retried"] is False
    assert [s["action_type"] for s in result["trajectory"]] == ["find_files", "open_file_or_folder"]
    assert log[0][0] == "find_files" and log[0][1]["query"] == "resume"
    # Two-tier: first turn on the action model, routine turns on the fast one.
    assert [r for r in chat.roles if r.startswith("actor")] == ["actor", "actor_fast", "actor_fast"]
    # Streaming progress: every phase and each tool call.
    steps = [u["current_step"] for u in updates]
    assert any("Recalling" in s for s in steps)
    assert any("Looking for files" in s for s in steps)
    assert any("Opening" in s for s in steps)
    # Results are streamed too, not just actions.
    assert "✅ Found 1 file(s) matching 'resume' (newest first):" in steps
    assert updates[-1]["status"] == "completed" and updates[-1]["progress"] == 1.0
    # Learning happened: SQL awaited, semantic in the background.
    assert learned["sql"] and learned["sql"][0]["success"] is True
    assert learned["semantic"] and learned["semantic"][0]["task_id"] == "t-engine"
    assert "What I did: ✓ find_files" in result["explanation"]
    assert "planned from scratch" in result["explanation"]


async def test_failed_attempt_is_retried_with_new_strategy(engine):
    install, learned, reflection = engine
    chat = _ScriptedChat(
        actor_replies=[
            llm_message(tool_calls=[tool_call("find_files", {"query": "report"})]),
            llm_message("I could not find the report."),
            llm_message(tool_calls=[tool_call("list_recent_files", {"path": "downloads"}, "c3")]),
            llm_message("Found and opened C:/Users/x/Downloads/report.pdf"),
        ],
        plans=["1. find_files report", "1. list_recent_files downloads"],
    )
    install({
        "find_files": ["No files matching 'report'"],
        "list_recent_files": ["C:/Users/x/Downloads/report.pdf"],
    }, chat)

    result, updates = await _run("open the latest report")

    assert result["success"] is True
    assert result["retried"] is True
    assert [s["action_type"] for s in result["trajectory"]] == ["list_recent_files"]
    assert reflection.outcomes == 1
    # The retry prompt carries the failure and the lesson.
    retry_prompt = chat.actor_messages[2][1]["content"]
    assert "PREVIOUS ATTEMPT FAILED" in retry_prompt and "list recent downloads" in retry_prompt
    # The failed first attempt is stored as its own experience.
    assert {k["task_id"] for k in learned["semantic"]} == {"t-engine#attempt1", "t-engine"}
    # Trajectory comparison: the final lesson is written with the failed first attempt to diff against.
    final = next(k for k in learned["semantic"] if k["task_id"] == "t-engine")
    assert final["previous_attempt"]["error"]
    assert [s["action_type"] for s in final["previous_attempt"]["trajectory"]] == ["find_files"]
    assert any(u["status"] == "replanning" for u in updates)
    assert "retried once" in result["explanation"]


async def test_retry_blocked_after_irreversible_step(engine):
    install, _, reflection = engine
    chat = _ScriptedChat([
        llm_message(tool_calls=[tool_call("send_keys", {"window_hint": "WhatsApp", "text": "hello"})]),
        llm_message("Failed: the message could not be verified."),
    ])
    install({"send_keys": ["Sent to 'WhatsApp' (whatsapp.root): hello"]}, chat)

    result, _ = await _run("open WhatsApp and message Rakesh hello")

    assert result["success"] is False
    assert result["retried"] is False
    assert reflection.outcomes == 0
    assert "did not retry automatically" in result["explanation"]
    assert "send_keys" in result["explanation"]


async def test_opening_app_alone_is_marked_incomplete(engine):
    install, _, _ = engine
    chat = _ScriptedChat([
        llm_message(tool_calls=[tool_call("open_app", {"query": "WhatsApp"})]),
        llm_message("Task complete: WhatsApp is open."),
        # Retry is allowed (open_app is safe to repeat); second attempt does the work.
        llm_message(tool_calls=[tool_call("send_keys", {"window_hint": "WhatsApp", "text": "Rakesh"}, "c2")]),
        llm_message("Searched Rakesh in WhatsApp."),
    ])
    install({"open_app": ["Launched: WhatsApp"], "send_keys": ["Sent to 'WhatsApp' (whatsapp.root): Rakesh"]}, chat)

    result, _ = await _run("open WhatsApp and search Rakesh")

    assert result["retried"] is True
    assert "nothing was typed or clicked" in result["explanation"] or "first attempt failed" in result["explanation"]
    assert result["success"] is True


async def test_recalled_experiences_feed_strategy_and_explanation(engine):
    install, learned, reflection = engine
    learned["experiences"] = [Experience(
        task_id="old", instruction="message bhawesh on whatsapp", domain="local", scope="local", success=True,
        tools=["open_app", "send_keys"], lesson="use ctrl+f", error="", feedback=1, user_disputed=False,
        created_at="2026-09-10T10:00:00+00:00", similarity=0.7,
    )]
    reflection.advice = StrategyAdvice("STRATEGY:\n- open_app then ctrl+f\nAVOID:\n- clicking", from_llm=True)
    chat = _ScriptedChat([
        llm_message(tool_calls=[tool_call("send_keys", {"window_hint": "WhatsApp", "keys": "^f"})]),
        llm_message("Sent hello to Priyal."),
    ])
    install({"send_keys": ["Sent to 'WhatsApp' (whatsapp.root): ^f"]}, chat)

    result, _ = await _run("message Priyal hello on WhatsApp")

    assert result["success"] is True
    assert "1 similar past task" in result["explanation"]
    assert "open_app then ctrl+f" in result["explanation"]


async def test_llm_crash_on_action_model_fails_cleanly(engine):
    install, _, _ = engine
    chat = _ScriptedChat([RuntimeError("groq exploded")] * 3)
    install({"find_files": ["x"]}, chat)
    result, _ = await _run("find a file")
    assert result["success"] is False
    assert "LLM call failed" in result["error"]


async def test_fast_model_error_escalates_instead_of_failing(engine):
    install, _, _ = engine
    chat = _ScriptedChat([
        llm_message(tool_calls=[tool_call("find_files", {"query": "a"})]),
        RuntimeError("fast model 404"),
        llm_message("Found C:/a.txt"),
    ])
    install({"find_files": ["Found 1 file(s) matching 'a' (newest first):\n  2026-09-01 10:00  C:/a.txt"]}, chat)
    result, _ = await _run("find a.txt")
    assert result["success"] is True
    assert [r for r in chat.roles if r.startswith("actor")] == ["actor", "actor_fast", "actor"]


# =============================================================================
# Toolkit units
# =============================================================================

def test_retry_block_reason():
    assert retry_block_reason([], "timeout")
    assert retry_block_reason([{"action_type": "find_files", "success": True}], "verification") is None
    assert retry_block_reason([{"action_type": "send_keys", "success": False}], "verification") is None
    assert "send_keys" in retry_block_reason([{"action_type": "send_keys", "success": True}], "verification")


def test_local_completion_shortfall():
    assert local_completion_shortfall("open WhatsApp and search Rakesh", [{"action_type": "open_app"}])
    assert local_completion_shortfall("open WhatsApp", [{"action_type": "open_app"}]) is None
    assert local_completion_shortfall(
        "open WhatsApp and search Rakesh", [{"action_type": "open_app"}, {"action_type": "send_keys"}]
    ) is None


def test_describe_action():
    assert describe_action("send_keys", {"window_hint": "WhatsApp", "text": "Rakesh"}) == "⌨️ Typing 'Rakesh' in WhatsApp"
    assert describe_action("send_keys", {"window_hint": "WhatsApp", "keys": "^f"}) == "⌨️ Pressing Ctrl+F in WhatsApp"
    assert describe_action("navigate_browser", {"url": "https://a.com"}).endswith("https://a.com")
    assert describe_action("mystery", {}) == "🔧 mystery"


async def test_tasks_run_one_at_a_time(engine, monkeypatch):
    """Found 2026-09-15: a typed task started while a wake-word task was still
    running, and both drove the same browser tab."""
    install, _, _ = engine
    spans: list[tuple[float, float]] = []

    class _FindArgs(BaseModel):
        query: str = ""

    class _SlowFind(BaseTool):
        name: str = "find_files"
        description: str = "slow fake"
        args_schema: type[BaseModel] = _FindArgs

        def _run(self, query: str = "") -> str:
            start = time.monotonic()
            time.sleep(0.3)
            spans.append((start, time.monotonic()))
            return f"Found 1 file(s) matching '{query}' (newest first):\n  2026-09-01 10:00  C:/{query}.txt"

    chat = _ScriptedChat([
        llm_message(tool_calls=[tool_call("find_files", {"query": "a"})]),
        llm_message("Found C:/a.txt"),
        llm_message(tool_calls=[tool_call("find_files", {"query": "b"}, "c2")]),
        llm_message("Found C:/b.txt"),
    ])
    install({"find_files": ["unused"]}, chat)
    toolset = {"find_files": _SlowFind()}
    monkeypatch.setattr(recall_node, "build_toolset", lambda scope: toolset)
    monkeypatch.setattr(actor_node, "build_toolset", lambda scope: toolset)

    waiting_updates: list[dict] = []
    first = asyncio.create_task(AgentRunner().run_task("find a", task_id="lane-a", scope="local"))
    await asyncio.sleep(0.05)
    second = asyncio.create_task(
        AgentRunner(status_callback=waiting_updates.append).run_task("find b", task_id="lane-b", scope="local")
    )
    result_a, result_b = await asyncio.gather(first, second)

    assert result_a["success"] and result_b["success"]
    assert len(spans) == 2 and spans[0][1] <= spans[1][0], "the second task's tool ran only after the first finished"
    assert any("Waiting for the current task" in u["current_step"] for u in waiting_updates)


async def test_browser_tools_are_enabled_on_demand_for_local_tasks(engine):
    install, _, _ = engine
    chat = _ScriptedChat([
        llm_message(tool_calls=[tool_call("use_browser", {}, "c0")]),
        llm_message(tool_calls=[tool_call("navigate_browser", {"query": "notes"}, "c1")]),
        llm_message("Opened the notes page."),
    ])
    log = install({
        "find_files": ["Found 1 file(s) matching 'notes' (newest first):\n  2026-09-01 10:00  C:/notes.txt"],
        "navigate_browser": ["Navigated"],
    }, chat)

    result, updates = await _run("open my notes file")

    first = {t["function"]["name"] for t in chat.tool_lists[0]}
    second = {t["function"]["name"] for t in chat.tool_lists[1]}
    assert first == {"find_files", "use_browser"}
    assert "navigate_browser" in second and "use_browser" not in second
    assert [s["action_type"] for s in result["trajectory"]] == ["navigate_browser"]
    assert log[0][0] == "navigate_browser"
    assert any("Enabling browser tools" in u["current_step"] for u in updates)
    assert result["success"] is True


async def test_rate_limited_fast_model_is_skipped(engine, monkeypatch):
    install, _, _ = engine
    monkeypatch.setattr(actor_node, "role_is_rate_limited", lambda role: True)
    chat = _ScriptedChat([
        llm_message(tool_calls=[tool_call("find_files", {"query": "a"})]),
        llm_message(tool_calls=[tool_call("find_files", {"query": "b"}, "c2")]),
        llm_message("Found C:/a.txt"),
    ])
    install({"find_files": ["Found 1 file(s) matching 'a' (newest first):\n  2026-09-01 10:00  C:/a.txt"]}, chat)
    await _run("find a.txt")
    assert [r for r in chat.roles if r.startswith("actor")] == ["actor", "actor", "actor"]


def _whatsapp_calls(*steps):
    return [("send_keys", {"window_hint": "WhatsApp", **s}) for s in steps]


def test_send_tracker_search_text_is_not_a_message():
    from app.agent.toolkit import SendTracker

    tracker = SendTracker("Open WhatsApp and search Rakesh and say hello to him")
    notes = [tracker.record(n, a) for n, a in _whatsapp_calls(
        {"keys": "^f"}, {"keys": "^a{BACKSPACE}"}, {"text": "Rakesh"}, {"keys": "{DOWN}"}, {"keys": "{ENTER}"},
        {"text": "Hello"}, {"keys": "{ENTER}"},
    )]
    assert tracker.sent == {("whatsapp", "hello"): "Hello"}
    assert "[SENT] 'Hello'" in notes[-1] and not any(notes[:-1])
    assert tracker.duplicate_of("send_keys", {"window_hint": "WhatsApp", "text": "Hello"}) == "Hello"
    assert tracker.duplicate_of("send_keys", {"window_hint": "WhatsApp", "text": "Rakesh"}) is None
    # A new search after the send is still allowed.
    tracker.record("send_keys", {"window_hint": "WhatsApp", "keys": "^f"})
    assert tracker.duplicate_of("send_keys", {"window_hint": "WhatsApp", "text": "hello"}) is None


def test_send_tracker_search_typed_via_keys_and_enter_only_once():
    from app.agent.toolkit import SendTracker

    tracker = SendTracker("message Rakesh hello")
    for n, a in _whatsapp_calls({"text": "Hello"}, {"keys": "{DOWN}"}, {"keys": "{ENTER}"}):
        tracker.record(n, a)
    assert tracker.sent == {}, "text followed by other keys before Enter was not what Enter submitted"


def test_send_tracker_allows_requested_repeats():
    from app.agent.toolkit import SendTracker

    tracker = SendTracker("say hello to Rakesh twice on WhatsApp")
    for n, a in _whatsapp_calls({"text": "Hello"}, {"keys": "{ENTER}"}):
        tracker.record(n, a)
    assert tracker.duplicate_of("send_keys", {"window_hint": "WhatsApp", "text": "Hello"}) is None


def test_perception_turn_uses_the_action_model():
    from app.utils.llm import actor_role_for_turn

    assert actor_role_for_turn(3, False, "send_keys") == "actor_fast"
    assert actor_role_for_turn(3, False, "see_window") == "actor"


async def test_message_already_sent_is_never_sent_again(engine):
    """Replays the production run: after sending, the model looked at the window,
    clicked, and tried to type 'Hello' again (twice)."""
    install, _, _ = engine
    sk = lambda i, **a: llm_message(tool_calls=[tool_call("send_keys", {"window_hint": "WhatsApp", **a}, f"c{i}")])
    chat = _ScriptedChat([
        llm_message(tool_calls=[tool_call("open_app", {"query": "WhatsApp"}, "c0")]),
        sk(1, keys="^f"), sk(2, keys="^a{BACKSPACE}"), sk(3, text="Rakesh"), sk(4, keys="{DOWN}"), sk(5, keys="{ENTER}"),
        sk(6, text="Hello"), sk(7, keys="{ENTER}"),
        llm_message(tool_calls=[tool_call("see_window", {"window_hint": "WhatsApp"}, "c8")]),
        sk(9, text="Hello"),
        sk(10, text="Hello"),
        llm_message("should not be reached"),
    ])
    log = install({
        "open_app": ["Launched: WhatsApp"],
        "send_keys": ["Sent to 'WhatsApp' (whatsapp.root): done"],
        "see_window": ["chat with Rakesh"],
    }, chat)

    result, updates = await _run("Open WhatsApp and search Rakesh and say hello to him")

    typed_hello = [a for n, a in log if n == "send_keys" and a.get("text") == "Hello"]
    assert len(typed_hello) == 1, "Hello must be typed exactly once"
    assert result["success"] is True
    assert "exactly once" in result["summary"]
    assert any("Not sending 'Hello' again" in u["current_step"] for u in updates)
    # The turn after see_window ran on the action model.
    actor_roles = [r for r in chat.roles if r.startswith("actor")]
    assert actor_roles[9] == "actor"


def test_compact_messages_shortens_only_old_tool_results():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    msgs += [{"role": "tool", "content": "x" * 2000} for _ in range(12)]
    out = compact_messages(msgs)
    assert "older result shortened" in out[2]["content"]
    assert out[-1]["content"] == "x" * 2000
    assert len(out) == len(msgs)
