"""
Level 3 learning: semantic experience memory, LLM reflection, user feedback,
A/B strategy tracking and explanations.

Embeddings use the offline HashEmbedding from conftest, and LLM calls use a
scripted completion function, so these tests need no network or model download.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.agent.explain import build_explanation
from app.config import get_settings
from app.state.brain import BrainMemory, local_skill_type_for
from app.state.reflection import (
    OutcomeReflection,
    ReflectionEngine,
    StrategyAdvice,
    _parse_json_object,
    fallback_strategy,
)
from app.state.semantic_memory import Experience, summarize_strategies, tool_sequence
from tests.conftest import ScriptedCompletion, llm_message

WHATSAPP_OK = [
    {"action_type": "open_app", "success": True},
    {"action_type": "send_keys", "success": True},
    {"action_type": "send_keys", "success": True},
    {"action_type": "see_window", "success": True},
]


@pytest.fixture
def level3_settings(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "semantic_memory_enabled", True)
    monkeypatch.setattr(s, "llm_reflection_enabled", True)
    monkeypatch.setattr(s, "semantic_min_similarity", 0.2)
    monkeypatch.setattr(s, "semantic_recall_k", 5)
    monkeypatch.setattr(s, "reflection_cache_ttl_seconds", 3600)
    return s


def _exp(task_id="t", instruction="send hello on whatsapp", success=True, tools=None, feedback=0,
         disputed=False, lesson="", error="", similarity=0.8, created_at=None) -> Experience:
    return Experience(
        task_id=task_id, instruction=instruction, domain="local", scope="local", success=success,
        tools=tools if tools is not None else ["open_app", "send_keys"], lesson=lesson, error=error,
        feedback=feedback, user_disputed=disputed,
        created_at=created_at or datetime.now(timezone.utc).isoformat(), similarity=similarity,
    )


# =============================================================================
# Semantic memory
# =============================================================================

def test_tool_sequence_collapses_repeats():
    assert tool_sequence(WHATSAPP_OK) == ["open_app", "send_keys", "see_window"]
    assert tool_sequence([]) == []


def test_upsert_and_recall_by_meaning(semantic_memory, level3_settings):
    semantic_memory.upsert_sync(task_id="a", instruction="open whatsapp and send hello to rakesh", domain="local",
                                scope="local", success=True, trajectory=WHATSAPP_OK, lesson="use ctrl+f")
    semantic_memory.upsert_sync(task_id="b", instruction="search iphone price on amazon", domain="amazon.com",
                                scope="browser", success=True, trajectory=[{"action_type": "navigate_browser"}], lesson="")
    found = semantic_memory.recall_sync("send hello to priyal on whatsapp")
    assert found, "a similar WhatsApp task must be recalled"
    assert found[0].task_id == "a"
    assert found[0].tools == ["open_app", "send_keys", "see_window"]
    assert found[0].lesson == "use ctrl+f"
    assert all(e.task_id != "b" for e in found), "an unrelated task stays below the similarity floor"


def test_recall_excludes_current_task(semantic_memory, level3_settings):
    semantic_memory.upsert_sync(task_id="same", instruction="open notepad and write hi", domain="local",
                                scope="local", success=True, trajectory=[], lesson="")
    assert semantic_memory.recall_sync("open notepad and write hi", exclude_task_id="same") == []


def test_feedback_reorders_recall(semantic_memory, level3_settings):
    for tid in ("x", "y"):
        semantic_memory.upsert_sync(task_id=tid, instruction="open whatsapp and message bhawesh hello",
                                    domain="local", scope="local", success=True, trajectory=WHATSAPP_OK, lesson=tid)
    semantic_memory.record_feedback_sync("x", -1)
    semantic_memory.record_feedback_sync("y", +1)
    found = semantic_memory.recall_sync("open whatsapp and message bhawesh hello")
    assert [e.task_id for e in found] == ["y", "x"]
    disputed = next(e for e in found if e.task_id == "x")
    assert disputed.user_disputed and not disputed.effective_success


def test_feedback_before_write_is_queued_then_applied(semantic_memory, level3_settings):
    assert semantic_memory.record_feedback_sync("late", +1, "nice") is None
    semantic_memory.upsert_sync(task_id="late", instruction="play lofi music on youtube", domain="youtube.com",
                                scope="browser", success=True, trajectory=[], lesson="")
    exp = semantic_memory.recall_sync("play lofi music on youtube")[0]
    assert exp.feedback == 1


def test_feedback_is_clamped(semantic_memory, level3_settings):
    semantic_memory.upsert_sync(task_id="c", instruction="open calculator", domain="local", scope="local",
                                success=False, trajectory=[], lesson="")
    for _ in range(10):
        semantic_memory.record_feedback_sync("c", -1)
    assert semantic_memory.recall_sync("open calculator")[0].feedback == -5


def test_unavailable_memory_degrades_to_empty(tmp_path, level3_settings):
    from app.state.semantic_memory import SemanticMemory

    class Broken:
        def __call__(self, input):
            raise RuntimeError("no model")

    memory = SemanticMemory(path=tmp_path / "file.txt", embedding_function=Broken())
    (tmp_path / "file.txt").write_text("not a directory")
    assert memory.recall_sync("anything") == []
    assert memory.upsert_sync(task_id="z", instruction="x", domain="", scope="", success=True,
                              trajectory=[], lesson="") is False


def test_experience_age_text():
    now = datetime.now(timezone.utc)
    assert _exp(created_at=(now - timedelta(days=3)).isoformat()).age_text(now) == "3 days ago"
    assert _exp(created_at=(now - timedelta(seconds=30)).isoformat()).age_text(now) == "just now"
    assert _exp(created_at="garbage").age_text(now) == "earlier"


# =============================================================================
# A/B strategy tracking and anti-skills
# =============================================================================

def test_summarize_strategies_ranks_by_success_rate():
    exps = [
        _exp("1", tools=["open_app", "click_window"], success=False),
        _exp("2", tools=["open_app", "click_window"], success=True),
        _exp("3", tools=["open_app", "send_keys"], success=True),
        _exp("4", tools=["open_app", "send_keys"], success=True),
        _exp("5", tools=["open_app", "send_keys"], success=True, feedback=-1, disputed=True),
    ]
    ranked = summarize_strategies(exps)
    assert ranked[0] == ("open_app -> send_keys", 2, 3)
    assert ranked[1] == ("open_app -> click_window", 1, 2)


def test_fallback_strategy_lists_approaches_and_avoid():
    text = fallback_strategy([
        _exp("1", success=True),
        _exp("2", success=False, tools=["run_command"], error="xdotool not found", instruction="type hi with xdotool"),
    ])
    assert "STRATEGY" in text and "AVOID" in text
    assert "xdotool not found" in text


def test_local_skill_priority_prefers_interaction():
    assert local_skill_type_for(["open_app", "send_keys"]) == "app_interaction"
    assert local_skill_type_for({"open_app"}) == "open_app"
    assert local_skill_type_for(["navigate_browser"]) is None


# =============================================================================
# Reflection engine
# =============================================================================

async def test_plan_strategy_uses_llm_and_caches(level3_settings):
    fake = ScriptedCompletion([llm_message("STRATEGY:\n- use Ctrl+F\nAVOID:\n- clicking blindly")])
    engine = ReflectionEngine(completion_fn=fake)
    exps = [_exp("a"), _exp("b", success=False, error="clicked wrong chat")]
    advice = await engine.plan_strategy("message priyal on whatsapp", exps)
    assert advice.from_llm and "Ctrl+F" in advice.text
    again = await engine.plan_strategy("Message Priyal on WhatsApp!", exps)
    assert again is advice and len(fake.calls) == 1, "normalized instruction hits the cache"
    prompt = fake.calls[0]["messages"][1]["content"]
    assert "worked 1/1" in prompt or "worked 1/2" in prompt or "APPROACH SUCCESS RATES" in prompt


async def test_plan_strategy_cache_invalidated_by_feedback(level3_settings):
    fake = ScriptedCompletion([llm_message("STRATEGY:\n- a"), llm_message("STRATEGY:\n- b")])
    engine = ReflectionEngine(completion_fn=fake)
    await engine.plan_strategy("task", [_exp("a")])
    await engine.plan_strategy("task", [_exp("a", feedback=-1, disputed=True)])
    assert len(fake.calls) == 2


async def test_plan_strategy_falls_back_on_llm_error(level3_settings):
    engine = ReflectionEngine(completion_fn=ScriptedCompletion([RuntimeError("groq down")]))
    advice = await engine.plan_strategy("task", [_exp("a")])
    assert advice.from_llm is False and "STRATEGY" in advice.text


async def test_plan_strategy_without_experiences_is_none(level3_settings):
    assert await ReflectionEngine(completion_fn=ScriptedCompletion([])).plan_strategy("x", []) is None


class _SlowStream:
    """Async stream yielding chunks, the second one after a long pause."""

    def __init__(self, pieces, pause_after_first):
        self.pieces = list(pieces)
        self.pause = pause_after_first
        self.sent = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        import asyncio

        if not self.pieces:
            raise StopAsyncIteration
        if self.sent == 1:
            await asyncio.sleep(self.pause)
        self.sent += 1
        piece = self.pieces.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])


async def test_streaming_strategy_returns_partial_at_deadline(level3_settings):
    stream = _SlowStream(["STRATEGY:\n- reuse ctrl+f\n", "AVOID:\n- clicks"], pause_after_first=5)
    engine = ReflectionEngine(completion_fn=ScriptedCompletion([stream]))
    started = time.monotonic()
    advice = await engine.plan_strategy("task", [_exp("a")], soft_deadline=time.monotonic() + 0.5)
    assert time.monotonic() - started < 3
    assert advice.partial and advice.from_llm
    assert "reuse ctrl+f" in advice.text and "AVOID" not in advice.text


async def test_streaming_strategy_complete(level3_settings):
    stream = _SlowStream(["STRATEGY:\n- a\n", "AVOID:\n- b"], pause_after_first=0)
    fake = ScriptedCompletion([stream])
    advice = await ReflectionEngine(completion_fn=fake).plan_strategy(
        "task", [_exp("a")], soft_deadline=time.monotonic() + 5
    )
    assert not advice.partial and "AVOID" in advice.text
    assert fake.calls[0]["stream"] is True


async def test_reflect_on_outcome_parses_json_and_compares(level3_settings):
    reply = '```json\n{"what_succeeded": "keyboard search", "what_failed": "None", "root_cause": "", "what_to_try_next": "reuse"}\n```'
    fake = ScriptedCompletion([llm_message(reply)])
    engine = ReflectionEngine(completion_fn=fake)
    result = await engine.reflect_on_outcome(
        instruction="message rakesh", trajectory=WHATSAPP_OK, success=True, error=None,
        raw_result="sent", similar=[_exp("old", success=False, error="typed Rockies")],
    )
    assert result.from_llm and result.what_succeeded == "keyboard search"
    assert "Worked: keyboard search" in result.as_lesson() and "Failed" not in result.as_lesson()
    assert "EARLIER SIMILAR ATTEMPTS THAT FAILED" in fake.calls[0]["messages"][1]["content"]


async def test_reflect_on_outcome_returns_none_on_garbage(level3_settings):
    engine = ReflectionEngine(completion_fn=ScriptedCompletion([llm_message("I think it went fine")]))
    assert await engine.reflect_on_outcome(instruction="x", trajectory=[], success=True, error=None, raw_result="") is None


def test_parse_json_object_handles_think_blocks():
    assert _parse_json_object('<think>hmm</think>{"a": 1}') == {"a": 1}
    assert _parse_json_object("no json") is None


# =============================================================================
# Brain integration: learning and feedback
# =============================================================================

async def test_learn_semantic_stores_llm_lesson(semantic_memory, level3_settings, monkeypatch):
    monkeypatch.setattr(level3_settings, "reflexion_enabled", False)
    reply = '{"what_succeeded": "ctrl+f search", "what_failed": "None", "root_cause": "None", "what_to_try_next": "same"}'
    brain = BrainMemory(semantic=semantic_memory, reflection=ReflectionEngine(ScriptedCompletion([llm_message(reply)])))
    reflection = await brain.learn_semantic(
        task_id="L1", instruction="open whatsapp and say hi to rakesh", domain="local", scope="local",
        trajectory=WHATSAPP_OK, success=True, error=None, raw_result="done",
    )
    assert reflection.from_llm
    stored = semantic_memory.recall_sync("open whatsapp and say hi to rakesh")[0]
    assert "ctrl+f search" in stored.lesson


async def test_learn_semantic_falls_back_to_template(semantic_memory, level3_settings, monkeypatch):
    monkeypatch.setattr(level3_settings, "reflexion_enabled", False)
    brain = BrainMemory(semantic=semantic_memory, reflection=ReflectionEngine(ScriptedCompletion([RuntimeError("down")])))
    reflection = await brain.learn_semantic(
        task_id="L2", instruction="find my resume pdf", domain="local", scope="local",
        trajectory=[{"action_type": "find_files", "success": False}], success=False,
        error="No files matching resume", raw_result="",
    )
    assert reflection is not None and reflection.from_llm is False
    assert semantic_memory.recall_sync("find my resume pdf")[0].success is False


async def test_record_feedback_thumbs_down_penalizes_skills(semantic_memory, level3_settings, monkeypatch):
    brain = BrainMemory(semantic=semantic_memory, reflection=ReflectionEngine(ScriptedCompletion([])))
    semantic_memory.upsert_sync(task_id="F1", instruction="message bhawesh on whatsapp", domain="local",
                                scope="local", success=True, trajectory=WHATSAPP_OK, lesson="")
    penalized = []

    async def fake_penalize(domain, tools):
        penalized.append((domain, tools))
        return 1

    monkeypatch.setattr(brain, "_penalize_skills_for_tools", fake_penalize)
    result = await brain.record_feedback("F1", -1)
    assert result["applied"] and result["skills_penalized"] == 1
    assert penalized == [("local", ["open_app", "send_keys", "see_window"])]

    up = await brain.record_feedback("F1", +1)
    assert up["applied"] and up["skills_penalized"] == 0


async def test_record_feedback_queued_when_not_stored_yet(semantic_memory, level3_settings):
    brain = BrainMemory(semantic=semantic_memory, reflection=ReflectionEngine(ScriptedCompletion([])))
    result = await brain.record_feedback("not-yet", +1)
    assert result["queued"] and not result["applied"]


async def test_semantic_disabled_returns_nothing(semantic_memory, level3_settings, monkeypatch):
    monkeypatch.setattr(level3_settings, "semantic_memory_enabled", False)
    brain = BrainMemory(semantic=semantic_memory)
    assert await brain.recall_semantic("anything") == []
    assert (await brain.record_feedback("t", 1))["applied"] is False


# =============================================================================
# Explainability
# =============================================================================

def test_explanation_with_memory_and_retry():
    text = build_explanation(
        trajectory=[{"action_type": "open_app", "input_value": "WhatsApp", "success": True},
                    {"action_type": "send_keys", "input_value": "{}", "success": True}],
        success=True,
        experiences=[_exp("a", instruction="message bhawesh on whatsapp", feedback=1, similarity=0.72,
                          created_at=(datetime.now(timezone.utc) - timedelta(days=2)).isoformat())],
        strategy="STRATEGY:\n- use Ctrl+F\n- clear the box\nAVOID:\n- clicking",
        strategy_from_llm=True,
        retried=True,
        first_attempt_error="clicked the wrong chat",
        retry_blocked_reason="",
        error="",
    )
    assert "✓ open_app (WhatsApp)" in text and "✓ send_keys" in text
    assert "1 similar past task" in text and "72% similar" in text and "2 days ago" in text and "👍" in text
    assert "use Ctrl+F; clear the box" in text
    assert "retried once" in text and "clicked the wrong chat" in text


def test_explanation_without_memory_and_blocked_retry():
    text = build_explanation(
        trajectory=[], success=False, experiences=[], strategy="", strategy_from_llm=False,
        retried=False, first_attempt_error="", retry_blocked_reason="it already sent a message",
        error="window not found",
    )
    assert "planned from scratch" in text
    assert "did not retry automatically because it already sent a message" in text
    assert "window not found" in text


def test_strategy_advice_and_reflection_dataclasses():
    assert StrategyAdvice("x", from_llm=False).partial is False
    assert OutcomeReflection("None", "None", "None", "None").as_lesson() == ""
