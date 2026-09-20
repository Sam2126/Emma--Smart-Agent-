"""
Regression tests for the second audit against tech_audit.md and the
2026-09-15 WhatsApp run log.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agent.context import cancel_active_runs, track_run, untrack_run
from app.agent.explain import build_explanation
from app.agent.toolkit import (
    BROWSER_TOOL_NAMES,
    USE_BROWSER_TOOL,
    build_toolset,
    initial_tool_schemas,
    instruction_needs_web,
)
from app.config import Settings, get_settings
from app.state.reflection import ReflectionEngine, fallback_strategy
from app.state.semantic_memory import Experience, anti_skills
from app.utils import litellm_patch, llm
from app.utils.key_pool import KeyPool
from tests.conftest import ScriptedCompletion, llm_message


# =============================================================================
# The system prompt reaches the model whole
# =============================================================================

def test_system_prompt_and_task_message_are_never_truncated():
    from app.agent.prompts import actor_system_prompt, actor_user_prompt

    system = actor_system_prompt("local")
    task = actor_user_prompt("open WhatsApp and search Rakesh and say hello to him", "local", "1. open_app\n" * 300)
    history = [{"role": "tool", "tool_call_id": str(i), "content": "r" * 2000} for i in range(10)]
    kwargs = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": task}] + history}

    litellm_patch._prepare(kwargs)
    out = kwargs["messages"]

    assert len(system) > 3000 and out[0]["content"] == system
    assert len(task) > 3000 and out[1]["content"] == task
    assert "history compressed" in out[2]["content"]
    assert out[-1]["content"] == "r" * 2000 and out[-2]["content"] == "r" * 2000


def test_cache_headers_are_still_stripped_everywhere():
    kwargs = {"messages": [
        {"role": "system", "content": "s", "cache_control": {"type": "ephemeral"}},
        {"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {}}]},
    ]}
    litellm_patch._prepare(kwargs)
    assert "cache_control" not in kwargs["messages"][0]
    assert "cache_control" not in kwargs["messages"][1]["content"][0]


def test_litellm_console_noise_is_suppressed():
    import litellm

    assert litellm.suppress_debug_info is True
    assert logging.getLogger("LiteLLM").getEffectiveLevel() >= logging.WARNING


# =============================================================================
# Rate limits are per model
# =============================================================================

def test_key_pool_cooldown_is_per_model():
    pool = KeyPool(["gsk_aaaaaa1", "gsk_bbbbbb2"])
    fast, big = "groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b"
    for key in ("gsk_aaaaaa1", "gsk_bbbbbb2"):
        pool.mark_rate_limited(key, cooldown_seconds=30, model=fast)
    assert pool.min_remaining_cooldown(model=fast) > 25
    assert pool.min_remaining_cooldown(model=big) == 0.0
    assert pool.acquire(model=big) in ("gsk_aaaaaa1", "gsk_bbbbbb2")
    pool.mark_success("gsk_aaaaaa1", model=fast)
    assert pool.min_remaining_cooldown(model=fast) == 0.0
    assert pool.status()[1]["cooling_models"].get(fast, 0) > 25


def test_key_wide_cooldown_still_blocks_every_model():
    pool = KeyPool(["gsk_only01"])
    pool.mark_rate_limited("gsk_only01", cooldown_seconds=30)
    assert pool.min_remaining_cooldown(model="groq/anything") > 25
    assert pool.min_remaining_cooldown() > 25


def test_role_is_rate_limited_reads_the_pool(monkeypatch):
    pool = KeyPool(["gsk_only01"])
    monkeypatch.setattr("app.utils.key_pool.get_key_pool", lambda: pool)
    monkeypatch.setattr(llm, "_rl_adapter_model", lambda: None)
    model = llm.model_for("actor_fast")[0]
    assert llm.role_is_rate_limited("actor_fast") is False
    pool.mark_rate_limited("gsk_only01", cooldown_seconds=30, model=model)
    assert llm.role_is_rate_limited("actor_fast") is True


# =============================================================================
# RL adapter inference hook (kept as Level 4 infrastructure)
# =============================================================================

def test_rl_adapter_hook_routes_policy_roles(monkeypatch, tmp_path):
    import app.rl.registry as registry

    monkeypatch.setattr(get_settings(), "llm_provider", "groq")
    llm._reset_for_tests()
    monkeypatch.setattr(registry, "get_current_production_adapter", lambda *a, **k: "ollama/agent-sft-v1")
    assert llm.model_for("actor")[0] == "ollama/agent-sft-v1"
    assert llm.model_for("planner")[0] == "ollama/agent-sft-v1"
    assert llm.model_for("judge")[0].startswith("groq/")

    llm._reset_for_tests()
    monkeypatch.setattr(registry, "get_current_production_adapter", lambda *a, **k: str(tmp_path))
    assert llm.model_for("actor")[0].startswith("groq/"), "a LoRA folder on disk is not servable"
    llm._reset_for_tests()


# =============================================================================
# Browser tools on demand for local tasks
# =============================================================================

def test_local_non_web_task_defers_browser_tools():
    schemas, deferred = initial_tool_schemas(build_toolset("local"), "local", "open WhatsApp and message Rakesh hello")
    names = {s["function"]["name"] for s in schemas}
    assert deferred and USE_BROWSER_TOOL in names
    assert not names & BROWSER_TOOL_NAMES
    assert {"open_app", "send_keys", "see_window", "click_window"} <= names


@pytest.mark.parametrize("instruction", [
    "open youtube.com and play lofi",
    "search the web for pizza near me",
    "log in to gmail and open the latest invoice",
])
def test_web_tasks_get_browser_tools_up_front(instruction):
    assert instruction_needs_web(instruction)
    schemas, deferred = initial_tool_schemas(build_toolset("local"), "local", instruction)
    assert not deferred and "navigate_browser" in {s["function"]["name"] for s in schemas}


def test_browser_scope_is_unchanged():
    schemas, deferred = initial_tool_schemas(build_toolset("browser"), "browser", "open my notes")
    names = {s["function"]["name"] for s in schemas}
    assert not deferred and USE_BROWSER_TOOL not in names and names == BROWSER_TOOL_NAMES


def test_browser_tool_names_match_the_registry():
    assert BROWSER_TOOL_NAMES == set(build_toolset("browser"))


# =============================================================================
# Anti-skills and trajectory comparison
# =============================================================================

def _exp(tid, tools, success):
    return Experience(
        task_id=tid, instruction="message rakesh on whatsapp", domain="local", scope="local", success=success,
        tools=tools, lesson="", error="" if success else "clicked the wrong chat", feedback=0,
        user_disputed=False, created_at="2026-09-10T00:00:00+00:00", similarity=0.8,
    )


EXPS = [
    _exp("1", ["open_app", "click_window"], False),
    _exp("2", ["open_app", "click_window"], False),
    _exp("3", ["open_app", "send_keys"], True),
    _exp("4", ["open_app", "send_keys"], True),
    _exp("5", ["run_command"], False),
]


def test_anti_skills_are_approaches_that_keep_failing():
    assert anti_skills(EXPS) == [("open_app -> click_window", 0, 2)]
    text = fallback_strategy(EXPS)
    assert "ANTI-SKILLS" in text and "open_app -> click_window" in text


async def test_strategy_prompt_lists_anti_skills(monkeypatch):
    monkeypatch.setattr(get_settings(), "llm_reflection_enabled", True)
    fake = ScriptedCompletion([llm_message("STRATEGY:\n- use keyboard search")])
    await ReflectionEngine(completion_fn=fake).plan_strategy("message priyal on whatsapp", EXPS)
    assert "ANTI-SKILLS" in fake.calls[0]["messages"][1]["content"]


def test_explanation_names_the_avoided_anti_skill():
    text = build_explanation(
        trajectory=[{"action_type": "open_app", "success": True}, {"action_type": "send_keys", "success": True}],
        success=True, experiences=EXPS, strategy="", strategy_from_llm=False, retried=False,
        first_attempt_error="", retry_blocked_reason="", error="",
    )
    assert "Avoided: open_app -> click_window" in text


async def test_retry_reflection_diffs_the_failed_first_attempt(monkeypatch):
    monkeypatch.setattr(get_settings(), "llm_reflection_enabled", True)
    reply = ('{"what_succeeded": "closed the popup before searching", "what_failed": "None", '
             '"root_cause": "a popup covered the search box", "what_to_try_next": "dismiss popups first"}')
    fake = ScriptedCompletion([llm_message(reply)])
    result = await ReflectionEngine(completion_fn=fake).reflect_on_outcome(
        instruction="search shoes on myntra",
        trajectory=[{"action_type": "click_element", "input_value": "#close-popup", "success": True}],
        success=True, error=None, raw_result="done",
        previous_attempt={
            "trajectory": [{"action_type": "type_into_element", "input_value": "#search", "success": False,
                            "error": "element covered by popup"}],
            "error": "blocked by popup",
        },
    )
    prompt = fake.calls[0]["messages"][1]["content"]
    assert "FIRST ATTEMPT FAILED; THIS RETRY SUCCEEDED" in prompt
    assert "blocked by popup" in prompt and "type_into_element" in prompt and "#close-popup" in prompt
    assert result.what_to_try_next == "dismiss popups first"


# =============================================================================
# Shutdown cancels running tasks
# =============================================================================

async def test_cancel_active_runs_stops_running_tasks():
    started = asyncio.Event()

    async def fake_run():
        me = track_run()
        try:
            started.set()
            await asyncio.sleep(60)
        finally:
            untrack_run(me)

    task = asyncio.create_task(fake_run())
    await started.wait()
    assert await cancel_active_runs(timeout=2) == 1
    assert task.cancelled()
    assert await cancel_active_runs(timeout=1) == 0


# =============================================================================
# Vosk is the wake word engine
# =============================================================================

def test_vosk_is_the_default_wake_word_engine():
    assert Settings.model_fields["wake_word_engine"].default == "vosk"


def test_unknown_word_placeholders_keep_wake_word_position():
    from app.wake_listener import WakeWordListener

    listener = WakeWordListener(wake_word="hello", stop_word="done")
    assert listener._matches_wake_word("[unk] hello [unk]") is False
    assert listener._matches_wake_word("hello [unk]") is True
    assert listener._matches_wake_word("hello") is True
    assert listener._matches_stop_word("[unk] hello [unk] done") is True
    assert listener._matches_stop_word("[unk] hello [unk]") is False


def test_vosk_engine_loads_the_real_model_and_grammar(monkeypatch):
    pytest.importorskip("vosk")
    from app.wake_listener import WakeWordListener

    model_path = Path(__file__).resolve().parents[1] / "data" / "vosk-model-small-en-us-0.15"
    if not model_path.exists():
        pytest.skip("Vosk model not installed")
    monkeypatch.setattr(get_settings(), "wake_word_engine", "vosk")
    monkeypatch.setattr(get_settings(), "vosk_model_path", str(model_path))

    listener = WakeWordListener(wake_word="hello", stop_word="done")
    listener._load_vosk_if_configured()
    assert listener._vosk_model is not None
    assert "hello" in listener._vosk_grammar and "done" in listener._vosk_grammar

    silence = SimpleNamespace(get_raw_data=lambda convert_rate, convert_width: b"\x00\x00" * 16000)
    assert listener._recognize_vosk(silence) == ""
