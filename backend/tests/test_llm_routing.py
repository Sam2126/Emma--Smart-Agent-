"""
Model routing, the two-tier actor, the Gemini fallback in the LiteLLM patch,
the routed LLM judge, and the vision provider order.
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.utils import litellm_patch, llm
from app.utils.vision import provider_order
from app.verifier.llm_judge import LLMJudge
from tests.conftest import ScriptedCompletion, llm_message


@pytest.fixture(autouse=True)
def clean_router(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "llm_provider", "groq")
    monkeypatch.setattr(s, "prefer_llama_models", True)
    monkeypatch.setattr(s, "two_tier_actor", True)
    monkeypatch.setattr(s, "groq_planning_model", "openai/gpt-oss-120b")
    monkeypatch.setattr(s, "groq_action_model", "openai/gpt-oss-120b")
    monkeypatch.setattr(s, "groq_fast_action_model", "llama-3.1-8b-instant")
    monkeypatch.setattr(s, "reflection_model", "")
    monkeypatch.setattr(llm, "_rl_adapter_model", lambda: None)
    llm._reset_for_tests()
    yield s
    llm._reset_for_tests()


# =============================================================================
# Router
# =============================================================================

def test_unknown_availability_never_uses_unvalidated_models(clean_router):
    assert llm.resolve_groq_model("planner") == "openai/gpt-oss-120b"
    assert llm.resolve_groq_model("actor_fast") == "openai/gpt-oss-120b"


def test_account_without_llama_falls_through_to_gpt_oss(clean_router):
    llm._set_available_for_tests({"openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"})
    assert llm.resolve_groq_model("planner") == "openai/gpt-oss-120b"
    assert llm.resolve_groq_model("actor") == "openai/gpt-oss-120b"
    assert llm.resolve_groq_model("actor_fast") == "openai/gpt-oss-20b"


def test_llama_preferred_when_offered(clean_router):
    llm._set_available_for_tests({"llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-120b"})
    assert llm.resolve_groq_model("actor") == "llama-3.3-70b-versatile"
    assert llm.resolve_groq_model("actor_fast") == "llama-3.1-8b-instant"


def test_explicit_reflection_model_wins(clean_router, monkeypatch):
    monkeypatch.setattr(clean_router, "reflection_model", "openai/gpt-oss-20b")
    llm._set_available_for_tests({"llama-3.3-70b-versatile", "openai/gpt-oss-20b", "openai/gpt-oss-120b"})
    assert llm.resolve_groq_model("reflection") == "openai/gpt-oss-20b"


def test_openai_provider_routes_to_gpt_4o_mini(clean_router, monkeypatch):
    monkeypatch.setattr(clean_router, "llm_provider", "openai")
    monkeypatch.setattr(clean_router, "openai_api_key", "sk-test")
    monkeypatch.setattr(clean_router, "openai_model", "gpt-4o-mini")
    model, extra = llm.model_for("actor")
    assert model == "openai/gpt-4o-mini" and extra == {"api_key": "sk-test"}


def test_openai_provider_without_key_stays_on_groq(clean_router, monkeypatch):
    monkeypatch.setattr(clean_router, "llm_provider", "openai")
    monkeypatch.setattr(clean_router, "openai_api_key", "")
    assert llm.model_for("actor")[0].startswith("groq/")


def test_unknown_role_raises():
    with pytest.raises(ValueError):
        llm.candidates_for("poet", availability_known=True)


@pytest.mark.parametrize(
    "turn,failed,expected",
    [(0, False, "actor"), (1, False, "actor_fast"), (2, True, "actor"), (5, False, "actor"), (6, False, "actor_fast")],
)
def test_two_tier_actor_turns(turn, failed, expected):
    assert llm.actor_role_for_turn(turn, failed) == expected


def test_two_tier_disabled(clean_router, monkeypatch):
    monkeypatch.setattr(clean_router, "two_tier_actor", False)
    assert llm.actor_role_for_turn(3, False) == "actor"


async def test_chat_builds_tool_request():
    fake = ScriptedCompletion([llm_message("hi")])
    tools = [{"type": "function", "function": {"name": "x", "parameters": {}}}]
    await llm.chat("actor", [{"role": "user", "content": "go"}], tools=tools, completion_fn=fake, max_tokens=99)
    call = fake.calls[0]
    assert call["model"] == "groq/openai/gpt-oss-120b"
    assert call["tool_choice"] == "auto" and call["parallel_tool_calls"] is False
    assert call["max_tokens"] == 99 and call["reasoning_effort"] == "low"


async def test_chat_without_tools_sends_no_tool_keys():
    fake = ScriptedCompletion([llm_message("hi")])
    await llm.chat("planner", [{"role": "user", "content": "go"}], completion_fn=fake)
    assert "tools" not in fake.calls[0] and "tool_choice" not in fake.calls[0]


# =============================================================================
# LiteLLM patch: Groq key pool and Gemini fallback
# =============================================================================

class _FakePool:
    size = 1

    def __init__(self):
        self.limited = 0

    def acquire(self, model=None):
        return "groq-key"

    def mark_success(self, key, model=None):
        pass

    def mark_rate_limited(self, key, cooldown_seconds=None, model=None):
        self.limited += 1

    def min_remaining_cooldown(self, model=None):
        return 0.0


@pytest.fixture
def patched_llm(monkeypatch, clean_router):
    calls = []
    pool = _FakePool()
    monkeypatch.setattr("app.utils.key_pool.get_key_pool", lambda: pool)
    monkeypatch.setattr(litellm_patch, "_MIN_CALL_INTERVAL", 0.0)
    monkeypatch.setattr(clean_router, "gemini_api_key", "gemini-test-key")
    monkeypatch.setattr(clean_router, "llm_fallback_model", "gemini/gemini-3.6-flash")

    def install(groq_error: Exception | None):
        async def fake_acompletion(*args, **kwargs):
            calls.append(dict(kwargs))
            if kwargs["model"].startswith("groq/") and groq_error is not None:
                raise groq_error
            return llm_message(f"answer from {kwargs['model']}")

        monkeypatch.setattr(litellm_patch, "_orig_acompletion", fake_acompletion)
        return calls

    return install, pool


async def test_groq_outage_falls_back_to_gemini(patched_llm):
    install, _ = patched_llm
    calls = install(RuntimeError("503 Service Unavailable"))
    result = await litellm_patch._sanitized_acompletion(
        model="groq/openai/gpt-oss-120b", messages=[{"role": "user", "content": "hi"}],
        max_tokens=100, parallel_tool_calls=False,
    )
    assert result.choices[0].message.content == "answer from gemini/gemini-3.6-flash"
    fallback = calls[-1]
    assert fallback["api_key"] == "gemini-test-key"
    assert fallback["max_tokens"] >= 4096
    assert "parallel_tool_calls" not in fallback


async def test_rate_limited_on_every_key_falls_back(patched_llm):
    install, pool = patched_llm
    calls = install(RuntimeError("429 rate_limit_exceeded"))
    await litellm_patch._sanitized_acompletion(model="groq/openai/gpt-oss-20b", messages=[{"role": "user", "content": "x"}])
    assert pool.limited == 2
    assert calls[-1]["model"] == "gemini/gemini-3.6-flash"


async def test_a_request_too_large_for_groq_goes_to_gemini_without_cooling_keys(patched_llm):
    """Found 2026-09-17: 'Request too large' was handled as a rate limit and cooled every key."""
    install, pool = patched_llm
    calls = install(RuntimeError(
        'litellm.RateLimitError: GroqException - {"error":{"message":"Request too large for model '
        '`openai/gpt-oss-120b` in organization `org_x` service tier `on_demand` on tokens per minute (TPM): '
        'Limit 8000, Requested 8573, please reduce your message size and try again."}}'
    ))
    result = await litellm_patch._sanitized_acompletion(model="groq/openai/gpt-oss-120b", messages=[{"role": "user", "content": "x"}])
    assert result.choices[0].message.content == "answer from gemini/gemini-3.6-flash"
    assert pool.limited == 0
    assert [c["model"] for c in calls] == ["groq/openai/gpt-oss-120b", "gemini/gemini-3.6-flash"]


async def test_non_transient_groq_error_is_raised(patched_llm):
    install, _ = patched_llm
    calls = install(ValueError("400 invalid request: bad schema"))
    with pytest.raises(ValueError):
        await litellm_patch._sanitized_acompletion(model="groq/openai/gpt-oss-120b", messages=[{"role": "user", "content": "x"}])
    assert all(c["model"].startswith("groq/") for c in calls)


async def test_non_groq_requests_keep_their_own_key(patched_llm):
    install, _ = patched_llm
    calls = install(None)
    await litellm_patch._sanitized_acompletion(model="gemini/gemini-3.6-flash", api_key="mine", messages=[{"role": "user", "content": "x"}])
    assert calls[0]["api_key"] == "mine"


def test_no_gemini_key_means_no_fallback(clean_router, monkeypatch):
    monkeypatch.setattr(clean_router, "gemini_api_key", "")
    assert litellm_patch._fallback_kwargs({"model": "groq/x"}) is None


def test_relaxed_schema_keeps_only_arguments_without_defaults():
    kwargs = {"tools": [{"type": "function", "function": {"name": "send_keys", "strict": True, "parameters": {
        "properties": {"window_hint": {"type": "string"}, "text": {"type": "string", "default": ""}},
        "required": ["window_hint", "text"],
    }}}]}
    litellm_patch._relax_tool_schemas(kwargs)
    fn = kwargs["tools"][0]["function"]
    assert fn["parameters"]["required"] == ["window_hint"] and "strict" not in fn


# =============================================================================
# Judge (routed through LiteLLM, not ChatGroq) and vision order
# =============================================================================

def test_parse_judgment_tolerates_markdown_and_percent():
    j = LLMJudge._parse_judgment("**VERDICT:** PASS\n**CONFIDENCE:** 85%\nREASONING: results shown")
    assert j.passed and j.confidence == pytest.approx(0.85) and j.reasoning == "results shown"
    f = LLMJudge._parse_judgment("VERDICT: FAIL\nCONFIDENCE: high")
    assert not f.passed and f.confidence == 0.5


async def test_text_judge_uses_judge_role_through_litellm():
    from app.browser.page_state import PageState

    fake = ScriptedCompletion([llm_message("VERDICT: PASS\nCONFIDENCE: 0.9\nREASONING: ok")])
    judgment = await LLMJudge(completion_fn=fake).verify_task(
        "search shoes", PageState(url="https://x.com/s?q=shoes", title="shoes"), "1. navigate"
    )
    assert judgment.passed and judgment.source == "text"
    assert fake.calls[0]["model"].startswith("groq/")


async def test_text_judge_failure_is_conservative():
    from app.browser.page_state import PageState

    judgment = await LLMJudge(completion_fn=ScriptedCompletion([RuntimeError("down")])).verify_task(
        "x", PageState(url="https://x.com"), ""
    )
    assert not judgment.passed and judgment.confidence == 0.0


def test_vision_provider_order(clean_router, monkeypatch):
    monkeypatch.setattr(clean_router, "gemini_api_key", "k")
    monkeypatch.setattr(clean_router, "vision_provider", "auto")
    assert provider_order() == ["gemini", "groq"]
    monkeypatch.setattr(clean_router, "vision_provider", "groq")
    assert provider_order() == ["groq", "gemini"]
    monkeypatch.setattr(clean_router, "gemini_api_key", "")
    monkeypatch.setattr(clean_router, "vision_provider", "auto")
    assert provider_order() == ["groq"]
