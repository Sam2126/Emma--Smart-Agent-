"""
Model routing for every LLM call the agent makes.

One place decides which model serves which role, so the planner, the actor,
the reflection engine and the verification judge all follow the same rules:

  planner, reflection, judge -> the planning model
  actor                      -> the action model
  actor_fast                 -> a smaller, faster model for routine steps

Two-tier actor: the fast model runs routine steps while they keep succeeding;
the action model runs the first step, any step right after a failure, and
every fifth turn. See `actor_role_for_turn`.

Groq is the primary provider. Models are validated against the account's real
/models list at startup because what an account can use changes. Verified on
this project's account (2026-09-15): no Llama chat models are offered — only
openai/gpt-oss-120b, openai/gpt-oss-20b, qwen/qwen3.8-27b and a few others. So
the Llama preference (prefer_llama_models) currently falls through to the
configured gpt-oss models, and the fast tier resolves to gpt-oss-20b. Every
choice is logged at startup.

LLM_PROVIDER=openai with OPENAI_API_KEY routes the text roles to OPENAI_MODEL
(gpt-4o-mini by default). Gemini is the automatic fallback when every Groq key
is rate-limited or Groq is down — see utils/litellm_patch.py.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx
import structlog

from app.config import get_settings

logger = structlog.get_logger(__name__)

ROLES = ("planner", "actor", "actor_fast", "reflection", "judge")
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"

_lock = threading.Lock()
_groq_available: set[str] | None = None
_resolved: dict[str, str] = {}
_report: dict[str, Any] = {}


def _bare(model: str) -> str:
    return model[len("groq/"):] if model.startswith("groq/") else model


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = _bare(item or "").strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def candidates_for(role: str, *, availability_known: bool) -> list[str]:
    """Ordered model candidates for a role.

    Preference-only candidates (the Llama models) are included only when the
    account's model list is known, so an unvalidated preference can never
    cause a 404 before the startup check has run.
    """
    s = get_settings()
    prefer_llama = s.prefer_llama_models and availability_known
    if role in ("planner", "reflection", "judge"):
        configured = s.reflection_model if (role == "reflection" and s.reflection_model) else s.groq_planning_model
        pref = ["llama-3.3-70b-versatile"] if prefer_llama and not (role == "reflection" and s.reflection_model) else []
        return _dedupe(pref + [configured, "openai/gpt-oss-120b"])
    if role == "actor":
        pref = ["llama-3.3-70b-versatile"] if prefer_llama else []
        return _dedupe(pref + [s.groq_action_model, "openai/gpt-oss-120b"])
    if role == "actor_fast":
        if not availability_known:
            # Can't validate the fast model yet: stay on the action model.
            return _dedupe([s.groq_action_model])
        pref = ["llama-3.1-8b-instant"] if s.prefer_llama_models else []
        return _dedupe(pref + [s.groq_fast_action_model, "openai/gpt-oss-20b", s.groq_action_model])
    raise ValueError(f"Unknown LLM role: {role}")


def resolve_groq_model(role: str) -> str:
    with _lock:
        if role in _resolved:
            return _resolved[role]
        available = _groq_available
    cands = candidates_for(role, availability_known=available is not None)
    if available is None:
        return cands[0]
    chosen = next((c for c in cands if c in available), cands[-1])
    with _lock:
        _resolved[role] = chosen
    return chosen


# --- RL adapter inference hook (Level 4 infrastructure) ---------------------
# The former CrewAI agents swapped the registry's accepted adapter in for the
# base model (app/rl/registry.get_current_production_adapter). The engine keeps
# that hook for the policy roles. An adapter is used only when it is registered
# as a LiteLLM model string with a provider prefix (e.g. "ollama/agent-sft-v1"):
# a bare LoRA folder on disk cannot be served by any configured provider, so it
# is logged once and the base model is kept.
_RL_ROLES = ("planner", "actor", "actor_fast")
_RL_CHECK_SECONDS = 60.0
_SERVABLE_PREFIXES = {
    "ollama", "ollama_chat", "hosted_vllm", "openai", "huggingface", "together_ai", "fireworks_ai", "groq",
}
_rl_cache: tuple[float, str | None] = (0.0, None)
_rl_warned: set[str] = set()


def _rl_adapter_model() -> str | None:
    global _rl_cache
    now = time.monotonic()
    if _rl_cache[0] and now - _rl_cache[0] < _RL_CHECK_SECONDS:
        return _rl_cache[1]
    model = None
    try:
        from app.rl.registry import get_current_production_adapter

        adapter = get_current_production_adapter()
        if adapter:
            prefix = adapter.split("/", 1)[0].lower()
            if "/" in adapter and prefix in _SERVABLE_PREFIXES and not Path(adapter).exists():
                model = adapter
            elif adapter not in _rl_warned:
                _rl_warned.add(adapter)
                logger.warning(
                    "rl_adapter_not_servable",
                    adapter=adapter,
                    hint="register the adapter as a LiteLLM model string, e.g. ollama/<name>",
                )
    except Exception as e:
        logger.debug("rl_registry_unavailable", error=str(e)[:120])
    _rl_cache = (now, model)
    return model


def model_for(role: str) -> tuple[str, dict[str, Any]]:
    """LiteLLM model string plus extra kwargs (api_key for non-Groq providers)."""
    if role in _RL_ROLES:
        adapter = _rl_adapter_model()
        if adapter:
            return adapter, {}
    s = get_settings()
    if s.llm_provider.strip().lower() == "openai" and s.openai_api_key:
        return f"openai/{s.openai_model}", {"api_key": s.openai_api_key}
    return f"groq/{resolve_groq_model(role)}", {}


# Deciding what to do after looking at the screen is the hardest kind of turn:
# in production the fast model misread a see_window result right after a
# message was sent and typed the message again.
PERCEPTION_TOOLS = frozenset({"see_window", "see_page", "perceive_page"})


def actor_role_for_turn(turn: int, last_step_failed: bool, last_tool: str | None = None) -> str:
    """Two-tier actor: pick "actor" or "actor_fast" for this turn."""
    if not get_settings().two_tier_actor:
        return "actor"
    if turn == 0 or last_step_failed or turn % 5 == 0 or last_tool in PERCEPTION_TOOLS:
        return "actor"
    return "actor_fast"


# Groq vision models, best first. Llama 4 Scout is the tech audit's choice
# "if Groq re-enables it"; it is used automatically when the account's model
# list offers it, otherwise the configured groq_vision_model (qwen) is used.
GROQ_VISION_PREFERENCE = (
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
)


def resolve_groq_vision_model() -> str:
    with _lock:
        available = _groq_available
    if available:
        for model in GROQ_VISION_PREFERENCE:
            if model in available:
                return model
    return get_settings().groq_vision_model or "qwen/qwen3.8-27b"


def gemini_models(setting_value: str) -> list[str]:
    """A comma-separated Gemini model setting as a list."""
    return [m.strip() for m in (setting_value or "").split(",") if m.strip()]


# --- Gemini circuit breaker ---------------------------------------------------
# Found 2026-09-15: gemini-3.6-flash answered "This model is currently
# experiencing high demand" (503) and hung other requests until a 90 s timeout,
# so a task's result check waited ~100 s and every later screenshot tried the
# overloaded model first again. A model that fails that way is skipped for a
# while and the next free model is used straight away.
_gemini_cooldown: dict[str, float] = {}
_GEMINI_QUOTA_MARKERS = ("429", "quota", "resource_exhausted", "rate limit", "ratelimit")
_GEMINI_OVERLOAD_MARKERS = ("503", "unavailable", "overloaded", "high demand", "timeout", "timed out", "deadline")


def gemini_model_available(model: str) -> bool:
    with _lock:
        return time.monotonic() >= _gemini_cooldown.get(model, 0.0)


def mark_gemini_unavailable(model: str, error: BaseException | str) -> float:
    """Skip a model after an overload, rate limit or timeout. Returns the cooldown seconds (0 = not transient)."""
    text = f"{type(error).__name__ if isinstance(error, BaseException) else ''} {error}".lower()
    if any(marker in text for marker in _GEMINI_QUOTA_MARKERS):
        seconds = 300.0
    elif any(marker in text for marker in _GEMINI_OVERLOAD_MARKERS):
        seconds = 120.0
    else:
        return 0.0
    with _lock:
        _gemini_cooldown[model] = time.monotonic() + seconds
    logger.warning("gemini_model_cooling", model=model, seconds=seconds)
    return seconds


def order_by_availability(models: list[str]) -> list[str]:
    """Available models first (configured order kept), cooling ones last."""
    return sorted(models, key=lambda model: not gemini_model_available(model))


def role_is_rate_limited(role: str) -> bool:
    """True when every Groq key is cooling for the model that serves `role`."""
    model, _ = model_for(role)
    if not model.startswith("groq/"):
        return False
    try:
        from app.utils.key_pool import get_key_pool

        return get_key_pool().min_remaining_cooldown(model=model) > 0
    except Exception:
        return False


async def chat(
    role: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    reasoning_effort: str | None = "low",
    timeout: float | None = None,
    completion_fn: Callable[..., Awaitable[Any]] | None = None,
    **extra: Any,
) -> Any:
    """One routed chat completion. Returns the raw LiteLLM response."""
    model, provider_kwargs = model_for(role)
    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **provider_kwargs,
        **extra,
    }
    if reasoning_effort:
        params["reasoning_effort"] = reasoning_effort
    if tools:
        params["tools"] = tools
        params["tool_choice"] = tool_choice or "auto"
        params["parallel_tool_calls"] = False
    fn = completion_fn
    if fn is None:
        import litellm

        fn = litellm.acompletion
    coro = fn(**params)
    return await (asyncio.wait_for(coro, timeout) if timeout else coro)


async def refresh_model_availability() -> dict[str, Any]:
    """Validate configured models against the providers and log the routing."""
    global _groq_available
    s = get_settings()
    report: dict[str, Any] = {"provider": s.llm_provider}

    try:
        from app.utils.key_pool import get_key_pool

        key = get_key_pool().acquire()
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(GROQ_MODELS_URL, headers={"Authorization": f"Bearer {key}"})
        if r.status_code == 200:
            ids = {m["id"] for m in r.json().get("data", [])}
            with _lock:
                _groq_available = ids
                _resolved.clear()
            report["groq_models_available"] = sorted(ids)
        else:
            report["groq_models_error"] = f"HTTP {r.status_code}"
    except Exception as e:
        report["groq_models_error"] = str(e)[:200]

    routing: dict[str, str] = {}
    unavailable_preferences: dict[str, list[str]] = {}
    for role in ROLES:
        routing[role] = model_for(role)[0]
        if _groq_available is not None:
            missing = [c for c in candidates_for(role, availability_known=True) if c not in _groq_available]
            if missing:
                unavailable_preferences[role] = missing
    report["routing"] = routing
    report["groq_vision_model"] = resolve_groq_vision_model()
    if unavailable_preferences:
        report["not_offered_to_this_account"] = unavailable_preferences

    if s.llm_provider.strip().lower() == "openai" and not s.openai_api_key:
        report["openai_warning"] = "LLM_PROVIDER=openai but OPENAI_API_KEY is empty; using Groq"

    if s.gemini_api_key:
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                r = await client.get(GEMINI_MODELS_URL, headers={"x-goog-api-key": s.gemini_api_key})
            if r.status_code == 200:
                names = {m["name"].split("/", 1)[1] for m in r.json().get("models", [])}
                for label, setting in (("fallback", s.llm_fallback_model), ("vision", s.gemini_vision_model)):
                    offered = {model: model.split("/", 1)[-1] in names for model in gemini_models(setting)}
                    report[f"gemini_{label}_models"] = offered
                    report[f"gemini_{label}_offered"] = any(offered.values())
            else:
                report["gemini_error"] = f"HTTP {r.status_code}"
        except Exception as e:
            report["gemini_error"] = str(e)[:200]
    else:
        report["gemini"] = "no GEMINI_API_KEY: no fallback provider, vision uses Groq"

    with _lock:
        _report.clear()
        _report.update(report)
    logger.info(
        "model_routing",
        routing=routing,
        not_offered=unavailable_preferences or None,
        gemini_fallback=report.get("gemini_fallback_offered"),
        gemini_vision=report.get("gemini_vision_offered"),
    )
    return report


def routing_report() -> dict[str, Any]:
    with _lock:
        return dict(_report)


def _reset_for_tests() -> None:
    global _groq_available, _rl_cache
    with _lock:
        _groq_available = None
        _resolved.clear()
        _report.clear()
        _gemini_cooldown.clear()
    _rl_cache = (0.0, None)


def _set_available_for_tests(ids: set[str]) -> None:
    global _groq_available
    with _lock:
        _groq_available = set(ids)
        _resolved.clear()
