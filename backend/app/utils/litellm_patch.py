"""
Shared LiteLLM patch module.

Applies to every litellm.completion / litellm.acompletion call in the process:

1. Payload sanitization: strips Anthropic-specific cache headers and compresses
   oversized message history that would trip Groq's token-per-minute limits.
2. Tool guards for Groq-hosted gpt-oss models: tool_choice='none' -> 'auto',
   and argument schemas relaxed so only arguments without defaults are required.
3. Groq key-pool rotation on rate limits (429 / TPM / RPM), with call spacing.
4. Gemini fallback: when every Groq key is exhausted, or Groq is down, the same
   request is retried once on the free Gemini model (GEMINI_API_KEY,
   llm_fallback_model). Without a key, behaviour is unchanged.

Only requests whose model starts with "groq/" use the key pool. Requests for
other providers (gemini/, openai/) are sanitized and passed straight through
with their own api_key. Previously a Groq key was forced into every request,
which would have broken any non-Groq call.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

import litellm
import structlog

logger = structlog.get_logger(__name__)

litellm.drop_params = True
litellm.enable_cache = False
# LiteLLM printed two INFO lines for every call and a "Give Feedback / Get
# Help" banner for every handled error, burying the agent's own step log.
litellm.suppress_debug_info = True
for _litellm_logger in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy"):
    logging.getLogger(_litellm_logger).setLevel(logging.WARNING)

_orig_completion = litellm.completion
_orig_acompletion = litellm.acompletion

_MIN_CALL_INTERVAL: float = 2.0
_last_call_time: float = 0.0
_call_lock = threading.Lock()
_is_patched = False

_TRANSIENT_MARKERS = (
    "503", "502", "500 ", "internal server error", "service unavailable",
    "overloaded", "bad gateway", "timed out", "timeout", "connection error",
    "connecterror", "apiconnectionerror",
)


_HISTORY_COMPRESS_CHARS = 16000
_MAX_MESSAGE_CHARS = 3000


def _strip_cache_keys(data) -> None:
    """Remove Anthropic cache headers, anywhere in a message, that Groq/OpenAI reject with 400."""
    if isinstance(data, list):
        for item in data:
            _strip_cache_keys(item)
    elif isinstance(data, dict):
        data.pop("cache_breakpoint", None)
        data.pop("cache_control", None)
        for v in data.values():
            if isinstance(v, (list, dict)):
                _strip_cache_keys(v)


def _sanitize_payload(messages) -> None:
    """
    1. Remove Anthropic cache headers that Groq/OpenAI reject with 400.
    2. Keep oversized history from tripping Groq's tokens-per-minute limits.

    Never shortened: the system prompt, the first user message (the task, the
    plan, the exact spellings) and the latest two messages (the newest tool
    result). Found 2026-09-15: this used to cut EVERY message over 3000 chars
    to 2500, and every older message to 450 chars once a conversation passed
    6000 chars — the system prompt included. The local actor's 6356-char
    protocol reached the model as 2534 chars on every call, and as 478 chars a
    few steps into a task, so rules such as "after sending, check once and
    report — never send again" were never seen.
    """
    _strip_cache_keys(messages)
    if not isinstance(messages, list):
        return
    first_user = next(
        (i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "user"), None
    )
    recent_from = len(messages) - 2
    total_len = sum(
        len(m["content"]) for m in messages if isinstance(m, dict) and isinstance(m.get("content"), str)
    )
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") == "system" or i == first_user or i >= recent_from:
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        if total_len > _HISTORY_COMPRESS_CHARS and len(content) > 600:
            msg["content"] = content[:300] + "\n...[history compressed]...\n" + content[-150:]
        elif len(content) > _MAX_MESSAGE_CHARS:
            msg["content"] = content[:2500] + "\n...[truncated for token limit]..."


def _fix_tool_choice(kwargs: dict) -> None:
    """
    Fix Groq's "Tool choice is none, but model called a tool" error.

    When tool_choice='none' (or {'type': 'none'}), the caller intends to force a
    plain-text final answer. But Groq-hosted openai/gpt-oss-* models (trained
    with baked-in "harmony" tool habits) essentially always try to call a
    tool again as soon as a prior tool result is in the conversation —
    verified empirically (both the 20b and 120b variants) against this
    project's actual Groq account, including with `tools` removed from the
    request entirely: Groq's 400 fires either way, because it's rejecting
    the *shape* of the model's raw generation, not validating against the
    tools list. Stripping tools therefore cannot fix this — it always fails
    the same way, it just also disables any tools for the rest of the call.

    The one request shape that reliably avoids the 400 (also verified
    empirically) is tool_choice='auto' with tools still attached: the model
    then either answers in plain text, or legitimately calls a tool again —
    which the tool-calling loop continues from as a normal step, not an error.
    """
    tc = kwargs.get("tool_choice")
    is_none = (
        str(tc).lower() == "none"
        or tc is None and "tools" not in kwargs
        or (isinstance(tc, dict) and str(tc.get("type", "")).lower() == "none")
    )
    if not is_none:
        return
    if kwargs.get("tools"):
        kwargs["tool_choice"] = "auto"
    else:
        kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)


def _relax_tool_schemas(kwargs: dict) -> None:
    """
    Make arguments that have a default optional again in every tool schema.

    CrewAI converted each tool's Pydantic model with OpenAI's *strict-mode*
    helper, which marks EVERY property as required and sets "strict": true —
    including arguments that have defaults, like send_keys' `text`, `keys` and
    `delay_ms`. Groq validates each tool call against that list, so a perfectly
    sensible call such as send_keys(window_hint='WhatsApp', keys='^f') was
    rejected with "missing properties: 'text', 'delay_ms'" and the whole task
    failed. Verified against this project's Groq account: the same call fails
    with that schema and succeeds once only the genuinely required arguments
    (those without a "default") are listed and "strict" is dropped. The engine's
    own BaseTool already emits relaxed schemas; this stays as a safety net.
    """
    tools = kwargs.get("tools")
    if not isinstance(tools, list):
        return
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(fn, dict):
            continue
        params = fn.get("parameters")
        if isinstance(params, dict):
            props = params.get("properties")
            required = params.get("required")
            if isinstance(props, dict) and isinstance(required, list):
                params["required"] = [
                    name for name in required
                    if not (isinstance(props.get(name), dict) and "default" in props[name])
                ]
        if fn.get("strict") is True:
            fn.pop("strict", None)


def _prepare(kwargs: dict) -> None:
    if "messages" in kwargs:
        _sanitize_payload(kwargs["messages"])
    _fix_tool_choice(kwargs)
    _relax_tool_schemas(kwargs)


def _is_groq_request(args: tuple, kwargs: dict) -> bool:
    model = kwargs.get("model") or (args[0] if args else "")
    return str(model).startswith("groq/")


def _is_rate_limit(err_str: str) -> bool:
    low = err_str.lower()
    return (
        "rate_limit" in low or "429" in err_str or "tpm" in low
        or "rpm" in low or "ratelimit" in low
    )


def _is_too_large(err_str: str) -> bool:
    """The request alone is over the model's per-minute token limit: no Groq key can serve it.

    Found 2026-09-17: Groq answers "Request too large ... please reduce your
    message size" as a rate limit, so every key was tried and put on a
    cooldown (slowing the next, normal-sized call) before Gemini got the request.
    """
    low = err_str.lower()
    return "request too large" in low or "reduce your message size" in low


def _is_tool_use_failed(err_str: str) -> bool:
    low = err_str.lower()
    return "tool choice is none" in low or "tool_use_failed" in low


def _is_transient(err_str: str) -> bool:
    low = err_str.lower()
    return any(marker in low for marker in _TRANSIENT_MARKERS)


def _retry_with_auto_tool_choice(kwargs: dict) -> None:
    # _fix_tool_choice() already converts tool_choice='none' to 'auto' before
    # every call, so this should rarely trigger. If it still does, stripping
    # tools does NOT help (verified: Groq 400s on this model family either way).
    if kwargs.get("tools"):
        kwargs["tool_choice"] = "auto"
    else:
        kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)


def _fallback_models() -> list[str]:
    """Gemini models to try, in order, when Groq cannot serve a request."""
    from app.config import get_settings

    s = get_settings()
    if not s.gemini_api_key:
        return []
    return [m.strip() for m in (s.llm_fallback_model or "").split(",") if m.strip()]


def _max_cooldown_wait() -> float:
    from app.config import get_settings

    return float(get_settings().groq_max_cooldown_wait_seconds)


def _run_fallbacks_sync(kwargs: dict, last_error: Exception):
    from app.utils.llm import mark_gemini_unavailable, order_by_availability

    for model in order_by_availability(_fallback_models()):
        logger.warning("groq_unavailable_using_gemini_fallback", model=model, error=str(last_error)[:160])
        try:
            return _orig_completion(**_fallback_kwargs(kwargs, model))
        except Exception as e:
            last_error = e
            mark_gemini_unavailable(model, e)
            logger.warning("gemini_fallback_failed", model=model, error=(str(e) or type(e).__name__)[:160])
    raise last_error


async def _run_fallbacks_async(kwargs: dict, last_error: Exception):
    from app.utils.llm import mark_gemini_unavailable, order_by_availability

    for model in order_by_availability(_fallback_models()):
        logger.warning("groq_unavailable_using_gemini_fallback", model=model, error=str(last_error)[:160])
        try:
            return await _orig_acompletion(**_fallback_kwargs(kwargs, model))
        except Exception as e:
            last_error = e
            mark_gemini_unavailable(model, e)
            logger.warning("gemini_fallback_failed", model=model, error=(str(e) or type(e).__name__)[:160])
    raise last_error


def _fallback_kwargs(kwargs: dict, model: str | None = None) -> dict | None:
    """The same request re-targeted at a Gemini fallback model, or None."""
    from app.config import get_settings

    models = _fallback_models()
    if not models:
        return None
    fb = dict(kwargs)
    fb["model"] = model or models[0]
    fb["api_key"] = get_settings().gemini_api_key
    for key in ("api_base", "base_url", "parallel_tool_calls"):
        fb.pop(key, None)
    # Gemini 3.x uses output tokens for internal reasoning before answering.
    fb["max_tokens"] = max(int(fb.get("max_tokens") or 0), 4096)
    return fb


def _sanitized_completion(*args, **kwargs):
    global _last_call_time
    _prepare(kwargs)
    if not _is_groq_request(args, kwargs):
        return _orig_completion(*args, **kwargs)

    from app.utils.key_pool import get_key_pool, extract_retry_delay

    try:
        pool = get_key_pool()
    except RuntimeError:
        return _orig_completion(*args, **kwargs)

    with _call_lock:
        elapsed = time.monotonic() - _last_call_time
        if elapsed < _MIN_CALL_INTERVAL:
            time.sleep(_MIN_CALL_INTERVAL - elapsed)
        _last_call_time = time.monotonic()

    # Groq rate limits are per model, so cooldowns are tracked per (key, model).
    model_name = str(kwargs.get("model") or (args[0] if args else ""))
    max_attempts = pool.size * 2
    last_error = None

    for attempt in range(max_attempts):
        key = pool.acquire(model=model_name)
        kwargs["api_key"] = key

        cooling = pool.min_remaining_cooldown(model=model_name)
        if cooling > 0:
            if not args and cooling > _max_cooldown_wait() and _fallback_models():
                # Every key is cooling for this model: Gemini now beats waiting.
                logger.warning("all_keys_cooling_switching_to_gemini", model=model_name, cooling_seconds=f"{cooling:.1f}s")
                last_error = last_error or RuntimeError(f"All Groq keys are rate-limited for {model_name}")
                break
            logger.warning("all_keys_cooling_sleeping", sleep_seconds=f"{cooling:.1f}s", attempt=attempt)
            time.sleep(cooling)

        try:
            result = _orig_completion(*args, **kwargs)
            pool.mark_success(key, model=model_name)
            return result
        except Exception as e:
            err_str = str(e)
            if _is_too_large(err_str):
                logger.warning("groq_request_too_large", model=model_name, error=err_str[:160])
                last_error = e
                break
            if _is_rate_limit(err_str):
                pool.mark_rate_limited(key, cooldown_seconds=extract_retry_delay(err_str), model=model_name)
                last_error = e
                time.sleep(0.5)
                continue
            if _is_tool_use_failed(err_str):
                _retry_with_auto_tool_choice(kwargs)
                last_error = e
                continue
            if _is_transient(err_str):
                last_error = e
                break
            raise

    if not args and last_error is not None and _fallback_models():
        return _run_fallbacks_sync(kwargs, last_error)
    raise last_error or RuntimeError("All key pool attempts exhausted")


async def _sanitized_acompletion(*args, **kwargs):
    global _last_call_time
    _prepare(kwargs)
    if not _is_groq_request(args, kwargs):
        return await _orig_acompletion(*args, **kwargs)

    from app.utils.key_pool import get_key_pool, extract_retry_delay

    try:
        pool = get_key_pool()
    except RuntimeError:
        return await _orig_acompletion(*args, **kwargs)

    with _call_lock:
        elapsed = time.monotonic() - _last_call_time
        wait = _MIN_CALL_INTERVAL - elapsed
        _last_call_time = time.monotonic()
    if wait > 0:
        await asyncio.sleep(wait)

    # Groq rate limits are per model, so cooldowns are tracked per (key, model).
    model_name = str(kwargs.get("model") or (args[0] if args else ""))
    max_attempts = pool.size * 2
    last_error = None

    for attempt in range(max_attempts):
        key = pool.acquire(model=model_name)
        kwargs["api_key"] = key

        cooling = pool.min_remaining_cooldown(model=model_name)
        if cooling > 0:
            if not args and cooling > _max_cooldown_wait() and _fallback_models():
                # Every key is cooling for this model: Gemini now beats waiting.
                logger.warning("all_keys_cooling_switching_to_gemini", model=model_name, cooling_seconds=f"{cooling:.1f}s")
                last_error = last_error or RuntimeError(f"All Groq keys are rate-limited for {model_name}")
                break
            logger.warning("all_keys_cooling_sleeping_async", sleep_seconds=f"{cooling:.1f}s", attempt=attempt)
            await asyncio.sleep(cooling)

        try:
            result = await _orig_acompletion(*args, **kwargs)
            pool.mark_success(key, model=model_name)
            return result
        except Exception as e:
            err_str = str(e)
            if _is_too_large(err_str):
                logger.warning("groq_request_too_large", model=model_name, error=err_str[:160])
                last_error = e
                break
            if _is_rate_limit(err_str):
                pool.mark_rate_limited(key, cooldown_seconds=extract_retry_delay(err_str), model=model_name)
                last_error = e
                await asyncio.sleep(0.5)
                continue
            if _is_tool_use_failed(err_str):
                _retry_with_auto_tool_choice(kwargs)
                last_error = e
                continue
            if _is_transient(err_str):
                last_error = e
                break
            raise

    if not args and last_error is not None and _fallback_models():
        return await _run_fallbacks_async(kwargs, last_error)
    raise last_error or RuntimeError("All key pool attempts exhausted (async)")


def apply_litellm_patch():
    """Apply sanitization, key-pool and fallback patches to LiteLLM globally."""
    global _is_patched
    if _is_patched:
        return
    litellm.completion = _sanitized_completion
    litellm.acompletion = _sanitized_acompletion
    _is_patched = True
    logger.debug("litellm_patch_applied")
