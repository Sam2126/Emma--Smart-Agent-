"""
Vision provider chain for screenshots.

Used by see_window (desktop apps), see_page and perceive_page (browser) and
the vision judge in task verification.

Order is set by `vision_provider`:
  auto   -> Gemini first when GEMINI_API_KEY is set, Groq qwen as fallback
  gemini -> Gemini first, Groq fallback
  groq   -> Groq first, Gemini fallback

Verified 2026-09-15 with this project's key: gemini-3.6-flash read the word in
a test image and placed it correctly as a percentage position through LiteLLM.
The Groq model (qwen/qwen3.8-27b) works too but this account caps its output at
1000 tokens per minute, which is why Gemini goes first when available.

Returns None when no provider could describe the image, so callers can say so
honestly instead of guessing.
"""

from __future__ import annotations

import asyncio
import re

import httpx
import structlog

from app.config import get_settings

logger = structlog.get_logger(__name__)

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_VISION_MODEL = "qwen/qwen3.8-27b"
GEMINI_VISION_TIMEOUT_SECONDS = 25
_THINK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


_PCT_PAIR = re.compile(r"\(\s*(\d{1,3}(?:\.\d+)?)\s*%?\s*,\s*(\d{1,3}(?:\.\d+)?)\s*%?\s*\)")


def percent_to_pixels(text: str, width: int, height: int, off_x: int = 0, off_y: int = 0) -> str:
    """Rewrite every `(X%, Y%)` pair in the vision output as real pixels.

    The model is asked for percentages because it estimates relative position
    far better than absolute pixel counts; this converts them once, using the
    true captured size, so the agent receives directly clickable coordinates.

    `off_x`/`off_y` are the captured area's own origin, added for a whole-screen
    capture so the result is an absolute screen position even when the virtual
    desktop starts left of or above (0, 0).
    """
    def _sub(m: re.Match) -> str:
        try:
            xp, yp = float(m.group(1)), float(m.group(2))
        except ValueError:
            return m.group(0)
        if xp > 100 or yp > 100:       # already pixels — leave alone
            return m.group(0)
        return f"({off_x + int(round(xp / 100.0 * width))}, {off_y + int(round(yp / 100.0 * height))})"

    return _PCT_PAIR.sub(_sub, text)


def provider_order() -> list[str]:
    s = get_settings()
    has_gemini = bool(s.gemini_api_key)
    pref = (s.vision_provider or "auto").strip().lower()
    if pref == "groq":
        return ["groq", "gemini"] if has_gemini else ["groq"]
    return ["gemini", "groq"] if has_gemini else ["groq"]


async def describe_image(
    image_b64: str,
    prompt: str,
    *,
    mime: str = "image/png",
    max_tokens: int = 900,
) -> str | None:
    for provider in provider_order():
        if provider == "gemini":
            text = await _describe_with_gemini(image_b64, prompt, mime, max_tokens)
        else:
            text = await _describe_with_groq(image_b64, prompt, mime, max_tokens)
        if text:
            return text
    return None


def _content(prompt: str, image_b64: str, mime: str) -> list[dict]:
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
    ]


async def _describe_with_gemini(image_b64: str, prompt: str, mime: str, max_tokens: int) -> str | None:
    """Try each configured Gemini vision model in order (each has its own free quota).

    Models that were just overloaded or rate-limited are skipped (see
    llm.mark_gemini_unavailable); if every model is cooling, all are tried.
    """
    from app.utils.llm import gemini_model_available, gemini_models, mark_gemini_unavailable

    s = get_settings()
    import litellm

    models = gemini_models(s.gemini_vision_model)
    for model in [m for m in models if gemini_model_available(m)] or models:
        try:
            response = await asyncio.wait_for(
                litellm.acompletion(
                    model=model,
                    api_key=s.gemini_api_key,
                    # Gemini 3.x spends output tokens on internal reasoning before
                    # answering; a small budget returned truncated text in testing.
                    max_tokens=max(max_tokens, 2048),
                    reasoning_effort="low",
                    messages=[{"role": "user", "content": _content(prompt, image_b64, mime)}],
                ),
                # Healthy answers took 4-5 s in testing. The old 90 s limit let
                # one overloaded model stall a task's result check for 100 s.
                timeout=GEMINI_VISION_TIMEOUT_SECONDS,
            )
            text = _THINK.sub("", response.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception as e:
            mark_gemini_unavailable(model, e)
            logger.warning("vision_gemini_failed", model=model, error=(str(e) or type(e).__name__)[:200])
    return None


async def _describe_with_groq(image_b64: str, prompt: str, mime: str, max_tokens: int) -> str | None:
    from app.utils.key_pool import get_key_pool
    from app.utils.llm import resolve_groq_vision_model

    model = resolve_groq_vision_model()
    pool_model = f"groq/{model}"
    try:
        pool = get_key_pool()
        if pool is None or pool.size == 0:
            return None
        key = pool.acquire(model=pool_model)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": _content(prompt, image_b64, mime)}],
            "max_tokens": max_tokens,
        }
        async with httpx.AsyncClient(timeout=90.0) as client:
            r = await client.post(GROQ_CHAT_URL, headers={"Authorization": f"Bearer {key}"}, json=payload)
        if r.status_code == 200:
            pool.mark_success(key, model=pool_model)
            text = _THINK.sub("", r.json()["choices"][0]["message"]["content"] or "").strip()
            return text or None
        if r.status_code == 429:
            pool.mark_rate_limited(key, cooldown_seconds=10.0, model=pool_model)
        logger.warning("vision_groq_failed", status=r.status_code, body=r.text[:300])
    except Exception as e:
        logger.warning("vision_groq_error", error=str(e)[:200])
    return None
