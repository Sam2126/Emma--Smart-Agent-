"""
Groq API Key Pool — round-robin rotation with per-key cooldown.

Instead of sleeping when one key hits a 429, we immediately hot-swap
to the next available key in the pool. Each key tracks its own cooldown
window independently. Only when ALL keys are cooling simultaneously
do we fall back to sleeping.

Usage:
    pool = KeyPool(["gsk_key1", "gsk_key2", "gsk_key3", "gsk_key4"])
    key = pool.acquire()          # get next available key
    pool.mark_rate_limited(key, cooldown_seconds=13.0)  # on 429
    pool.mark_success(key)        # on success
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


class KeyPool:
    """
    Thread-safe round-robin API key pool with per-key, per-model cooldown.

    When a key is rate-limited it cools down for the exact duration Groq
    specifies (extracted from the error message), and callers immediately
    receive the next non-cooling key.

    Groq's limits are per model: a key that hit the tokens-per-minute limit on
    openai/gpt-oss-20b can still serve openai/gpt-oss-120b. Found 2026-09-15: a
    local task ran the fast model into its limit on all four keys, and the
    key-wide cooldown then also stalled the action model. Cooldowns are now
    kept per (key, model). Callers that pass no model (voice, vision) set and
    respect a key-wide cooldown, as before.
    """

    _ANY_MODEL = "*"

    def __init__(self, keys: list[str]) -> None:
        if not keys:
            raise ValueError("KeyPool requires at least one API key.")
        self._keys: list[str] = [k.strip() for k in keys if k.strip()]
        # Cooldown expiry timestamps per (key, model); missing = not cooling
        self._cooldown_until: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()
        # Round-robin pointer
        self._index: int = 0
        logger.info("key_pool_initialized", total_keys=len(self._keys))

    @property
    def size(self) -> int:
        return len(self._keys)

    def _until(self, key: str, model: str | None) -> float:
        """When `key` is usable again for `model` (caller holds the lock)."""
        until = self._cooldown_until.get((key, self._ANY_MODEL), 0.0)
        if model:
            until = max(until, self._cooldown_until.get((key, model), 0.0))
        return until

    def acquire(self, model: str | None = None) -> str:
        """
        Return the next key that is not cooling for `model`.

        Cycles through all keys starting from the current round-robin position.
        If all keys are cooling, returns the one that will cool down soonest
        (caller should sleep for `min_remaining_cooldown(model)` seconds first).
        """
        with self._lock:
            now = time.monotonic()
            n = len(self._keys)

            for offset in range(n):
                idx = (self._index + offset) % n
                key = self._keys[idx]
                if self._until(key, model) <= now:
                    # Advance pointer for next call
                    self._index = (idx + 1) % n
                    return key

            # All keys are cooling — return the one expiring soonest
            return min(self._keys, key=lambda k: self._until(k, model))

    def min_remaining_cooldown(self, model: str | None = None) -> float:
        """Seconds until at least one key is usable for `model` (0.0 if one already is)."""
        now = time.monotonic()
        with self._lock:
            soonest = min(self._until(k, model) for k in self._keys)
        return max(0.0, soonest - now)

    def mark_rate_limited(self, key: str, cooldown_seconds: float = 5.0, model: str | None = None) -> None:
        """Mark a key as rate-limited for `model` (or for every model) for `cooldown_seconds`."""
        with self._lock:
            now = time.monotonic()
            self._cooldown_until[(key, model or self._ANY_MODEL)] = now + cooldown_seconds
            remaining = sum(1 for k in self._keys if self._until(k, model) <= now)
        logger.warning(
            "key_rate_limited",
            key_suffix=key[-6:],
            model=model or "any",
            cooldown_seconds=f"{cooldown_seconds:.1f}s",
            remaining_available=remaining,
        )

    def mark_success(self, key: str, model: str | None = None) -> None:
        """A call succeeded: clear the key-wide cooldown and this model's cooldown."""
        with self._lock:
            self._cooldown_until.pop((key, self._ANY_MODEL), None)
            if model:
                self._cooldown_until.pop((key, model), None)

    def status(self) -> list[dict]:
        """Return current status of all keys (for monitoring)."""
        now = time.monotonic()
        with self._lock:
            return [
                {
                    "key_suffix": k[-6:],
                    "available": self._until(k, None) <= now,
                    "cools_in_seconds": max(0.0, self._until(k, None) - now),
                    "cooling_models": {
                        m: round(t - now, 1)
                        for (kk, m), t in self._cooldown_until.items()
                        if kk == k and m != self._ANY_MODEL and t > now
                    },
                }
                for k in self._keys
            ]


def extract_retry_delay(error_msg: str) -> float:
    """
    Extract the retry-after delay from a Groq/OpenAI 429 error message.

    Groq uses multiple formats depending on the error type:
      - "Please try again in 13.5s."
      - "Please retry after 1.3 seconds."
      - "Retry after 5 seconds"
      - JSON body: {"error":{"message":"...Please try again in 1.3s..."}}
      - LiteLLM wrapper: "litellm.RateLimitError: GroqException - {...}"

    Falls back to 8.0s (not 5s) to avoid hammering the API.
    """
    s = str(error_msg)

    # Pattern 1: "try again in X.Xs" or "try again in Xs"
    m = re.search(r"try again in ([\.\d]+)\s*s", s, re.IGNORECASE)
    if m:
        return float(m.group(1)) + 0.5

    # Pattern 2: "retry after X.X seconds" or "retry after X seconds"
    m = re.search(r"retry after ([\.\d]+)\s*s(?:ec(?:ond)?s?)?", s, re.IGNORECASE)
    if m:
        return float(m.group(1)) + 0.5

    # Pattern 3: "wait X seconds" / "wait Xs"
    m = re.search(r"wait ([\.\d]+)\s*s(?:ec(?:ond)?s?)?", s, re.IGNORECASE)
    if m:
        return float(m.group(1)) + 0.5

    # Pattern 4: X-RateLimit-Reset-Tokens or similar header values embedded in msg
    m = re.search(r"reset.{0,20}in ([\.\d]+)\s*s", s, re.IGNORECASE)
    if m:
        return float(m.group(1)) + 0.5

    # None of the patterns matched — log the raw error so we can add the right
    # regex pattern. 8s fallback is conservative but safe.
    logger.debug(
        "retry_delay_fallback_used",
        hint="Could not parse retry delay from error; using 8s default",
        error_snippet=s[:200],
    )
    return 8.0  # Conservative fallback — better to wait 8s than hammer all keys


def build_key_pool_from_env() -> Optional[KeyPool]:
    """
    Build a KeyPool from GROQ_API_KEYS (comma-separated) or fall back to GROQ_API_KEY.

    Priority:
      1. GROQ_API_KEYS=key1,key2,key3,key4  (multi-key pool)
      2. GROQ_API_KEY=single_key            (single key, still wrapped in pool for uniformity)
    """
    import os
    multi = os.getenv("GROQ_API_KEYS", "")
    single = os.getenv("GROQ_API_KEY", "")
    if not (multi or single):
        # pydantic-settings reads .env without exporting it to os.environ, so
        # the keys were only found when some other library had happened to
        # load .env first (found 2026-09-15: a standalone script got
        # "No Groq API keys configured" with keys present in .env).
        try:
            from app.config import get_settings

            settings = get_settings()
            multi, single = settings.groq_api_keys, settings.groq_api_key
        except Exception:
            pass
    if multi:
        keys = [k.strip() for k in multi.split(",") if k.strip()]
        if keys:
            return KeyPool(keys)

    if single:
        return KeyPool([single])

    return None


# Module-level singleton (created lazily on first use)
_pool: Optional[KeyPool] = None
_pool_lock = threading.Lock()


def get_key_pool() -> KeyPool:
    """Get or create the global key pool singleton."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = build_key_pool_from_env()
                if _pool is None:
                    raise RuntimeError(
                        "No Groq API keys configured. "
                        "Set GROQ_API_KEY or GROQ_API_KEYS in your .env file."
                    )
    return _pool
