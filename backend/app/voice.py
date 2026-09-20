"""
Voice input: speech-to-text via Groq Whisper for task instructions.

The extension / local-agent UI records the user's voice, sends the audio over
the WebSocket, and this module transcribes it with Groq's whisper model (same
provider and key pool as the rest of the stack — no extra API keys needed).
The transcript becomes the task instruction, so voice tasks flow through the
exact same planning / execution / learning pipeline as typed ones.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import structlog

from app.utils.key_pool import get_key_pool

logger = structlog.get_logger(__name__)

GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
DEFAULT_MODEL = "whisper-large-v3-turbo"
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # Groq limit


async def transcribe_audio(
    audio: bytes,
    filename: str = "audio.webm",
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    prompt: str | None = None,
) -> str:
    """
    Transcribe audio bytes to text via Groq Whisper.

    Cycles through the key pool on rate limits (same resilience as chat calls).
    Raises RuntimeError when no key is available or transcription fails.
    """
    if not audio:
        raise ValueError("Empty audio payload.")
    if len(audio) > MAX_AUDIO_BYTES:
        raise ValueError(f"Audio too large: {len(audio)} bytes (max {MAX_AUDIO_BYTES}).")

    pool = get_key_pool()
    if pool is None or pool.size == 0:
        raise RuntimeError(
            "No Groq API keys configured — voice input unavailable. "
            "Add GROQ_API_KEYS to .env."
        )

    last_error = ""
    for attempt in range(pool.size):
        key = pool.acquire()
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    GROQ_TRANSCRIBE_URL,
                    headers={"Authorization": f"Bearer {key}"},
                    files={"file": (filename, audio)},
                    data={
                        "model": model,
                        "response_format": "json",
                        "temperature": "0",
                        # A prompt is how a speech model is told which words to
                        # expect. Without it "Myntra" comes back as "Mentra"
                        # and the agent searches for a word that does not exist.
                        **({"prompt": prompt} if prompt else {}),
                        **({"language": language} if language else {}),
                    },
                )
        except Exception as e:
            # Network trouble: another key's request may still get through.
            last_error = str(e)
            logger.warning("voice_transcribe_error", error=str(e)[:200])
            continue
        if response.status_code == 200:
            text = (response.json().get("text") or "").strip()
            pool.mark_success(key)
            logger.info("voice_transcribed", chars=len(text), model=model)
            return text
        if response.status_code == 429:
            delay = 2.0
            pool.mark_rate_limited(key, cooldown_seconds=delay)
            last_error = f"rate limited (key #{attempt + 1})"
            logger.warning("voice_transcribe_rate_limited", key_index=attempt)
            await asyncio.sleep(delay)
            continue
        # 4xx (bad audio) / 5xx: retrying other keys cannot help — fail fast
        # with Groq's actual message. This raise used to sit inside the try
        # above, whose except caught it, so the same rejected audio was sent
        # again with every key before the task failed anyway.
        body = response.text[:300]
        logger.warning("voice_transcribe_failed", status=response.status_code, body=body)
        raise RuntimeError(f"Groq rejected the audio (HTTP {response.status_code}): {body}")

    raise RuntimeError(f"Voice transcription failed: {last_error}")


def guess_extension(mime: str) -> str:
    """Pick a sensible filename extension from a mime type (for Whisper decoding)."""
    mime = (mime or "").lower()
    if "ogg" in mime:
        return "audio.ogg"
    if "mp3" in mime or "mpeg" in mime:
        return "audio.mp3"
    if "wav" in mime:
        return "audio.wav"
    if "mp4" in mime:
        return "audio.mp4"
    if "flac" in mime:
        return "audio.flac"
    return "audio.webm"
