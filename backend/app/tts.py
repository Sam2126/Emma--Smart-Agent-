"""
Emma's voice: text to speech, spoken through the computer's speakers.

Added 2026-09-20. The agent could hear but not answer: everything it had to say
arrived as text on a screen the user was often not looking at. This speaks it.

=============================================================================
Three voices, tried in order, so Emma is never silent
=============================================================================
  1. Groq  "canopylabs/orpheus-v1-english"  - natural and expressive, free on
     the project's existing Groq key. The model needs its terms accepted once
     in the Groq console; until then this tier reports itself unavailable and
     the next one answers, so nothing breaks while that is pending.
  2. Gemini "gemini-3.1-flash-tts-preview"  - high quality, free tier, shares
     the same key the vision calls use.
  3. Windows' own voice (Zira / Hazel / David) - robotic, but instant, offline,
     unlimited, and always installed. The floor beneath everything else.

A tier that fails, is rate limited or times out is put on a short cooldown and
the next one takes over mid-sentence. Cloud tiers are skipped entirely for
anything marked sensitive (see `say(..., sensitive=True)`).

=============================================================================
Why it is built this way
=============================================================================
SPEAKING MUST NOT WAKE THE AGENT. The wake word listener is always on. When
Emma says "Emma is working on it", the microphone hears "emma" and the agent
triggers itself, forever. `is_speaking()` exists for the listener to gate on,
and stays true for a short tail after the audio stops so the room's echo is
covered too.

SENTENCE AT A TIME. The first sentence is synthesized and played while the
second is still being made, so the user hears a reply in under a second
instead of waiting for a whole paragraph to be rendered.

NOTHING IS SAID TWICE AT FULL PRICE. Synthesized audio is cached on disk by
(text, voice, tier), so "Task complete" costs one request in the lifetime of
the installation.

STOP MEANS STOP. `stop()` clears the queue and cuts the audio immediately; the
worker is a daemon thread and dies with the backend, so nothing keeps talking
after the user has stopped the agent.
"""

from __future__ import annotations

import hashlib
import os
import queue
import re
import struct
import subprocess
import threading
import time
from base64 import b64decode
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import structlog

from app.config import get_settings

logger = structlog.get_logger(__name__)

CACHE_DIR = Path.home() / ".self_improving_agent" / "tts_cache"
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

GROQ_SPEECH_URL = "https://api.groq.com/openai/v1/audio/speech"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# How long a tier sits out after failing, so one outage does not slow every line.
COOLDOWN_SECONDS = 120.0
# A single request's worth of speech. Longer text is split into sentences first.
MAX_CHUNK_CHARS = 240
# The microphone stays gated this long after the audio ends, to cover the room's
# echo and the tail of the speakers.
SPEAKING_TAIL_SECONDS = 0.35


class Priority(IntEnum):
    """Lower value speaks first, and may interrupt what is already speaking."""

    URGENT = 0     # an error, or a question the agent needs answered
    ANSWER = 1     # the reply to something the user asked
    RESULT = 2     # what happened at the end of a task
    PROGRESS = 3   # step-by-step chatter; dropped when the queue is backed up


@dataclass(order=True)
class _Utterance:
    priority: int
    sequence: int
    text: str = field(compare=False)
    sensitive: bool = field(compare=False, default=False)


# =============================================================================
# Making text sound like speech rather than a screen
# =============================================================================

_EMOJI = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002700-\U000027BF" "\U0001F1E6-\U0001F1FF"
    "\U00002600-\U000026FF" "\U0000FE0F" "\U00002190-\U000021FF" "]+"
)
_URL = re.compile(r"https?://\S+|www\.\S+")
_WINDOWS_PATH = re.compile(r"[A-Za-z]:\\[^\s'\"]+")
_MARKDOWN = re.compile(r"[*_`#>]{1,3}")
_RUPEES = re.compile(r"(?:Rs\.?|\u20b9|INR)\s?([\d,]+(?:\.\d+)?)", re.IGNORECASE)
_TOOL_NAME = re.compile(r"\b([a-z]+(?:_[a-z]+){1,3})\b")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# Hindi and other Indian scripts. The offline Windows voices are English only
# and read these as noise, so the text is sent to a voice that can speak them.
_INDIC = re.compile(r"[\u0900-\u097F\u0A80-\u0AFF\u0980-\u09FF\u0B80-\u0BFF]")

_TIME = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)(?::[0-5]\d)?\b")
_PERCENT = re.compile(r"\b(\d+(?:\.\d+)?)\s?%")
_ORDINAL = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
_RANGE = re.compile(r"\b(\d+)\s?-\s?(\d+)\b")
_DECIMAL = re.compile(r"\b(\d+)\.(\d+)\b")
_SIZE = re.compile(r"\b(\d+(?:\.\d+)?)\s?(KB|MB|GB|TB)\b", re.IGNORECASE)
_PLAIN_NUMBER = re.compile(r"\b\d{1,9}(?:,\d{2,3})*\b")
# A hash, a task id or a session key: never worth hearing digit by digit.
_IDENTIFIER = re.compile(r"\b(?=[a-z0-9-]*\d)(?=[a-z0-9-]*[a-z])[a-f0-9][a-f0-9-]{11,}\b", re.IGNORECASE)
_SHOUTING = re.compile(r"\b[A-Z]{4,}\b")
_REPEATED_PUNCTUATION = re.compile(r"([!?.]){2,}")

# Letters that should be spoken as letters, not as a word.
_SPELLED_OUT = {
    "OTP": "O T P", "PDF": "P D F", "URL": "U R L", "ID": "I D", "API": "A P I",
    "CPU": "C P U", "GPU": "G P U", "USB": "U S B", "PC": "P C", "AI": "A I",
    "CSV": "C S V", "HTML": "H T M L", "JSON": "Jason", "SMS": "S M S",
    "PIN": "PIN", "OK": "okay", "FAQ": "F A Q", "UPI": "U P I", "PNG": "P N G",
}
_UNITS = {"kb": "kilobytes", "mb": "megabytes", "gb": "gigabytes", "tb": "terabytes"}

_TOOL_NAMES_SPOKEN = {
    "see_window": "looking at the window", "see_page": "looking at the page",
    "read_page_as_markdown": "reading the page", "read_window_as_markdown": "reading the window",
    "click_element": "clicking", "click_at_position": "clicking", "click_window": "clicking",
    "type_into_element": "typing", "send_keys": "typing", "navigate_browser": "opening the site",
    "perceive_page": "reading the page", "open_app": "opening the app",
    "send_email": "sending the email", "search_emails": "searching your mail",
    "find_files": "searching your files", "run_command": "running a command",
    "add_to_cart": "adding to the cart", "list_windows": "checking what is open",
}
_SPOKEN_TOOL_NAMES = _TOOL_NAMES_SPOKEN     # kept under the old name too

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_ORDINAL_WORDS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh",
    8: "eighth", 9: "ninth", 10: "tenth", 11: "eleventh", 12: "twelfth", 13: "thirteenth",
    20: "twentieth", 21: "twenty first", 30: "thirtieth", 31: "thirty first",
}


def _under_thousand(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + _under_thousand(n % 100))


def _say_number(digits: str) -> str:
    """'1,299' -> 'one thousand two hundred ninety nine'.

    Indian grouping is used above a hundred thousand, because that is how the
    user's own prices and counts are said: 2,822 items, 1 lakh 1,187 items.
    """
    try:
        value = int(str(digits).replace(",", ""))
    except ValueError:
        return str(digits)
    if value < 0:
        return "minus " + _say_number(str(-value))
    if value >= 10_000_000:
        rest = value % 10_000_000
        return _say_number(str(value // 10_000_000)) + " crore" + (
            "" if not rest else " " + _say_number(str(rest)))
    if value >= 100_000:
        rest = value % 100_000
        return _say_number(str(value // 100_000)) + " lakh" + (
            "" if not rest else " " + _say_number(str(rest)))
    if value < 1000:
        return _under_thousand(value)
    return _under_thousand(value // 1000) + " thousand" + (
        "" if value % 1000 == 0 else " " + _under_thousand(value % 1000))


def _say_decimal(number: str) -> str:
    """'2.5' -> 'two point five'; whole numbers keep their plain form."""
    whole, _, fraction = str(number).partition(".")
    spoken = _say_number(whole)
    if fraction:
        spoken += " point " + " ".join(_ONES[int(d)] for d in fraction)
    return spoken


def _say_time(hour: int, minute: int) -> str:
    """16:19 -> 'four nineteen in the afternoon'; 9:05 -> 'nine oh five in the morning'."""
    part = "in the morning" if hour < 12 else ("in the afternoon" if hour < 17 else "in the evening")
    spoken_hour = hour % 12 or 12
    if minute == 0:
        return f"{_ONES[spoken_hour]} o'clock {part}"
    if minute < 10:
        return f"{_ONES[spoken_hour]} oh {_ONES[minute]} {part}"
    return f"{_ONES[spoken_hour]} {_under_thousand(minute)} {part}"


def _say_ordinal(number: int) -> str:
    if number in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[number]
    if number % 10 == 1 and number % 100 != 11:
        return _say_number(str(number)) + "st"
    return _say_number(str(number)) + "th"


def is_indic(text: str) -> bool:
    """True when the text is in a script the offline English voices cannot read."""
    return bool(_INDIC.search(text or ""))


def speakable(text: str) -> str:
    """Turn what the agent wrote on screen into what it should sound like.

    Markdown, emoji, URLs, paths, tool names, clock times, sizes, percentages
    and long identifiers all read terribly aloud: "see_window" becomes "see
    underscore window", "16:19" becomes "sixteen nineteen", and a task id
    becomes forty seconds of hexadecimal.
    """
    if not text:
        return ""
    out = _EMOJI.sub(" ", text)
    # Before the markdown pass, which would strip the arrow's ">" and leave a dash.
    out = out.replace("->", " then ").replace("→", " then ")
    out = _URL.sub("the link", out)
    out = _WINDOWS_PATH.sub(lambda m: "the file " + Path(m.group(0)).name, out)
    out = _IDENTIFIER.sub("an identifier", out)
    out = _RUPEES.sub(lambda m: _say_number(m.group(1)) + " rupees", out)
    out = _SIZE.sub(lambda m: _say_decimal(m.group(1)) + " " + _UNITS[m.group(2).lower()], out)
    out = _TIME.sub(lambda m: _say_time(int(m.group(1)), int(m.group(2))), out)
    out = _PERCENT.sub(lambda m: _say_number(m.group(1).split(".")[0]) + " percent", out)
    out = _ORDINAL.sub(lambda m: _say_ordinal(int(m.group(1))), out)
    out = _RANGE.sub(lambda m: f"{_say_number(m.group(1))} to {_say_number(m.group(2))}", out)
    out = _DECIMAL.sub(lambda m: f"{_say_number(m.group(1))} point {' '.join(_ONES[int(d)] for d in m.group(2))}", out)
    out = _TOOL_NAME.sub(lambda m: _TOOL_NAMES_SPOKEN.get(m.group(1), m.group(1).replace("_", " ")), out)
    out = re.sub(r"\b(" + "|".join(_SPELLED_OUT) + r")\b", lambda m: _SPELLED_OUT[m.group(1).upper()], out)
    out = _SHOUTING.sub(lambda m: m.group(0).capitalize(), out)
    out = _PLAIN_NUMBER.sub(lambda m: _say_number(m.group(0)), out)
    out = _MARKDOWN.sub("", out)
    out = out.replace("|", ", ").replace("&", " and ").replace("@", " at ")
    out = _REPEATED_PUNCTUATION.sub(r"\1", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def sentences(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split into pieces small enough to synthesize quickly, on sentence ends."""
    chunks: list[str] = []
    for piece in _SENTENCE_END.split(text):
        piece = piece.strip()
        while len(piece) > limit:
            cut = piece.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            chunks.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            chunks.append(piece)
    return chunks


# =============================================================================
# Audio
# =============================================================================

def _wav_from_pcm(pcm: bytes, rate: int = 24000, channels: int = 1, width: int = 2) -> bytes:
    """Wrap raw PCM (what Gemini returns) in a WAV header so it can be played."""
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack(
        "<IHHIIHH", 16, 1, channels, rate, rate * channels * width, channels * width, width * 8
    ) + b"data" + struct.pack("<I", len(pcm))
    return header + pcm


def _looks_like_wav(data: bytes) -> bool:
    return len(data) > 44 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


# =============================================================================
# The three voices
# =============================================================================

class Tier:
    GROQ = "groq"
    GEMINI = "gemini"
    WINDOWS = "windows"


class _Voices:
    """Synthesis, one method per tier, each returning WAV bytes or None."""

    def __init__(self) -> None:
        self._cooling: dict[str, float] = {}
        self._lock = threading.Lock()

    def available(self, tier: str) -> bool:
        with self._lock:
            return time.monotonic() >= self._cooling.get(tier, 0.0)

    def cool(self, tier: str, seconds: float = COOLDOWN_SECONDS, reason: str = "") -> None:
        with self._lock:
            self._cooling[tier] = time.monotonic() + seconds
        logger.warning("tts_tier_cooling", tier=tier, seconds=seconds, reason=reason[:160])

    # -- 1. Groq Orpheus ---------------------------------------------------
    def groq(self, text: str, voice: str) -> bytes | None:
        import httpx

        from app.utils.key_pool import get_key_pool

        settings = get_settings()
        try:
            key = get_key_pool().acquire()
        except Exception as e:
            self.cool(Tier.GROQ, reason=f"no key: {e}")
            return None
        try:
            response = httpx.post(
                GROQ_SPEECH_URL,
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": settings.tts_groq_model,
                    "input": text,
                    "voice": voice or settings.tts_groq_voice,
                    "response_format": "wav",
                },
                timeout=settings.tts_timeout_seconds,
            )
        except Exception as e:
            self.cool(Tier.GROQ, reason=str(e))
            return None

        if response.status_code == 200 and _looks_like_wav(response.content):
            return response.content
        detail = response.text[:200]
        # "requires terms acceptance" is not a fault to retry every line: the
        # user has to click once in the Groq console. Sit out for longer.
        if "terms acceptance" in detail:
            self.cool(Tier.GROQ, seconds=1800, reason="model terms not accepted yet")
            logger.warning("tts_groq_needs_terms_acceptance", url="https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english")
        else:
            self.cool(Tier.GROQ, reason=f"HTTP {response.status_code}: {detail}")
        return None

    # -- 2. Gemini ----------------------------------------------------------
    def gemini(self, text: str, voice: str) -> bytes | None:
        import httpx

        settings = get_settings()
        key = settings.gemini_api_key
        if not key:
            self.cool(Tier.GEMINI, seconds=3600, reason="no GEMINI_API_KEY")
            return None
        try:
            response = httpx.post(
                GEMINI_URL.format(model=settings.tts_gemini_model),
                params={"key": key},
                json={
                    "contents": [{"parts": [{"text": text}]}],
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                        "speechConfig": {
                            "voiceConfig": {
                                "prebuiltVoiceConfig": {"voiceName": voice or settings.tts_gemini_voice}
                            }
                        },
                    },
                },
                timeout=settings.tts_timeout_seconds,
            )
        except Exception as e:
            self.cool(Tier.GEMINI, reason=str(e))
            return None

        if response.status_code != 200:
            self.cool(Tier.GEMINI, reason=f"HTTP {response.status_code}: {response.text[:200]}")
            return None
        try:
            part = response.json()["candidates"][0]["content"]["parts"][0]["inlineData"]
            pcm = b64decode(part["data"])
            rate = 24000
            match = re.search(r"rate=(\d+)", part.get("mimeType", ""))
            if match:
                rate = int(match.group(1))
            return _wav_from_pcm(pcm, rate=rate)
        except Exception as e:
            self.cool(Tier.GEMINI, reason=f"unexpected reply: {e}")
            return None

    # -- 3. Windows' own voice ---------------------------------------------
    def windows(self, text: str, voice: str) -> bytes | None:
        """Offline, unlimited, always there. Also the voice for private text."""
        settings = get_settings()
        target = CACHE_DIR / f"_local_{os.getpid()}.wav"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        wanted = (voice or settings.tts_windows_voice or "").replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate = {int(settings.tts_windows_rate)}; "
            + (f"try {{ $s.SelectVoice('{wanted}') }} catch {{}}; " if wanted else "")
            + f"$s.SetOutputToWaveFile('{str(target).replace(chr(39), chr(39) * 2)}'); "
            f"$s.Speak('{text.replace(chr(39), chr(39) * 2)}'); $s.Dispose()"
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True, timeout=settings.tts_timeout_seconds, creationflags=_NO_WINDOW,
            )
            data = target.read_bytes()
            target.unlink(missing_ok=True)
            return data if _looks_like_wav(data) else None
        except Exception as e:
            logger.warning("tts_windows_voice_failed", error=str(e)[:160])
            return None

    # -- the chain ----------------------------------------------------------
    def synthesize(self, text: str, sensitive: bool) -> tuple[bytes | None, str]:
        settings = get_settings()
        if sensitive:
            # Private text never leaves the computer, whatever the preference is.
            # Hindi spoken by an English voice is poor, but a one-time code read
            # to a voice service is worse.
            return self.windows(text, settings.tts_windows_voice), Tier.WINDOWS

        order = [t.strip() for t in (settings.tts_order or "").split(",") if t.strip()]
        order = order or [Tier.GROQ, Tier.GEMINI, Tier.WINDOWS]
        if is_indic(text):
            # Zira, Hazel and David are English voices: given Devanagari they
            # produce noise. Gemini speaks it properly, so the local voice
            # becomes the last resort rather than an equal choice.
            order = [t for t in order if t != Tier.WINDOWS] + [Tier.WINDOWS]
            if Tier.GEMINI in order:
                order.remove(Tier.GEMINI)
                order.insert(0, Tier.GEMINI)
        for tier in order:
            if not self.available(tier):
                continue
            started = time.monotonic()
            if tier == Tier.GROQ:
                audio = self.groq(text, settings.tts_groq_voice)
            elif tier == Tier.GEMINI:
                audio = self.gemini(text, settings.tts_gemini_voice)
            else:
                audio = self.windows(text, settings.tts_windows_voice)
            if audio:
                logger.debug("tts_synthesized", tier=tier, ms=int((time.monotonic() - started) * 1000),
                             chars=len(text))
                return audio, tier
        return None, ""


# =============================================================================
# The speaker
# =============================================================================

class Speaker:
    """Queues what Emma says, speaks it, and can be stopped at any moment."""

    def __init__(self) -> None:
        self._queue: queue.PriorityQueue[_Utterance] = queue.PriorityQueue()
        self._voices = _Voices()
        self._sequence = 0
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._running = False
        self._speaking_until = 0.0
        self._generation = 0          # bumped by stop(), so in-flight audio is dropped
        self._last_said = ""
        self._current_text = ""       # what is being spoken, for echo suppression
        self._listeners: list = []    # called with True/False as speech starts and ends

    # -- state others can watch --------------------------------------------
    def is_speaking(self) -> bool:
        """True while audio is playing, and for a short tail afterwards.

        The wake word listener gates on this: without it, Emma saying "Emma"
        wakes Emma.
        """
        return time.monotonic() < self._speaking_until

    def current_text(self) -> str:
        """The words being spoken right now.

        The listener compares what the microphone heard against this: anything
        Emma is in the middle of saying is her own voice coming back through
        the speakers, not the user talking.
        """
        return self._current_text

    def on_speaking_change(self, callback) -> None:
        self._listeners.append(callback)

    def _announce(self, speaking: bool) -> None:
        for callback in list(self._listeners):
            try:
                callback(speaking)
            except Exception:
                pass

    # -- the public way to speak -------------------------------------------
    def say(self, text: str, priority: int = Priority.RESULT, sensitive: bool = False) -> bool:
        """Queue something to say. Returns False when nothing will be spoken."""
        settings = get_settings()
        if not settings.tts_enabled:
            return False
        spoken = speakable(text)
        if not spoken:
            return False
        if len(spoken) > settings.tts_max_chars:
            spoken = spoken[: settings.tts_max_chars].rsplit(" ", 1)[0] + "."
        # The same line twice in a row is an echo, not information.
        if spoken == self._last_said and priority >= Priority.PROGRESS:
            return False
        self._last_said = spoken

        if priority >= Priority.PROGRESS and self._queue.qsize() >= settings.tts_progress_queue_limit:
            return False  # chatter is dropped when it would arrive too late to matter
        if priority <= Priority.URGENT:
            self.stop()   # an error interrupts whatever is being said

        with self._lock:
            self._sequence += 1
            self._queue.put(_Utterance(int(priority), self._sequence, spoken, sensitive))
        self._ensure_worker()
        return True

    def stop(self) -> None:
        """Cut the audio and forget everything queued."""
        with self._lock:
            self._generation += 1
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._silence()
        self._speaking_until = 0.0
        self._announce(False)

    def shutdown(self) -> None:
        self._running = False
        self.stop()

    # -- the worker ---------------------------------------------------------
    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker and self._worker.is_alive():
                return
            self._running = True
            self._worker = threading.Thread(target=self._run, name="emma-voice", daemon=True)
            self._worker.start()

    def _run(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        # One sentence is spoken while the next is being synthesized. Without
        # this the user hears a gap of a whole request between every sentence;
        # with it, only the first sentence ever waits.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="emma-voice-synth") as ahead:
            while self._running:
                try:
                    utterance = self._queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                generation = self._generation
                chunks = sentences(utterance.text)
                if not chunks:
                    continue

                self._current_text = utterance.text
                pending = ahead.submit(self._audio_for, chunks[0], utterance.sensitive)
                for index, chunk in enumerate(chunks):
                    if generation != self._generation:
                        break
                    audio, tier = pending.result()
                    if index + 1 < len(chunks):
                        pending = ahead.submit(self._audio_for, chunks[index + 1], utterance.sensitive)
                    if not audio:
                        logger.warning("tts_no_voice_available", chars=len(chunk))
                        continue
                    logger.debug("tts_speaking", tier=tier, chars=len(chunk))
                    self._announce(True)
                    self._play(audio, generation)
                self._current_text = ""
                if self._queue.empty():
                    self._announce(False)

    def _audio_for(self, text: str, sensitive: bool) -> tuple[bytes | None, str]:
        """Cached audio for one sentence, or freshly synthesized and then cached."""
        audio = self._cached(text, sensitive)
        if audio is not None:
            return audio, "cache"
        started = time.monotonic()
        audio, tier = self._voices.synthesize(text, sensitive)
        if audio:
            self._cache(text, sensitive, audio)
            logger.debug("tts_synthesis_ms", ms=int((time.monotonic() - started) * 1000), tier=tier)
        return audio, tier

    # -- cache --------------------------------------------------------------
    def _key(self, text: str, sensitive: bool) -> Path:
        settings = get_settings()
        voice = settings.tts_windows_voice if sensitive else settings.tts_groq_voice
        digest = hashlib.sha256(f"{voice}|{settings.tts_order}|{text}".encode()).hexdigest()[:32]
        return CACHE_DIR / f"{digest}.wav"

    def _cached(self, text: str, sensitive: bool) -> bytes | None:
        if not get_settings().tts_cache_enabled:
            return None
        path = self._key(text, sensitive)
        try:
            return path.read_bytes() if path.exists() else None
        except OSError:
            return None

    def _cache(self, text: str, sensitive: bool, audio: bytes) -> None:
        if not get_settings().tts_cache_enabled:
            return
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self._key(text, sensitive).write_bytes(audio)
        except OSError as e:
            logger.debug("tts_cache_write_failed", error=str(e)[:120])

    # -- playback -----------------------------------------------------------
    def _play(self, audio: bytes, generation: int) -> None:
        """Play WAV bytes, holding the microphone gate open until they finish."""
        seconds = self._duration(audio)
        # The gate closes a moment AFTER the audio ends, so the room's echo
        # cannot be heard as a wake word.
        self._speaking_until = time.monotonic() + seconds + SPEAKING_TAIL_SECONDS
        try:
            import winsound

            path = CACHE_DIR / f"_play_{os.getpid()}.wav"
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_bytes(audio)
            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if generation != self._generation:
                    self._silence()
                    return
                time.sleep(0.05)
        except Exception as e:
            logger.warning("tts_playback_failed", error=str(e)[:160])
            self._speaking_until = 0.0

    def _silence(self) -> None:
        try:
            import winsound

            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass

    @staticmethod
    def _duration(audio: bytes) -> float:
        """Seconds of audio, read from the WAV header (no library needed)."""
        try:
            rate = struct.unpack("<I", audio[24:28])[0]
            bits = struct.unpack("<H", audio[34:36])[0]
            channels = struct.unpack("<H", audio[22:24])[0]
            frame = max(1, (bits // 8) * channels)
            return max(0.2, (len(audio) - 44) / float(frame * max(1, rate)))
        except Exception:
            return 2.0


# Said at the end of almost every task. Synthesized once in the background at
# startup so the first spoken reply of the day is instant rather than a request.
COMMON_LINES = (
    "Done.",
    "I could not finish that.",
    "Working on it.",
    "Sorry, I could not think of an answer just now.",
)


def prewarm() -> None:
    """Fill the cache with the lines Emma says most, without blocking startup."""
    def _fill() -> None:
        speaker = get_speaker()
        for line in COMMON_LINES:
            try:
                if speaker._cached(line, sensitive=False) is None:
                    audio, tier = speaker._voices.synthesize(line, sensitive=False)
                    if audio:
                        speaker._cache(line, sensitive=False, audio=audio)
                        logger.debug("tts_prewarmed", line=line[:30], tier=tier)
            except Exception as e:
                logger.debug("tts_prewarm_failed", error=str(e)[:120])
                return

    if not get_settings().tts_enabled or not get_settings().tts_cache_enabled:
        return
    threading.Thread(target=_fill, name="emma-voice-prewarm", daemon=True).start()


_speaker: Speaker | None = None
_speaker_lock = threading.Lock()


def get_speaker() -> Speaker:
    global _speaker
    with _speaker_lock:
        if _speaker is None:
            _speaker = Speaker()
        return _speaker


def say(text: str, priority: int = Priority.RESULT, sensitive: bool = False) -> bool:
    """Speak something. Safe to call from anywhere, returns immediately."""
    return get_speaker().say(text, priority=priority, sensitive=sensitive)


def stop_speaking() -> None:
    get_speaker().stop()


def is_speaking() -> bool:
    return get_speaker().is_speaking()
