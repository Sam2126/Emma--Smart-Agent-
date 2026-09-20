"""
Hands-free wake word listener — always-on microphone for voice activation.

When the backend starts, this module spawns a daemon thread that continuously
listens through the system microphone.  The state machine is:

    IDLE  →  wake word detected ("emma")  →  RECORDING
    RECORDING  →  stop word detected ("done")  →  PROCESSING
    RECORDING  →  30 s without the stop word  →  IDLE (nothing is run)
    PROCESSING  →  transcribe + dispatch task  →  IDLE

Wake word / stop word spotting runs on Vosk by default: an offline recognizer
restricted to a grammar of the wake word, the stop word and their phonetic
variants (no internet, no rate limits). If the Vosk package or model is
missing it falls back to Google's free online recognizer automatically
(wake_word_engine="google" selects Google explicitly). Matching is fuzzy over
phonetic variants either way. The instruction itself is always transcribed in
one pass by Groq Whisper.

Audio feedback uses ``winsound`` on Windows (zero extra deps).
"""

from __future__ import annotations

import asyncio
import io
import os
import re
import sys
import threading
import time
import wave
import json
from pathlib import Path
from enum import Enum, auto
from typing import TYPE_CHECKING

import structlog

from app.speech_vocab import clean_instruction, whisper_prompt

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class ListenerState(Enum):
    IDLE = auto()       # Passively listening for wake word
    RECORDING = auto()  # Actively recording the user's instruction
    PROCESSING = auto() # Transcribing + dispatching


# ---------------------------------------------------------------------------
# Phonetic variants — Google often mishears "Shrey" as these
# ---------------------------------------------------------------------------

# How each supported wake word is commonly misheard or said. "emma" became the
# wake word on 2026-09-17 (the user's choice); "hello" is still supported.
_WAKE_VARIANTS = {
    "hello": {
        "halo", "helo", "hullo", "hallo",
        "hello.", "hello!", "hello,", "hello?",
        "hey hello", "oh hello", "hello hello",
    },
    "emma": {"ema", "hey emma", "hi emma", "ok emma", "okay emma", "emma emma"},
}


def _build_wake_variants(wake_word: str) -> set[str]:
    """
    Build a set of phonetic variants for the wake word so that fuzzy
    matching works even when Google's free STT mishears the word.
    """
    base = wake_word.lower().strip()
    return {base} | _WAKE_VARIANTS.get(base, set())


def _build_stop_variants(stop_word: str) -> set[str]:
    """
    Build a set of phrases that should trigger stop.

    Found in production: the free recognizer consistently misheard a
    clearly-spoken "done, done" as "dan dan" — a plain phonetic mishearing,
    the same class of error _build_wake_variants already accounts for on
    "hello" (halo, helo, hullo, ...). Because the old list here only had
    exact synonyms ("stop", "finish", ...) and no phonetic variants of
    "done" itself, that mishearing matched nothing, so the stop word was
    never detected and recording ran the full 30s timeout, capturing a lot
    of extra unrelated audio. Phonetic variants close that gap.
    """
    base = stop_word.lower().strip()
    variants = {base}

    if base == "done":
        variants.update({
            "done", "done done", "i'm done", "i am done",
            "that's it", "stop", "finish", "finished",
            "stop listening", "done now", "okay done",
            "ok done", "all done", "we're done", "we done",
            # Phonetic mishearings of "done" from the free recognizer
            "dan", "dan dan", "dun", "dun dun", "don", "don don",
            "doon", "done.", "done!", "done,", "done?",
        })

    return variants


# ---------------------------------------------------------------------------
# Console output helpers (bold, colored — so the user SEES state changes)
# ---------------------------------------------------------------------------

def _speak_result(success: bool, summary: str, error: str = "") -> None:
    """Speak a short spoken-sized version of how a task ended."""
    try:
        from app.tts import Priority, say

        spoken = (summary or "").strip()
        # A report written for the screen opens with headings and bullet lists;
        # spoken, only its first couple of sentences carry the news.
        parts = [p.strip() for p in spoken.replace("\n", " ").split(". ") if p.strip()]
        spoken = ". ".join(parts[:2])
        if not spoken:
            spoken = "Done." if success else f"I could not finish that. {error}".strip()
        say(spoken, priority=Priority.RESULT if success else Priority.URGENT)
    except Exception as e:
        logger.debug("tts_say_failed", error=str(e)[:120])


# What a person says to make Emma stop talking. Deliberately short and plain:
# they are said over her voice, so they must survive a noisy recording.
INTERRUPT_WORDS = (
    "stop", "quiet", "shut up", "enough", "cancel", "wait", "shh", "hush",
)


def _is_interruption(heard: str, being_said: str) -> bool:
    """True when the user told Emma to be quiet - and it was not her own echo.

    Emma's own words come back through the microphone a fraction of a second
    later. If what was heard appears in the sentence she is speaking, it is
    that echo: "I have stopped the download" must not stop her.
    """
    heard = (heard or "").strip().lower()
    if not heard:
        return False
    spoken = (being_said or "").lower()
    words = [w for w in re.findall(r"[a-z]+", heard) if w]
    if not words:
        return False
    # Anything she is saying right now is echo, whatever it contains. Whole
    # words only: "stop" is not an echo of her saying "stopped", and treating
    # it as one would make her impossible to interrupt.
    spoken_words = set(re.findall(r"[a-z]+", spoken))
    if all(word in spoken_words for word in words):
        return False
    return any(phrase in heard for phrase in INTERRUPT_WORDS)


def _voice_is_speaking() -> bool:
    """True while Emma's own voice is coming out of the speakers."""
    try:
        from app.tts import is_speaking

        return is_speaking()
    except Exception:
        return False


def _voice_current_text() -> str:
    """The sentence Emma is speaking right now, or empty."""
    try:
        from app.tts import get_speaker

        return get_speaker().current_text()
    except Exception:
        return ""


def _stop_voice() -> None:
    try:
        from app.tts import stop_speaking

        stop_speaking()
    except Exception:
        pass


def _print_status(msg: str, color: str = "cyan") -> None:
    """Print a prominent status line to the console."""
    colors = {
        "cyan": "\033[96m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "magenta": "\033[95m",
        "reset": "\033[0m",
        "bold": "\033[1m",
    }
    c = colors.get(color, "")
    r = colors["reset"]
    b = colors["bold"]
    try:
        print(f"{b}{c}🎤 {msg}{r}", flush=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Audio feedback (Windows beep — zero deps)
# ---------------------------------------------------------------------------

def _beep_start():
    """Short rising beep: wake word detected, now recording."""
    try:
        import winsound
        winsound.Beep(880, 200)   # A5 for 200ms
        winsound.Beep(1100, 150)  # ~C#6 for 150ms
    except Exception:
        pass


def _beep_stop():
    """Short falling beep: recording stopped, processing."""
    try:
        import winsound
        winsound.Beep(1100, 150)
        winsound.Beep(880, 200)
    except Exception:
        pass


def _beep_error():
    """Low buzz: something went wrong."""
    try:
        import winsound
        winsound.Beep(300, 400)
    except Exception:
        pass


_last_console_step = {"text": ""}


def _console_progress(update: dict) -> None:
    """Print the agent's live steps for wake-word tasks, which have no UI."""
    if update.get("status") not in ("acting", "replanning", "thinking", "waiting_confirmation"):
        return
    step = (update.get("current_step") or "").strip()
    if step and step != _last_console_step["text"]:
        _last_console_step["text"] = step
        _print_status(f"   {step}", "cyan")


def _ui_broadcaster():
    """A function that sends a protocol message to every connected UI; never raises."""
    try:
        # uvicorn imports the app as "app.main"; that module owns the running server.
        from app.main import ws_server
    except Exception:
        ws_server = None

    def send(message) -> None:
        if ws_server is None:
            return
        try:
            task = asyncio.get_running_loop().create_task(ws_server.broadcast(message))
        except Exception as e:
            logger.debug("wake_broadcast_failed", error=str(e)[:120])
            return
        # The loop holds tasks only weakly; keep this one until it has run.
        _pending_broadcasts.add(task)
        task.add_done_callback(_pending_broadcasts.discard)

    return send


_pending_broadcasts: set = set()


def _beep_feedback(rating: int):
    """Two short tones: rising for a 👍 saved, falling for a 👎 saved."""
    try:
        import winsound
        tones = (700, 1000) if rating > 0 else (700, 450)
        for tone in tones:
            winsound.Beep(tone, 120)
    except Exception:
        pass


def _beep_task_done():
    """Triple rising beep: task completed successfully."""
    try:
        import winsound
        winsound.Beep(660, 100)
        winsound.Beep(880, 100)
        winsound.Beep(1100, 150)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Wake word listener (runs in a background thread)
# ---------------------------------------------------------------------------

# --- Vosk grammar -----------------------------------------------------------
# Measured on 36 real human recordings (tests/fixtures/voice, Lingua Libre):
#   grammar = wake/stop words + their phonetic spellings: every "hello" and
#     "done" detected, but 11 of 24 look-alike words triggered ("yellow",
#     "fellow", "hollow" -> "hello"; "dawn", "gone", "none" -> "don"/"dun")
#   no grammar (full vocabulary): 1 of 24 triggered, 1 "done" missed, 3x slower
#   grammar = real words + decoys below: 6/6 "hello", 6/6 "done", 0 of 24
#     triggered, ~22 ms per clip
# A closed grammar forces every sound onto its nearest word; decoys give
# look-alike sounds somewhere else to land.

# Spellings that only exist to catch Google's mishearings. In a Vosk grammar
# they attract look-alike words instead, so the grammar uses real words only.
_VOSK_SKIP_VARIANTS = {"dan", "dun", "don", "doon", "halo", "hullo", "helo", "hallo", "ema"}

# Words that sound like each wake or stop word. "emma": measured on real
# recordings on 2026-09-17, see tests/test_wake_word_emma.py.
_VOSK_DECOYS = {
    "hello": ("yellow", "fellow", "hollow", "follow", "below", "allow", "hotel", "help", "hell", "hi", "high", "low"),
    "done": ("dawn", "gone", "none", "down", "don't", "dune", "fun", "one", "won", "son", "sun", "ton", "dumb", "dance"),
    # "hammer" is deliberately absent: with it, one speaker's "Emma" was heard as "hammer".
    "emma": ("anna", "gemma", "emily", "ember", "enemy", "summer", "comma", "amber", "mama"),
}


def _build_stop_core(stop_word: str) -> set[str]:
    """
    The stop word itself plus its phonetic mishearings — deliberately NOT
    synonyms like "stop" or "finish", which also occur in real instructions.
    Used for the "said it repeatedly" rule in _matches_stop_word.
    """
    base = stop_word.lower().strip()
    core = {base}
    if base == "done":
        core.update({"dan", "dun", "don", "doon"})
    return core


def _truncate_at_stop_sentence(text: str, stop_variants: set[str]) -> str:
    """
    Cut a transcript at the first sentence made up ONLY of stop phrases.

    Found in production: Whisper returned "Open WhatsApp and search Rakesh
    and say hello to him. Done. Done. Done. Done. I read done. Go and eat
    done. Done done" because the user kept repeating "done" until recording
    stopped. Stripping one trailing stop word left all of that in the task
    instruction. The first "Done." sentence marks where the instruction
    ended, so everything from there on is dropped.
    """
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept: list[str] = []
    for sentence in sentences:
        words = [w.strip(".,!?'\"").lower() for w in sentence.split()]
        words = [w for w in words if w]
        if words:
            joined = " ".join(words)
            if joined in stop_variants or all(w in stop_variants for w in words):
                break
        kept.append(sentence)
    return " ".join(kept).strip()


class WakeWordListener:
    """
    Always-on microphone listener that detects a wake word, records the
    instruction until a stop word, then dispatches the task.

    Usage::

        listener = WakeWordListener(
            wake_word="shrey",
            stop_word="done",
            listen_timeout=30,
            loop=asyncio.get_event_loop(),
        )
        listener.start()   # spawns daemon thread
        ...
        listener.stop()    # clean shutdown
    """

    def __init__(
        self,
        wake_word: str = "shrey",
        stop_word: str = "done",
        listen_timeout: int = 30,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        self.wake_word = wake_word.lower().strip()
        self.stop_word = stop_word.lower().strip()
        self.listen_timeout = listen_timeout
        self._loop = loop
        self._state = ListenerState.IDLE
        self._stop_event = threading.Event()
        # Set by the "Talk to Emma" button: start recording at once, without
        # waiting to hear the wake word.
        self._talk_requested = threading.Event()
        self._thread: threading.Thread | None = None

        # Pre-build variant sets for fuzzy matching
        self._wake_variants = _build_wake_variants(self.wake_word)
        self._stop_variants = _build_stop_variants(self.stop_word)
        self._stop_core = _build_stop_core(self.stop_word)
        # Vosk hears some speakers' "done" as "don't" (verified on a real
        # recording that sits on the edge between the two). Only a phrase that
        # is exactly "don't" counts, never "don't" inside a sentence.
        self._stop_whole_phrase_only = {"don't"} if self.stop_word == "done" else set()
        # Optional offline keyword engine (wake_word_engine=vosk)
        self._vosk_model = None
        self._vosk_grammar = ""
        self._vosk_recognizer = None
        self.engine_in_use = "google"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the listener in a background daemon thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("wake_listener_already_running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="wake-word-listener",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "wake_listener_started",
            wake_word=self.wake_word,
            stop_word=self.stop_word,
            timeout=self.listen_timeout,
        )

    def stop(self) -> None:
        """Signal the listener to shut down."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("wake_listener_stopped")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Core loop (runs in thread)
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Main listener loop — runs until stop_event is set."""
        try:
            import speech_recognition as sr
        except ImportError:
            logger.error(
                "wake_listener_import_error",
                hint="Install SpeechRecognition: pip install SpeechRecognition PyAudio",
            )
            return

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 1.0

        try:
            mic = sr.Microphone()
        except Exception as e:
            logger.error("wake_listener_mic_error", error=str(e)[:200])
            _beep_error()
            return

        self._load_vosk_if_configured()

        # Brief ambient noise calibration
        logger.info("wake_listener_calibrating")
        _print_status("Calibrating microphone... (2 seconds of silence please)", "yellow")
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=2)

        logger.info(
            "wake_listener_ready",
            energy_threshold=int(recognizer.energy_threshold),
            hint=f"Say '{self.wake_word}' to activate",
        )
        _print_status(
            f"READY! Say \"{self.wake_word.upper()}\" to activate. "
            f"Listening variants: {', '.join(sorted(self._wake_variants)[:8])}...",
            "green",
        )

        while not self._stop_event.is_set():
            try:
                self._idle_listen(recognizer, mic)
            except Exception as e:
                logger.error("wake_listener_cycle_error", error=str(e)[:200])
                time.sleep(2)

    # ------------------------------------------------------------------
    # IDLE phase — listen for wake word
    # ------------------------------------------------------------------

    def _listen_for_interruption(self, recognizer, mic) -> None:
        """Listen, while Emma talks, only for a word that tells her to stop."""
        import speech_recognition as sr

        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=1.0, phrase_time_limit=1.5)
        except sr.WaitTimeoutError:
            return
        except Exception:
            return
        if not _voice_is_speaking():
            return  # she finished while this was recording; nothing to stop

        heard = self._quick_recognize(recognizer, audio)
        if not heard:
            return
        being_said = _voice_current_text()
        if _is_interruption(heard, being_said):
            logger.info("voice_interrupted_by_user", heard=heard[:60])
            _print_status(f'Stopping - heard "{heard}"', "yellow")
            _stop_voice()

    def request_conversation(self) -> bool:
        """Start listening now, as though the wake word had just been heard.

        This is what the desktop app's "Talk to Emma" button calls: pressing a
        button is easier than saying a word, and a reply that comes back
        spoken makes it a conversation rather than a command line.
        """
        if self._stop_event.is_set():
            return False
        self._talk_requested.set()
        logger.info("conversation_requested")
        return True

    def _idle_listen(self, recognizer, mic) -> None:
        """Listen for the wake word in short bursts."""
        import speech_recognition as sr

        self._state = ListenerState.IDLE

        # The button was pressed: skip the wake word entirely and record.
        if self._talk_requested.is_set():
            self._talk_requested.clear()
            _stop_voice()                       # she stops talking to listen
            _beep_start()
            _print_status("Listening - speak now, say \"done\" when finished", "magenta")
            self._record_instruction(recognizer, mic, initial_text="", wake_audio=None)
            return

        # While Emma is speaking the microphone hears HER: she says the word
        # "Emma" in her own replies, and left alone she would wake herself
        # forever. But going deaf while she talks means she cannot be
        # interrupted either, and a voice you cannot stop is worse than one you
        # cannot start. So during speech she listens for one thing only - a
        # word telling her to be quiet - and treats anything she is in the
        # middle of saying as her own echo.
        if _voice_is_speaking():
            self._listen_for_interruption(recognizer, mic)
            return

        with mic as source:
            try:
                audio = recognizer.listen(
                    source,
                    timeout=5,
                    phrase_time_limit=2,  # SHORT — only catch quick "hello", not long sentences
                )
            except sr.WaitTimeoutError:
                return

        # Audio captured WHILE Emma was talking is her own voice arriving late:
        # the burst above takes up to two seconds, and she may have started
        # speaking inside it.
        if _voice_is_speaking():
            return

        # Quick local recognition to check for wake word
        text = self._quick_recognize(recognizer, audio)
        if not text:
            return

        logger.debug("wake_listener_heard", text=text, state="IDLE")

        # Check if this is a STANDALONE wake word (not buried in a long sentence)
        wake_match = self._matches_wake_word(text)
        if wake_match:
            logger.info("wake_word_detected", text=text)
            _print_status(f"WAKE WORD DETECTED! Heard: \"{text}\"", "magenta")
            _beep_start()

            # If the user said "hello open whatsapp", extract "open whatsapp"
            # as the initial instruction. If they just said "hello", initial is empty.
            instruction_part = self._extract_instruction_after_wake(text)
            self._record_instruction(recognizer, mic, initial_text=instruction_part, wake_audio=audio)
        elif self._sounds_like_wake_attempt(text):
            # Only near-misses are worth showing. The offline grammar answers
            # with decoy words ("dune", "dawn", "sun") for ordinary room noise,
            # and printing each one filled the console.
            _print_status(f"[idle] Heard: \"{text}\" (not the wake word)", "cyan")

    # ------------------------------------------------------------------
    # Wake word matching — STRICT standalone detection
    # ------------------------------------------------------------------

    def _matches_wake_word(self, text: str) -> bool:
        """
        Check if the recognized text IS the wake word (standalone).

        STRICT rules to prevent false triggers: the wake word must be the
        FIRST word ("emma open chrome" → yes, "open chrome emma" → no), or the
        first two words must be a variant ("hey emma"). This keeps "search
        Rakesh and say hello" from triggering.

        Found 2026-09-17: a two-word phrase used to match on either word, so
        room noise heard as "amber emma" woke the agent and a conversation
        was recorded.
        """
        lower = text.lower().strip()
        words = lower.split()

        if not words:
            return False

        # "emma open whatsapp" → YES (wake word first, rest is instruction)
        # "open whatsapp and say emma" / "amber emma" → NO
        first_word = words[0].strip(".,!?'\"")
        if first_word in self._wake_variants:
            return True

        # Also check if the first two words form a wake variant
        # e.g., "hey hello open chrome"
        first_two = " ".join(words[:2]).strip(".,!?'\"")
        if first_two in self._wake_variants:
            return True

        return False

    def _sounds_like_wake_attempt(self, text: str) -> bool:
        """True when what was heard contains a wake-word variant somewhere."""
        words = {word.strip(".,!?'\"") for word in (text or "").lower().split()}
        return bool(words & self._wake_variants)

    def _extract_instruction_after_wake(self, text: str) -> str:
        """
        Extract instruction text that comes AFTER the wake word.

        "hello" → "" (no instruction)
        "hello open whatsapp" → "open whatsapp"
        "hey hello search amazon" → "search amazon"
        """
        lower = text.lower()
        words = text.split()

        if len(words) <= 1:
            return ""

        # Find the wake word position and return everything after it
        for i, word in enumerate(words):
            clean = word.lower().strip(".,!?'\"")
            if clean in self._wake_variants:
                # Return everything after this word
                remaining = " ".join(words[i + 1:]).strip()
                return remaining

        return ""

    # ------------------------------------------------------------------
    # Stop word matching — fuzzy
    # ------------------------------------------------------------------

    def _matches_stop_word(self, text: str) -> bool:
        """
        Check if the recognized text indicates the user wants to stop.

        STRICT rules to prevent false triggers — mirrors _matches_wake_word.
        The previous version matched a stop variant occurring ANYWHERE as a
        substring of the heard text, so a real instruction like "stop the
        alarm and open notepad" would abort recording immediately on
        "stop the alarm" alone, well before the actual instruction was
        finished. Now the stop word must be the whole utterance, or the
        LAST word/two words — i.e. how a stop cue is actually spoken, at
        the end of what the user says, not incidentally in the middle.
        """
        lower = text.lower().strip()
        if not lower:
            return False

        # Whole utterance is a stop phrase (e.g. just "done").
        if lower in self._stop_variants or lower.strip('.,!?"') in self._stop_whole_phrase_only:
            return True

        words = lower.split()
        if not words:
            return False

        # The stop cue is naturally spoken LAST ("...okay I'm done").
        last_word = words[-1].strip(".,!?'\"")
        if last_word in self._stop_variants:
            return True

        if len(words) >= 2:
            last_two = " ".join(w.strip(".,!?'\"") for w in words[-2:])
            if last_two in self._stop_variants:
                return True

        # Said repeatedly. Live chunks merge continuous speech, so a user
        # saying "done" over and over produces chunks like "done I read done
        # go and eat" that end on some other word. Found in production: this
        # let recording run to the 30s timeout. Two or more occurrences of
        # the core stop word itself is an unambiguous stop signal.
        core_hits = sum(1 for w in words if w.strip(".,!?'\"") in self._stop_core)
        if core_hits >= 2:
            return True

        return False

    # ------------------------------------------------------------------
    # RECORDING phase — capture everything until stop word
    # ------------------------------------------------------------------

    def _record_instruction(self, recognizer, mic, initial_text: str = "", wake_audio=None) -> None:
        """
        Record raw audio until the stop word is heard or timeout, then
        transcribe the WHOLE recording in ONE accurate pass through Groq
        Whisper — the same reliable pipeline already used by the extension
        and Local Agent UI's voice button (see app/voice.py).

        The previous version stitched together Google's free-API text from
        each short chunk individually. That API frequently returns empty
        ("speech not recognized") or badly garbled text for short isolated
        chunks and loses words at chunk boundaries — this is what produced
        instructions like "turn propose nahin kar raha hai" in practice.
        Google's free recognizer is still used per-chunk, but now ONLY as a
        fast, low-stakes cue for *when* to stop listening — never as the
        actual transcribed content.
        """
        import speech_recognition as sr

        self._state = ListenerState.RECORDING

        audio_chunks: list[bytes] = []
        fallback_text_parts: list[str] = []  # used only if Whisper fails
        chunk_sample_rate: int | None = None
        chunk_sample_width: int | None = None
        start_time = time.time()

        # Keep the wake phrase's own audio. When the user says the whole
        # command in one breath ("hello open WhatsApp and ..."), the words
        # after "hello" are in this chunk; previously only the free
        # recognizer's rough text of them survived, so Whisper never heard them.
        if wake_audio is not None:
            chunk_sample_rate = wake_audio.sample_rate
            chunk_sample_width = wake_audio.sample_width
            audio_chunks.append(wake_audio.frame_data)

        # If the initial phrase already contained text beyond the wake word
        # (e.g. "hello done" said as one breath), short-circuit on that —
        # its audio belongs to the IDLE-phase chunk and wasn't buffered, so
        # this one check still relies on the free recognizer's text, but
        # it's just a no-op guard, not the primary transcription path.
        initial_after_wake = self._strip_wake_word(initial_text)
        if initial_after_wake.strip() and self._matches_stop_word(initial_after_wake):
            _print_status("Stop word detected immediately — nothing to do", "yellow")
            return

        _print_status(
            f"🔴 RECORDING — speak your instruction, say \"{self.stop_word.upper()}\" when finished "
            f"(timeout: {self.listen_timeout}s)",
            "red",
        )

        # Keep the mic open for the ENTIRE recording session so we never
        # miss a short word like "done" between open/close cycles.
        old_pause = recognizer.pause_threshold
        recognizer.pause_threshold = 0.8  # Detect pauses faster during recording

        stop_detected = False

        with mic as source:
            while not self._stop_event.is_set():
                elapsed = time.time() - start_time
                if elapsed > self.listen_timeout:
                    # No stop word: never run what was recorded. Found 2026-09-17:
                    # "Emma" said in a conversation started a recording, and 30 s of
                    # people talking was run as a task.
                    logger.warning("wake_listener_timeout_discarded", seconds=self.listen_timeout)
                    _print_status(
                        f"⏰ No \"{self.stop_word}\" within {self.listen_timeout}s — nothing was run",
                        "yellow",
                    )
                    _beep_error()
                    break

                try:
                    audio = recognizer.listen(
                        source,
                        timeout=5,
                        phrase_time_limit=6,
                    )
                except sr.WaitTimeoutError:
                    remaining = int(self.listen_timeout - elapsed)
                    _print_status(f"[recording] Waiting for speech... ({remaining}s left)", "yellow")
                    continue

                # Skip near-silent blips (background noise, breathing, a
                # chair creak). These are exactly the audio that gets
                # randomly misheard by the free recognizer as "done" and
                # cut a real instruction off early — verified in practice.
                min_bytes = int(audio.sample_rate * audio.sample_width * 0.3)
                if len(audio.frame_data) < min_bytes:
                    continue

                if chunk_sample_rate is None:
                    chunk_sample_rate = audio.sample_rate
                    chunk_sample_width = audio.sample_width
                audio_chunks.append(audio.frame_data)

                # Cheap, low-latency check for the stop cue — accuracy of
                # the recognized text doesn't matter here, only whether the
                # stop word was said, so a quick free-API check is fine.
                quick_text = self._quick_recognize(recognizer, audio)
                if not quick_text:
                    _print_status("[recording] (captured audio, still listening)", "yellow")
                    continue

                logger.info("wake_listener_chunk", text=quick_text, state="RECORDING")
                _print_status(f"[recording] Heard: \"{quick_text}\"", "magenta")

                if self._matches_stop_word(quick_text):
                    logger.info("stop_word_detected", text=quick_text)
                    _print_status(f"✅ STOP WORD DETECTED! Heard: \"{quick_text}\"", "green")
                    _beep_stop()
                    stop_detected = True
                    break
                fallback_text_parts.append(quick_text)

        # Restore original pause threshold
        recognizer.pause_threshold = old_pause

        if not stop_detected or not audio_chunks:
            if stop_detected:
                logger.warning("wake_listener_no_audio_captured")
                _print_status("❌ No audio captured — going back to listening", "red")
            return

        # --- One accurate transcription of the FULL recording ---
        _print_status("🧠 Transcribing your instruction (Groq Whisper)...", "cyan")
        wav_bytes = self._audio_chunks_to_wav(audio_chunks, chunk_sample_rate, chunk_sample_width)
        transcript = self._transcribe_via_whisper(wav_bytes)
        if transcript.strip():
            # The wake phrase's audio is part of the recording, so Whisper
            # heard it too - often as something else entirely, because it is
            # clipped: a real run returned "M R Open Chrome Search Mentra Done"
            # for "Emma, open Chrome, search Myntra, done". This strips the
            # wake word however it was heard, and repairs names it knows.
            full_instruction = clean_instruction(
                transcript.strip(),
                wake_variants=self._wake_variants,
                stop_variants=self._stop_variants,
                extra_names=get_settings().speech_extra_names,
            )
            full_instruction = self._strip_wake_word(full_instruction)
        else:
            # Whisper failed outright (network/API issue) — fall back to
            # whatever the free recognizer caught rather than losing the
            # instruction entirely.
            fallback = " ".join(
                w for w in f"{initial_after_wake} {' '.join(fallback_text_parts)}".split() if w != "[unk]"
            )
            if fallback:
                logger.warning("wake_listener_whisper_empty_using_fallback", fallback=fallback[:200])
            full_instruction = fallback
        # The recording legitimately ends with the user saying the stop
        # word — strip it back off so it doesn't pollute the instruction.
        full_instruction = _truncate_at_stop_sentence(full_instruction, self._stop_variants)
        full_instruction = self._strip_stop_word(full_instruction).strip()

        if not full_instruction:
            logger.warning("wake_listener_empty_instruction")
            _print_status("❌ No instruction captured — going back to listening", "red")
            _beep_error()
            return

        # --- PROCESSING phase ---
        self._state = ListenerState.PROCESSING
        from app.voice_feedback import parse_voice_feedback

        feedback = parse_voice_feedback(full_instruction)
        if feedback is not None:
            # "hello, good job, done" rates the last task instead of running one.
            _print_status(f"🗳️ FEEDBACK: \"{full_instruction}\"", "green")
            self._dispatch_feedback(feedback, full_instruction)
            return
        _print_status(f"🚀 EXECUTING: \"{full_instruction}\"", "green")
        logger.info("wake_listener_dispatching", instruction=full_instruction)
        self._dispatch_task(full_instruction)

    # ------------------------------------------------------------------
    # Accurate transcription — reuses the same Groq Whisper pipeline as
    # the extension / Local Agent UI's voice button (app/voice.py).
    # ------------------------------------------------------------------

    @staticmethod
    def _audio_chunks_to_wav(chunks: list[bytes], sample_rate: int, sample_width: int) -> bytes:
        """Concatenate raw PCM frames from multiple speech_recognition
        AudioData chunks — all captured from the same Microphone source
        within one `with mic as source:` block, so their format is
        guaranteed consistent — into a single playable WAV file."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(chunks))
        return buf.getvalue()

    def _transcribe_via_whisper(self, wav_bytes: bytes) -> str:
        """Transcribe the full recording via Groq Whisper, run from this
        background thread through the backend's asyncio event loop."""
        from app.voice import transcribe_audio

        if not self._loop:
            logger.error("wake_listener_no_event_loop_for_transcription")
            return ""
        try:
            future = asyncio.run_coroutine_threadsafe(
                transcribe_audio(
                    wav_bytes,
                    filename="wake_instruction.wav",
                    language="en",
                    prompt=whisper_prompt(get_settings().speech_extra_names),
                ),
                self._loop,
            )
            return future.result(timeout=60)
        except Exception as e:
            logger.error("wake_listener_whisper_failed", error=str(e)[:200])
            _print_status(f"⚠️ Accurate transcription failed ({str(e)[:80]}), using fallback", "yellow")
            return ""

    # ------------------------------------------------------------------
    # Dispatch — send instruction to the agent engine (AgentRunner)
    # ------------------------------------------------------------------

    def _dispatch_task(self, instruction: str) -> None:
        """Send the transcribed instruction to AgentRunner for execution."""
        if not self._loop:
            logger.error("wake_listener_no_event_loop")
            _beep_error()
            return

        asyncio.run_coroutine_threadsafe(
            self._async_dispatch(instruction),
            self._loop,
        )

    def _dispatch_feedback(self, feedback, spoken: str) -> None:
        if not self._loop:
            logger.error("wake_listener_no_event_loop")
            _beep_error()
            return
        asyncio.run_coroutine_threadsafe(self._async_feedback(feedback, spoken), self._loop)

    async def _async_feedback(self, feedback, spoken: str) -> None:
        """Store a spoken 👍 / 👎 on the last task and show it in every window."""
        from app.voice_feedback import record_voice_feedback
        from app.websocket.protocol import FeedbackRecordedMessage, VoiceTranscriptMessage

        try:
            result = await record_voice_feedback(feedback)
        except Exception as e:
            logger.error("voice_feedback_failed", error=str(e)[:200])
            result = {"applied": False, "queued": False, "task_id": "", "message": f"Feedback could not be saved: {e}"}
        broadcast = _ui_broadcaster()
        broadcast(VoiceTranscriptMessage(text=spoken, scope="local"))
        saved = bool(result.get("applied") or result.get("queued"))
        if result.get("task_id"):
            broadcast(FeedbackRecordedMessage(
                task_id=result["task_id"],
                rating=feedback.rating,
                applied=bool(result.get("applied")),
                queued=bool(result.get("queued")),
                message=result.get("message", ""),
            ))
        logger.info(
            "voice_feedback",
            task_id=result.get("task_id", ""),
            rating=feedback.rating,
            saved=saved,
            note=feedback.note[:120],
        )
        message = result.get("message", "")
        if saved:
            thumb = "👍" if feedback.rating > 0 else "👎"
            about = result.get("instruction", "")[:80]
            _print_status(f"{thumb} Feedback saved for \"{about}\". {message}", "green")
            _beep_feedback(feedback.rating)
        else:
            _print_status(f"❌ {message}", "red")
            _beep_error()

    async def _async_dispatch(self, instruction: str) -> None:
        """Async coroutine that runs the task on the agent engine."""
        import uuid
        from app.agent.runner import AgentRunner
        from app.websocket.server import _resolve_voice_scope

        task_id = str(uuid.uuid4())
        scope = _resolve_voice_scope(instruction, "local")

        logger.info(
            "wake_task_started",
            task_id=task_id,
            instruction=instruction,
            scope=scope,
        )

        if scope == "browser":
            try:
                from app.main import ensure_browser_connection
                await ensure_browser_connection()
            except Exception as e:
                logger.warning("wake_task_browser_connect_failed", error=str(e)[:150])

        from app.websocket.protocol import StatusUpdateMessage, TaskCompleteMessage, VoiceTranscriptMessage

        # Stream steps to the console AND to every open window (desktop app,
        # Local Agent page, extension), so a hands-free task shows up there with
        # its explanation and 👍/👎 buttons. There is no confirmation channel:
        # irreversible browser actions are refused.
        broadcast = _ui_broadcaster()
        broadcast(VoiceTranscriptMessage(text=instruction, scope=scope))

        def _on_status(update: dict) -> None:
            _console_progress(update)
            broadcast(StatusUpdateMessage(
                task_id=task_id,
                status=update.get("status", "acting"),
                current_step=update.get("current_step", ""),
                progress=update.get("progress", 0.5),
                details=update.get("details", ""),
            ))

        runner = AgentRunner(status_callback=_on_status)

        try:
            result = await runner.run_task(
                instruction=instruction,
                task_id=task_id,
                scope=scope,
            )
            success = result.get("success", False)
            summary = result.get("summary", "")
            broadcast(TaskCompleteMessage(
                task_id=task_id,
                success=success,
                summary=summary or ("Task completed" if success else "Task failed"),
                error=result.get("error", ""),
                duration_seconds=result.get("duration_seconds", 0.0),
                explanation=result.get("explanation", ""),
                retried=bool(result.get("retried")),
            ))
            logger.info(
                "wake_task_completed",
                task_id=task_id,
                success=success,
                summary=summary[:200] if summary else "",
            )
            # Said out loud, not read out: one or two sentences about what
            # happened, while the full report stays on the screen.
            _speak_result(success, summary, result.get("error", ""))
            explanation = result.get("explanation", "")
            if explanation:
                for line in explanation.splitlines():
                    _print_status(f"   {line}", "cyan")
            if success:
                _print_status(f"✅ Task completed: {summary[:100]}", "green")
                _beep_task_done()
            else:
                error = result.get("error", "")
                _print_status(f"❌ Task failed: {error[:100]}", "red")
                _beep_error()
        except Exception as e:
            logger.error("wake_task_failed", task_id=task_id, error=str(e)[:200])
            _print_status(f"❌ Task error: {str(e)[:100]}", "red")
            _beep_error()
            broadcast(TaskCompleteMessage(task_id=task_id, success=False, summary="Task failed", error=str(e)[:300]))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _quick_recognize(self, recognizer, audio) -> str:
        """
        Keyword spotting for the wake and stop words. Returns "" on failure.

        Engine "vosk" (wake_word_engine=vosk): an offline Vosk recognizer
        restricted to a grammar of the wake word, the stop word and their
        variants, so it can only ever answer with those words or nothing — no
        internet, no rate limits. Engine "google": Google's free online
        recognizer. The spoken instruction itself is transcribed by Whisper
        either way; this only decides when to start and stop listening.
        """
        if self._vosk_model is not None:
            text = self._recognize_vosk(audio)
            if text is not None:
                return text
        try:
            return recognizer.recognize_google(audio)
        except Exception:
            return ""

    def _load_vosk_if_configured(self) -> None:
        from app.config import get_settings

        settings = get_settings()
        if (settings.wake_word_engine or "google").strip().lower() != "vosk":
            return
        model_path = Path(settings.vosk_model_path)
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parents[1] / model_path
        if not model_path.exists():
            logger.warning("vosk_model_missing", path=str(model_path), hint="python scripts/install_vosk_model.py")
            _print_status("Vosk model not found — using Google's recognizer for the wake word", "yellow")
            return
        try:
            from vosk import KaldiRecognizer, Model, SetLogLevel

            SetLogLevel(-1)
            model = Model(str(model_path))
            real_words = {
                word
                for phrase in (self._wake_variants | self._stop_variants)
                for word in phrase.strip(".,!?").split()
                if word and word not in _VOSK_SKIP_VARIANTS
            }
            decoys = {
                word
                for key in (self.wake_word, self.stop_word)
                for word in _VOSK_DECOYS.get(key, ())
                if word not in real_words
            }
            # Only words in the model's vocabulary: unknown ones can never be
            # recognized, and Kaldi printed a warning for each of them every
            # time a recognizer was built.
            find_word = getattr(model, "find_word", None) or getattr(model, "vosk_model_find_word", None)
            words = [w for w in sorted(real_words | decoys) if find_word is None or find_word(w) >= 0]
            self._vosk_grammar = json.dumps(words + ["[unk]"])
            # One recognizer, reset per phrase, instead of compiling the
            # grammar again for every chunk of audio.
            self._vosk_recognizer = KaldiRecognizer(model, 16000, self._vosk_grammar)
            self._vosk_model = model
            self.engine_in_use = "vosk"
            logger.info("vosk_wake_engine_ready", words=len(words))
            _print_status("Wake word engine: Vosk (offline)", "green")
        except Exception as e:
            self._vosk_model = None
            logger.warning("vosk_load_failed", error=str(e)[:200])
            _print_status("Vosk failed to load — using Google's recognizer for the wake word", "yellow")

    def _recognize_vosk(self, audio) -> str | None:
        """Grammar-restricted offline recognition. None means "engine failed"."""
        try:
            rec = self._vosk_recognizer
            if rec is None:
                from vosk import KaldiRecognizer

                rec = self._vosk_recognizer = KaldiRecognizer(self._vosk_model, 16000, self._vosk_grammar)
            rec.Reset()
            rec.AcceptWaveform(audio.get_raw_data(convert_rate=16000, convert_width=2))
            words = json.loads(rec.FinalResult()).get("text", "").split()
            # Keep the "[unk]" placeholders: they preserve WHERE the wake or
            # stop word sat in a longer phrase, which the standalone-word rules
            # in _matches_wake_word / _matches_stop_word depend on. Dropping
            # them would turn "open WhatsApp and say hello to him" into a bare
            # "hello" and trigger the wake word mid-sentence.
            if all(w == "[unk]" for w in words):
                return ""
            return " ".join(words)
        except Exception as e:
            logger.warning("vosk_recognition_failed", error=str(e)[:150])
            return None

    def _strip_wake_word(self, text: str) -> str:
        """
        Remove a LEADING wake word from text, keeping everything after.

        Only strips from the front — mirroring _matches_wake_word, which
        only recognizes the wake word there. The previous version searched
        for the EARLIEST occurrence of any variant anywhere in the text,
        which could cut into real instruction content (e.g. a wake variant
        substring appearing later in the sentence) instead of leaving it
        alone when the wake word isn't genuinely leading the phrase.
        """
        words = text.split()
        if not words:
            return text

        first = words[0].lower().strip(".,!?'\"")
        if first in self._wake_variants:
            return " ".join(words[1:]).strip(" ,.")

        if len(words) >= 2:
            first_two = " ".join(w.lower().strip(".,!?'\"") for w in words[:2])
            if first_two in self._wake_variants:
                return " ".join(words[2:]).strip(" ,.")

        return text

    def _strip_stop_word(self, text: str) -> str:
        """
        Remove a TRAILING stop word from text, keeping everything before it.

        Only strips from the end — mirroring _matches_stop_word, which
        only recognizes the stop word there. The previous version searched
        for the EARLIEST occurrence of any variant anywhere in the text, so
        a real instruction merely containing the word "stop" (e.g. "stop
        the alarm and open notepad") had everything after it chopped off,
        even though "stop" wasn't the actual stop cue.
        """
        words = text.split()
        if not words:
            return text

        last = words[-1].lower().strip(".,!?'\"")
        if last in self._stop_variants:
            return " ".join(words[:-1]).strip(" ,.")

        if len(words) >= 2:
            last_two = " ".join(w.lower().strip(".,!?'\"") for w in words[-2:])
            if last_two in self._stop_variants:
                return " ".join(words[:-2]).strip(" ,.")

        return text


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_listener: WakeWordListener | None = None


def start_wake_listener(
    wake_word: str = "shrey",
    stop_word: str = "done",
    listen_timeout: int = 30,
    loop: asyncio.AbstractEventLoop | None = None,
) -> WakeWordListener:
    """Create and start the global wake word listener."""
    global _listener
    if _listener and _listener.is_running:
        logger.info("wake_listener_reusing_existing")
        return _listener

    _listener = WakeWordListener(
        wake_word=wake_word,
        stop_word=stop_word,
        listen_timeout=listen_timeout,
        loop=loop,
    )
    _listener.start()
    return _listener


def stop_wake_listener() -> None:
    """Stop the global wake word listener."""
    global _listener
    if _listener:
        _listener.stop()
        _listener = None


def get_wake_listener_status() -> dict:
    """For /health and the desktop app's tray: is the wake word being listened for?"""
    listener = _listener
    if listener is None:
        return {"running": False, "engine": None, "wake_word": None, "state": None}
    return {
        "running": listener.is_running,
        "engine": listener.engine_in_use,
        "wake_word": listener.wake_word,
        "state": listener._state.name.lower(),
    }
