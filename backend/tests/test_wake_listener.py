"""
Tests for the hands-free wake-word listener's matching/parsing logic.

Covers two real bugs found in production use (see session notes):
1. _matches_stop_word used to match a stop variant occurring ANYWHERE as a
   substring of the heard text, so a genuine instruction like "stop the
   alarm and open notepad" aborted recording immediately after "stop the
   alarm" — well before the user finished speaking. Fixed to require the
   stop word be the whole utterance or the last word/two words.
2. The RECORDING phase used to stitch together Google's free-API text from
   each short chunk, which frequently returns empty or garbled text and
   loses words at chunk boundaries. Fixed to buffer raw audio and
   transcribe the whole recording once via Groq Whisper.

Pure logic (word matching, wake/stop-word stripping, WAV assembly) is
tested directly. Anything touching a real microphone is out of scope here.
"""

from __future__ import annotations

import wave
import io

from app.wake_listener import WakeWordListener


def _listener(wake_word: str = "hello", stop_word: str = "done") -> WakeWordListener:
    return WakeWordListener(wake_word=wake_word, stop_word=stop_word, listen_timeout=30)


# =============================================================================
# Wake-word matching
# =============================================================================

def test_wake_word_matches_standalone_hello():
    l = _listener()
    assert l._matches_wake_word("hello") is True
    assert l._matches_wake_word("Hello") is True
    assert l._matches_wake_word("halo") is True  # phonetic variant


def test_wake_word_matches_when_first_word_of_longer_phrase():
    l = _listener()
    assert l._matches_wake_word("hello open whatsapp") is True


def test_wake_word_does_not_match_when_embedded_mid_sentence():
    """The core "wait for HELLO" bug: a full command spoken in one breath
    with 'hello' buried inside it must NOT count as the wake trigger —
    otherwise the instruction before it is silently discarded."""
    l = _listener()
    assert l._matches_wake_word("open whatsapp and search rakesh and say hello") is False
    assert l._matches_wake_word("whatsapp and search rakesh and say hello") is False


def test_wake_word_does_not_match_unrelated_speech():
    l = _listener()
    assert l._matches_wake_word("open chrome and search amazon") is False
    assert l._matches_wake_word("") is False


# =============================================================================
# Stop-word matching — the false-positive-abort bug
# =============================================================================

def test_stop_word_matches_standalone_done():
    l = _listener()
    assert l._matches_stop_word("done") is True
    assert l._matches_stop_word("Done.") is True


def test_stop_word_matches_phonetic_mishearing_of_done():
    """Found in production: the free recognizer transcribed a clearly-
    spoken 'done, done' as 'dan dan', so the stop cue was never detected
    and recording ran the full 30s timeout, capturing a lot of unrelated
    audio in between. _build_wake_variants already handles this class of
    error for 'hello' (halo, helo, hullo); _build_stop_variants needed the
    same phonetic tolerance for 'done'."""
    l = _listener()
    assert l._matches_stop_word("dan dan") is True
    assert l._matches_stop_word("dan") is True
    assert l._matches_stop_word("open whatsapp and search rakesh dan dan") is True


def test_stop_word_matches_when_spoken_naturally_at_the_end():
    l = _listener()
    assert l._matches_stop_word("open whatsapp and search rakesh okay I'm done") is True
    assert l._matches_stop_word("send the message that's it") is True


def test_stop_word_does_not_match_when_incidental_mid_sentence():
    """The core false-abort bug: a real instruction containing a stop-word
    lookalike in the MIDDLE must not cut recording off early."""
    l = _listener()
    assert l._matches_stop_word("stop the alarm and open notepad") is False
    assert l._matches_stop_word("please finish the report first") is False


def test_stop_word_does_not_match_unrelated_speech():
    l = _listener()
    assert l._matches_stop_word("open whatsapp and search rakesh") is False
    assert l._matches_stop_word("") is False


# =============================================================================
# Extraction / stripping helpers
# =============================================================================

def test_extract_instruction_after_wake():
    l = _listener()
    assert l._extract_instruction_after_wake("hello") == ""
    assert l._extract_instruction_after_wake("hello open whatsapp") == "open whatsapp"


def test_strip_wake_word_keeps_remainder():
    l = _listener()
    assert l._strip_wake_word("hello search for earbuds") == "search for earbuds"
    assert l._strip_wake_word("no wake word here") == "no wake word here"


def test_strip_stop_word_keeps_prefix():
    l = _listener()
    assert l._strip_stop_word("send the message hello done") == "send the message hello"
    assert l._strip_stop_word("no stop word here") == "no stop word here"


# =============================================================================
# WAV assembly for the accurate one-shot Whisper transcription
# =============================================================================

def test_audio_chunks_to_wav_produces_valid_playable_wav():
    sample_rate = 16000
    sample_width = 2
    chunk_a = b"\x00\x01" * 100
    chunk_b = b"\x02\x03" * 50

    wav_bytes = WakeWordListener._audio_chunks_to_wav(
        [chunk_a, chunk_b], sample_rate, sample_width
    )

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == sample_width
        assert wf.getframerate() == sample_rate
        frames = wf.readframes(wf.getnframes())
        assert frames == chunk_a + chunk_b


# =============================================================================
# Repeated "done" and transcript cleanup — both from the same production run.
# =============================================================================

def test_stop_word_matches_when_repeated_inside_a_merged_chunk():
    """Live chunks merge continuous speech, so saying "done" over and over
    produced chunks that ended on some other word and never matched the
    last-word rule. Two or more core stop words in one chunk must stop."""
    l = _listener()
    assert l._matches_stop_word("done I read done go and eat") is True
    assert l._matches_stop_word("say hello to him dan dan okay") is True


def test_single_mid_sentence_done_still_does_not_stop():
    l = _listener()
    assert l._matches_stop_word("check if the download is done and open it") is False


def test_repeated_synonym_like_stop_does_not_use_the_repeat_rule():
    """The repeat rule only counts the stop word and its mishearings, not
    synonyms like "stop" that occur in real instructions."""
    l = _listener()
    assert l._matches_stop_word("stop the alarm and stop the music please") is False


def test_truncate_at_stop_sentence_drops_everything_after_the_first_done():
    from app.wake_listener import _truncate_at_stop_sentence

    l = _listener()
    transcript = (
        "Open WhatsApp and search Rakesh and say hello to him. Done. Done. "
        "Done. Done. I read done. Go and eat done. Done done"
    )
    assert _truncate_at_stop_sentence(transcript, l._stop_variants) == (
        "Open WhatsApp and search Rakesh and say hello to him."
    )


def test_truncate_at_stop_sentence_keeps_instructions_without_a_stop_sentence():
    from app.wake_listener import _truncate_at_stop_sentence

    l = _listener()
    transcript = "Open notepad. Write that the report is done."
    assert _truncate_at_stop_sentence(transcript, l._stop_variants) == transcript
