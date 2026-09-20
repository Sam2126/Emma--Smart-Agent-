"""
Tests for Emma's voice and for telling a question apart from a job.

No audio is played and no voice service is called: synthesis and playback are
stubbed, so these run anywhere. What they protect is the behaviour that makes
speaking safe - the microphone gate, the stop, the privacy rule and the
fallback order - and the wording rules that decide what is said.
"""

from __future__ import annotations

import time

import pytest

from app.config import get_settings
import app.tts as tts
from app.conversation import QUESTION, TASK, classify


# =============================================================================
# Turning written text into something worth hearing
# =============================================================================

@pytest.mark.parametrize("written, spoken", [
    ("**Done!**", "Done!"),
    ("Saved to C:\\Users\\sam\\Desktop\\report.txt", "Saved to the file report.txt"),
    ("See https://myntra.com/x for more", "See the link for more"),
    ("Added a Rs. 1,299 shirt", "Added a one thousand two hundred ninety nine rupees shirt"),
    ("₹450 total", "four hundred fifty rupees total"),
    ("Used click_element then see_page", "Used clicking then looking at the page"),
    ("Task complete 🎉", "Task complete"),
])
def test_written_text_is_rewritten_for_the_ear(written, spoken):
    assert tts.speakable(written) == spoken


def test_a_paragraph_is_split_on_sentences():
    assert tts.sentences("First one. Second one! Third?") == ["First one.", "Second one!", "Third?"]


def test_a_very_long_sentence_is_split_on_a_word_boundary():
    chunks = tts.sentences("word " * 120, limit=100)
    assert all(len(c) <= 100 for c in chunks)
    assert not any(c.endswith("wor") for c in chunks), "must not cut a word in half"


# =============================================================================
# Audio plumbing
# =============================================================================

def test_raw_audio_is_wrapped_so_it_can_be_played():
    # 24000 samples of 16-bit mono at 24 kHz is exactly one second of audio.
    wav = tts._wav_from_pcm(b"\x00\x01" * 24000, rate=24000)
    assert tts._looks_like_wav(wav)
    assert abs(tts.Speaker._duration(wav) - 1.0) < 0.05


def test_something_that_is_not_audio_is_rejected():
    assert not tts._looks_like_wav(b'{"error": "nope"}')


# =============================================================================
# The speaker
# =============================================================================

@pytest.fixture
def speaker(monkeypatch):
    """A speaker whose voice and speakers are stubbed out."""
    spoken: list[str] = []

    sp = tts.Speaker()
    monkeypatch.setattr(sp, "_play", lambda audio, generation: spoken.append(audio))
    monkeypatch.setattr(sp._voices, "synthesize", lambda text, sensitive: (text.encode(), "test"))
    monkeypatch.setattr(tts, "_speaker", sp)
    settings = get_settings()
    monkeypatch.setattr(settings, "tts_enabled", True)
    monkeypatch.setattr(settings, "tts_cache_enabled", False)
    sp.spoken = spoken
    return sp


def _drain(sp, seconds=2.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and (not sp._queue.empty()):
        time.sleep(0.02)
    time.sleep(0.15)


def test_what_is_said_reaches_the_speakers(speaker):
    assert speaker.say("Hello there.") is True
    _drain(speaker)
    assert b"Hello there." in speaker.spoken


def test_nothing_is_spoken_when_the_voice_is_switched_off(speaker, monkeypatch):
    monkeypatch.setattr(get_settings(), "tts_enabled", False)
    assert speaker.say("Hello") is False


def test_the_same_line_twice_in_a_row_is_not_repeated(speaker):
    assert speaker.say("Working on it.", priority=tts.Priority.PROGRESS) is True
    assert speaker.say("Working on it.", priority=tts.Priority.PROGRESS) is False


def test_chatter_is_dropped_when_it_would_arrive_too_late(speaker, monkeypatch):
    monkeypatch.setattr(get_settings(), "tts_progress_queue_limit", 2)
    speaker._queue.put(tts._Utterance(3, 1, "one"))
    speaker._queue.put(tts._Utterance(3, 2, "two"))
    assert speaker.say("three", priority=tts.Priority.PROGRESS) is False
    # An error still gets through a full queue.
    assert speaker.say("Something went wrong.", priority=tts.Priority.URGENT) is True


def test_a_long_report_is_shortened_rather_than_read_out(speaker, monkeypatch):
    monkeypatch.setattr(get_settings(), "tts_max_chars", 50)
    speaker.say("word " * 200)
    _drain(speaker)
    assert len(speaker.spoken[0]) <= 60


def test_stopping_clears_everything_waiting(speaker):
    for i in range(5):
        speaker.say(f"Sentence number {i}.", priority=tts.Priority.PROGRESS)
    speaker.stop()
    assert speaker._queue.empty()
    assert speaker.is_speaking() is False


def test_the_microphone_gate_is_open_only_while_speaking(speaker):
    assert speaker.is_speaking() is False
    speaker._speaking_until = time.monotonic() + 1.0
    assert speaker.is_speaking() is True
    speaker.stop()
    assert speaker.is_speaking() is False


def test_the_wake_listener_asks_whether_emma_is_speaking():
    # Without this gate Emma saying the word "Emma" wakes Emma, forever.
    from app.wake_listener import _voice_is_speaking

    assert _voice_is_speaking() in (True, False)


# =============================================================================
# Choosing a voice
# =============================================================================

def test_private_text_never_goes_to_a_cloud_voice(monkeypatch):
    voices = tts._Voices()
    called: list[str] = []
    monkeypatch.setattr(voices, "groq", lambda t, v: called.append("groq") or b"RIFF....WAVE")
    monkeypatch.setattr(voices, "gemini", lambda t, v: called.append("gemini") or b"RIFF....WAVE")
    monkeypatch.setattr(voices, "windows", lambda t, v: called.append("windows") or b"RIFF....WAVE")

    audio, tier = voices.synthesize("your one time code is 402913", sensitive=True)
    assert tier == tts.Tier.WINDOWS
    assert called == ["windows"], "a password or code must not be sent to a voice service"


def test_the_next_voice_answers_when_one_is_unavailable(monkeypatch):
    voices = tts._Voices()
    order: list[str] = []
    monkeypatch.setattr(voices, "groq", lambda t, v: order.append("groq") and None)
    monkeypatch.setattr(voices, "gemini", lambda t, v: order.append("gemini") and None)
    monkeypatch.setattr(voices, "windows", lambda t, v: (order.append("windows"), b"wav")[1])
    monkeypatch.setattr(get_settings(), "tts_order", "groq,gemini,windows")

    audio, tier = voices.synthesize("hello", sensitive=False)
    assert order == ["groq", "gemini", "windows"]
    assert tier == tts.Tier.WINDOWS and audio == b"wav"


def test_a_voice_that_failed_sits_out_the_next_line(monkeypatch):
    voices = tts._Voices()
    voices.cool(tts.Tier.GROQ, seconds=60, reason="test")
    assert voices.available(tts.Tier.GROQ) is False
    assert voices.available(tts.Tier.GEMINI) is True


# =============================================================================
# A question is answered; a job is done
# =============================================================================

@pytest.mark.parametrize("said, kind", [
    ("open myntra and filter oversized tshirt under 1000", TASK),
    ("what did you just do?", QUESTION),
    ("send shrey an email saying hello", TASK),
    ("can you open whatsapp", TASK),
    ("add the cheapest shirt to my cart", TASK),
    ("search for oversized tshirts", TASK),
    ("what did you just do?", QUESTION),
    ("what did you open in chrome", QUESTION),
    ("why did that fail?", QUESTION),
    ("how does the wake word work", QUESTION),
    ("what can you do", QUESTION),
    ("tell me about yourself", QUESTION),
    ("are you listening", QUESTION),
])
def test_questions_and_jobs_are_told_apart(said, kind):
    assert classify(said) == kind


@pytest.mark.parametrize("said", [
    "what did priya say about tomorrow",
    "how many unread emails do i have",
])
def test_a_question_needing_the_user_s_own_data_is_not_settled_by_wording(said):
    # Phrased as a question, but answering means opening mail. The patterns
    # must admit they cannot tell, so the model decides.
    from app.conversation import UNSURE

    assert classify(said) == UNSURE


async def test_an_ambiguous_question_falls_back_to_doing_the_work(monkeypatch):
    # When the model cannot be reached, doing the work is the recoverable
    # mistake; answering from imagination is not.
    import app.conversation as conversation

    async def broken(*a, **k):
        raise RuntimeError("no model")

    monkeypatch.setattr(conversation, "chat", broken)
    assert await conversation.route("what did priya say about tomorrow") == TASK


def test_an_empty_instruction_is_not_treated_as_a_question():
    assert classify("") == TASK
    assert classify("   ") == TASK


# =============================================================================
# Saying numbers, times and names the way a person would
# =============================================================================

@pytest.mark.parametrize("written, spoken", [
    ("at 16:19", "at four nineteen in the afternoon"),
    ("at 9:05", "at nine oh five in the morning"),
    ("at 12:00", "at twelve o'clock in the afternoon"),
    ("72% off", "seventy two percent off"),
    ("the 1st and the 3rd", "the first and the third"),
    ("2.5 MB", "two point five megabytes"),
    ("101187 items", "one lakh one thousand one hundred eighty seven items"),
    ("12345678 rows", "one crore twenty three lakh forty five thousand six hundred seventy eight rows"),
    ("2-3 per brand", "two to three per brand"),
    ("search -> filter", "search then filter"),
    ("your OTP", "your O T P"),
])
def test_numbers_and_times_are_spoken_not_read(written, spoken):
    assert tts.speakable(written) == spoken


def test_a_task_id_is_not_read_out_character_by_character():
    said = tts.speakable("Task 04fc954b-a131-4e19-8bc8-45c17eae91c5 finished.")
    assert said == "Task an identifier finished."


def test_hindi_is_recognised_as_a_script_the_local_voice_cannot_read():
    assert tts.is_indic("कल मीटिंग है") is True
    assert tts.is_indic("meeting tomorrow") is False


def test_hindi_goes_to_a_voice_that_can_speak_it(monkeypatch):
    voices = tts._Voices()
    tried: list[str] = []
    monkeypatch.setattr(voices, "groq", lambda t, v: tried.append("groq") and None)
    monkeypatch.setattr(voices, "gemini", lambda t, v: (tried.append("gemini"), b"wav")[1])
    monkeypatch.setattr(voices, "windows", lambda t, v: (tried.append("windows"), b"wav")[1])
    # Even with the local English voice configured first.
    monkeypatch.setattr(get_settings(), "tts_order", "windows,groq,gemini")

    audio, tier = voices.synthesize("कल मीटिंग है", sensitive=False)
    assert tier == tts.Tier.GEMINI
    assert "windows" not in tried, "an English voice reads Devanagari as noise"


# =============================================================================
# Interrupting her
# =============================================================================

SPEAKING = "I have stopped the download and opened the cart for you"


@pytest.mark.parametrize("heard, interrupts", [
    ("stop", True),
    ("quiet please", True),
    ("shut up emma", True),
    ("wait", True),
    # Her own voice coming back through the microphone a moment later.
    ("stopped the download", False),
    ("i have stopped", False),
    ("the cart for you", False),
    # A new instruction is not an interruption; it waits its turn.
    ("open my email", False),
    ("", False),
])
def test_only_a_real_interruption_stops_her(heard, interrupts):
    from app.wake_listener import _is_interruption

    assert _is_interruption(heard, SPEAKING) is interrupts


def test_what_she_is_saying_is_visible_while_she_says_it(speaker, monkeypatch):
    seen = {}
    monkeypatch.setattr(speaker, "_play", lambda audio, gen: seen.setdefault("text", speaker.current_text()))
    speaker.say("Filtering the results now.")
    _drain(speaker)
    assert "Filtering the results now." in seen.get("text", "")


def test_the_second_time_a_line_is_said_it_comes_from_the_cache(speaker, monkeypatch, tmp_path):
    # Its own cache directory: the real one already holds the lines Emma says,
    # which would make the first call a hit and prove nothing.
    monkeypatch.setattr(tts, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(get_settings(), "tts_cache_enabled", True)
    calls: list[str] = []
    monkeypatch.setattr(speaker._voices, "synthesize",
                        lambda text, sensitive: (calls.append(text), (b"RIFF" + b"0" * 60, "test"))[1])
    first_audio, first_tier = speaker._audio_for("A cached line for the test.", False)
    second_audio, second_tier = speaker._audio_for("A cached line for the test.", False)
    assert first_tier == "test" and second_tier == "cache"
    assert first_audio == second_audio
    assert len(calls) == 1, "a repeated line must not be synthesized twice"


# =============================================================================
# The voice model's terms are accepted per account
# =============================================================================
# Measured on this installation: of four Groq keys, ONE account had accepted
# the voice model's terms and three had not. Giving up on the first refusal
# meant the good key was never reached and Emma never used that voice at all.

class _Reply:
    def __init__(self, status, text="", content=b""):
        self.status_code = status
        self.text = text
        self.content = content


def _wav() -> bytes:
    return tts._wav_from_pcm(b"\x00\x01" * 100)


def test_the_key_whose_account_accepted_the_terms_is_found(monkeypatch):
    keys = ["key-no", "key-no-2", "key-yes", "key-no-3"]
    used = []

    class _Pool:
        size = len(keys)

        def acquire(self):
            key = keys[len(used) % len(keys)]
            used.append(key)
            return key

    monkeypatch.setattr("app.utils.key_pool.get_key_pool", lambda: _Pool())

    def post(url, headers=None, json=None, timeout=None):
        key = (headers or {})["Authorization"].split()[-1]
        if key == "key-yes":
            return _Reply(200, content=_wav())
        return _Reply(400, text='{"error":{"message":"requires terms acceptance"}}')

    monkeypatch.setattr("httpx.post", post)

    voices = tts._Voices()
    audio = voices.groq("hello", "autumn")
    assert audio is not None, "the key that may use the voice must be tried"
    assert "key-yes" in used


def test_the_voice_is_not_disabled_when_every_account_refuses(monkeypatch):
    class _Pool:
        size = 2

        def acquire(self):
            return "key"

    monkeypatch.setattr("app.utils.key_pool.get_key_pool", lambda: _Pool())
    monkeypatch.setattr(
        "httpx.post",
        lambda *a, **k: _Reply(400, text='{"error":{"message":"requires terms acceptance"}}'),
    )
    voices = tts._Voices()
    assert voices.groq("hello", "autumn") is None
    # It sits out rather than asking again for every sentence, since only a
    # click in the Groq console can change the answer.
    assert voices.available(tts.Tier.GROQ) is False


def test_a_wrong_voice_name_is_reported_as_a_setting_not_an_outage(monkeypatch):
    class _Pool:
        size = 4

        def acquire(self):
            return "key"

    calls = []
    monkeypatch.setattr("app.utils.key_pool.get_key_pool", lambda: _Pool())

    def post(*a, **k):
        calls.append(1)
        return _Reply(400, text='{"error":{"message":"voice must be one of the following voices: [autumn diana]"}}')

    monkeypatch.setattr("httpx.post", post)
    voices = tts._Voices()
    assert voices.groq("hello", "tara") is None
    assert len(calls) == 1, "a wrong name is the same on every key; trying them all is pointless"


def test_audio_cached_in_one_voice_is_not_replayed_in_another(monkeypatch, tmp_path):
    # While the Groq voice was unavailable, lines were cached in Gemini's
    # voice. Once it becomes available those must not keep playing, or Emma
    # answers in two voices depending on which sentences were cached.
    monkeypatch.setattr(tts, "CACHE_DIR", tmp_path)
    speaker = tts.Speaker()
    monkeypatch.setattr(get_settings(), "tts_order", "gemini,windows")
    while_gemini = speaker._key("Done.", sensitive=False)
    monkeypatch.setattr(get_settings(), "tts_order", "groq,gemini,windows")
    with_groq = speaker._key("Done.", sensitive=False)
    assert while_gemini != with_groq


def test_private_text_is_cached_under_the_offline_voice(monkeypatch, tmp_path):
    monkeypatch.setattr(tts, "CACHE_DIR", tmp_path)
    speaker = tts.Speaker()
    assert speaker._key("code 4021", sensitive=True) != speaker._key("code 4021", sensitive=False)
