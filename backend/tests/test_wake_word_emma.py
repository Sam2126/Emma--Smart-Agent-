"""
"Emma", the wake word since 2026-09-17, on real human recordings.

Measured when the word was chosen (4 speakers saying "Emma", 59 other words):
  no decoy words in the Vosk grammar: 4 of 4 detected, 10 of 59 false triggers
  (anna, gemma, hammer, mama, amber, ember, drama, karma, llama)
  decoys from wake_listener._VOSK_DECOYS["emma"]: 4 of 4 detected, 0 of 59
  "hammer" as a decoy as well: one speaker's "Emma" was heard as "hammer"
Some words here (hammer, karma, drama, llama) are deliberately NOT decoys, so
the test also covers look-alikes the grammar was not told about.
Recordings: Lingua Libre via Wikimedia Commons, see fixtures/voice/ATTRIBUTION.md.
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from app.config import Settings, get_settings
from app.wake_listener import WakeWordListener, _VOSK_DECOYS

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "voice"
MODEL = Path(__file__).resolve().parents[1] / "data" / "vosk-model-small-en-us-0.15"


def test_emma_is_the_default_wake_word():
    assert Settings(_env_file=None).wake_word == "emma"
    example = (Path(__file__).resolve().parents[2] / ".env.example").read_text(encoding="utf-8")
    assert "WAKE_WORD=emma" in example


def test_emma_phrases_are_recognised_and_stripped():
    listener = WakeWordListener(wake_word="emma", stop_word="done")
    for heard in ("emma", "Emma.", "hey emma", "okay emma", "emma open whatsapp"):
        assert listener._matches_wake_word(heard), heard
    assert listener._strip_wake_word("Emma, open WhatsApp and say hi") == "open WhatsApp and say hi"
    assert listener._strip_wake_word("Hey Emma, good job.") == "good job"
    assert listener._extract_instruction_after_wake("emma open notepad") == "open notepad"
    # "hello" is no longer the wake word
    assert not listener._matches_wake_word("hello open notepad")


def test_hammer_is_not_an_emma_decoy():
    assert "hammer" not in _VOSK_DECOYS["emma"]


def _emma_listener(monkeypatch) -> WakeWordListener:
    pytest.importorskip("vosk")
    if not MODEL.exists():
        pytest.skip("Vosk model not installed (scripts/install_vosk_model.py)")
    monkeypatch.setattr(get_settings(), "wake_word_engine", "vosk")
    monkeypatch.setattr(get_settings(), "vosk_model_path", str(MODEL))
    listener = WakeWordListener(wake_word="emma", stop_word="done")
    listener._load_vosk_if_configured()
    assert listener.engine_in_use == "vosk"
    return listener


def _heard(listener: WakeWordListener, path: Path) -> str:
    import speech_recognition as sr

    with wave.open(str(path)) as w:
        audio = sr.AudioData(w.readframes(w.getnframes()), w.getframerate(), w.getsampwidth())
    return listener._recognize_vosk(audio)


def test_vosk_hears_emma_from_real_speakers(monkeypatch):
    files = sorted(FIXTURES.glob("emma_*.wav"))
    if not files:
        pytest.skip("no recordings")
    listener = _emma_listener(monkeypatch)
    heard = {f.name: _heard(listener, f) for f in files}
    missed = {name: text for name, text in heard.items() if not listener._matches_wake_word(text)}
    assert not missed, f"Emma not detected (file: heard): {missed}"


def test_other_words_do_not_wake_emma(monkeypatch):
    files = sorted(FIXTURES.glob("emmaneg_*.wav")) + sorted(FIXTURES.glob("neg_*.wav")) \
        + sorted(FIXTURES.glob("hello_*.wav")) + sorted(FIXTURES.glob("done_*.wav"))
    if not files:
        pytest.skip("no recordings")
    listener = _emma_listener(monkeypatch)
    triggered = {f.name: text for f in files if (text := _heard(listener, f)) and listener._matches_wake_word(text)}
    assert not triggered, f"false wake-ups (file: heard): {triggered}"
