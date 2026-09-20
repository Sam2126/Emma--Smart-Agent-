"""
Tests for hearing the user correctly, and for fitting a request into one call.

The cases here are taken from a real run. The user said:

    "Emma, open Chrome, search Myntra, done"

and the agent acted on:

    "M R Open Chrome Search Mentra Done"

It then searched Google for "Mentra", a word that does not exist, and reported
success. The same run failed twice with "Request too large for model
openai/gpt-oss-120b" because five page readings in a row were sent whole.
"""

from __future__ import annotations

import pytest

from app.agent.toolkit import compact_messages
from app.config import get_settings
from app.speech_vocab import (
    clean_instruction,
    correct_names,
    strip_stop_word,
    strip_wake_word,
    whisper_prompt,
)


# =============================================================================
# What Whisper is told before it listens
# =============================================================================

def test_the_names_the_agent_works_with_are_given_to_whisper():
    prompt = whisper_prompt()
    for name in ("Myntra", "WhatsApp", "Flipkart", "Emma"):
        assert name in prompt


def test_the_user_can_add_their_own_names():
    prompt = whisper_prompt("Shrey, Bhawesh, Bennett University")
    assert "Shrey" in prompt and "Bennett University" in prompt


# =============================================================================
# Repairing what came back
# =============================================================================

@pytest.mark.parametrize("heard, meant", [
    ("Mentra", "Myntra"),
    ("Flipcart", "Flipkart"),
    ("Whatsup", "WhatsApp"),
    ("Youtub", "YouTube"),
])
def test_a_misheard_name_is_repaired(heard, meant):
    assert correct_names(heard) == meant


@pytest.mark.parametrize("ordinary", [
    "the mantra for the meeting",
    "meet me at the centre",
    "send the monitor number",
])
def test_ordinary_words_are_not_turned_into_brand_names(ordinary):
    assert correct_names(ordinary) == ordinary


@pytest.mark.parametrize("heard", ["M R open gmail", "Emma open gmail", "Amma, open gmail",
                                   "em open gmail", "hey emma open gmail"])
def test_the_wake_word_is_removed_however_it_was_heard(heard):
    # It is spoken at the very start of the recording and comes back clipped,
    # so the spelling "emma" alone is not enough to catch it.
    assert strip_wake_word(heard).lower().startswith("open gmail")


@pytest.mark.parametrize("heard", ["check my mail done", "check my mail dawn",
                                   "check my mail Done.", "check my mail done done"])
def test_the_stop_word_is_removed_however_it_was_heard(heard):
    assert strip_stop_word(heard).lower() == "check my mail"


def test_the_instruction_from_the_real_failure_is_recovered():
    assert clean_instruction("M R Open Chrome Search Mentra Done") == "Open Chrome Search Myntra"


def test_an_instruction_that_was_heard_correctly_is_left_alone():
    said = "open whatsapp and message Shrey about the meeting"
    assert clean_instruction(said) == said


# =============================================================================
# Fitting a long run into one request
# =============================================================================

def _run_of(page_readings: int, size: int = 6000) -> list[dict]:
    """The shape of the run that Groq refused: several big readings in a row."""
    messages: list[dict] = [
        {"role": "system", "content": "S" * 2000},
        {"role": "user", "content": "open chrome and search myntra"},
    ]
    for i in range(page_readings):
        messages.append({
            "role": "assistant", "content": None,
            "tool_calls": [{"id": f"c{i}", "function": {"name": "perceive_page"}}],
        })
        messages.append({
            "role": "tool", "tool_call_id": f"c{i}", "name": "perceive_page", "content": "P" * size,
        })
    return messages


def _size(messages: list[dict]) -> int:
    return sum(len(m.get("content") or "") for m in messages)


def test_a_long_run_is_squeezed_under_the_request_limit():
    messages = _run_of(5)
    assert _size(messages) > 30000, "this is the size Groq refused"
    fitted = compact_messages(messages)
    assert _size(fitted) <= get_settings().llm_request_char_budget


def test_nothing_is_dropped_only_shortened():
    # Each tool result has to stay paired with the call that asked for it;
    # removing a message breaks the conversation the model is given.
    messages = _run_of(5)
    fitted = compact_messages(messages)
    assert len(fitted) == len(messages)
    assert [m.get("role") for m in fitted] == [m.get("role") for m in messages]
    assert [m.get("tool_call_id") for m in fitted] == [m.get("tool_call_id") for m in messages]


def test_the_newest_result_survives_whole():
    # It is the one the agent is about to act on.
    fitted = compact_messages(_run_of(5))
    assert len(fitted[-1]["content"]) == 6000


def test_a_short_run_is_left_exactly_as_it_was():
    messages = _run_of(1, size=400)
    assert compact_messages(messages) == messages


def test_even_one_enormous_result_is_brought_within_the_limit():
    messages = _run_of(1, size=60000)
    fitted = compact_messages(messages)
    assert _size(fitted) <= get_settings().llm_request_char_budget
    assert len(fitted) == len(messages)


# =============================================================================
# Talking to her without the wake word
# =============================================================================

def test_the_button_asks_the_listener_to_start_listening():
    from app.wake_listener import WakeWordListener

    listener = WakeWordListener(wake_word="emma", stop_word="done")
    assert listener.request_conversation() is True
    assert listener._talk_requested.is_set()


def test_a_stopped_listener_refuses_to_start_a_conversation():
    from app.wake_listener import WakeWordListener

    listener = WakeWordListener(wake_word="emma", stop_word="done")
    listener._stop_event.set()
    assert listener.request_conversation() is False


def test_the_endpoint_says_so_when_nothing_is_listening():
    import asyncio

    import app.main as main

    main._wake_listener = None
    result = asyncio.run(main.start_talking())
    assert result["ok"] is False and "not running" in result["detail"]


# =============================================================================
# Saying hello is not a job
# =============================================================================

@pytest.mark.parametrize("said", [
    "hello", "hi", "hey emma", "hello whatsup", "good morning", "thanks emma",
    "how are you", "hi emma how are you", "bye", "thank you",
])
def test_a_greeting_is_answered_not_planned(said):
    from app.conversation import QUESTION, classify

    # A real run sent "hello what's up" through planning and came back asking
    # what the user wanted done with the phrase.
    assert classify(said) == QUESTION


@pytest.mark.parametrize("said", [
    "hello open whatsapp",
    "hi can you search myntra",
    "thanks now open my downloads",
])
def test_a_greeting_in_front_of_an_instruction_is_still_an_instruction(said):
    from app.conversation import TASK, classify

    assert classify(said) == TASK


# =============================================================================
# The two microphones
# =============================================================================

def test_the_conversation_microphone_is_marked_as_such():
    from app.websocket.protocol import TaskSubmitMessage, VoiceTaskMessage

    assert VoiceTaskMessage(audio_base64="x").conversation is False
    assert VoiceTaskMessage(audio_base64="x", conversation=True).conversation is True
    assert TaskSubmitMessage(instruction="hi", conversation=True).conversation is True


async def test_an_ambiguous_sentence_is_answered_in_a_conversation(monkeypatch):
    """The same words go different ways depending on which microphone sent them.

    Only for wording that could be either. "Open Myntra" is a job from both.
    """
    import app.conversation as conversation

    async def unreachable(*a, **k):
        raise RuntimeError("no model")

    monkeypatch.setattr(conversation, "chat", unreachable)
    ambiguous = "what did priya say about tomorrow"
    assert await conversation.route(ambiguous, conversation=False) == conversation.TASK
    assert await conversation.route(ambiguous, conversation=True) == conversation.QUESTION


async def test_a_plain_instruction_is_still_carried_out_in_a_conversation():
    import app.conversation as conversation

    assert await conversation.route("open myntra and search shoes", conversation=True) == conversation.TASK


# =============================================================================
# The window has both buttons
# =============================================================================

def test_the_window_offers_a_task_microphone_and_a_talking_one():
    from pathlib import Path

    page = Path(__file__).resolve().parents[1] / "app" / "static" / "local_agent.html"
    html = page.read_text(encoding="utf-8")
    assert 'id="mic"' in html and 'id="talk"' in html
    # The talking one sends the flag; the task one does not.
    assert "conversation: conversation" in html
    assert "beginRecording({ conversation: true })" in html
    assert "beginRecording({ conversation: false })" in html
