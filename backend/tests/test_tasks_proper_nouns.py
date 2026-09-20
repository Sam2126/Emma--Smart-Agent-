"""
Tests for exact-spelling anchoring of proper nouns in the actor prompt.

Found in production: an instruction correctly transcribed as "...search
Rakesh and say hello to him..." led the model to type "Rockies" into
WhatsApp's search box — it re-spelled the name from memory while composing
the send_keys call instead of reading it back off the instruction. The actor
prompt pins probable proper nouns (capitalized, non-sentence-initial words)
as a short "copy these verbatim" list.
"""

from __future__ import annotations

from app.agent.prompts import (
    actor_user_prompt,
    extract_probable_proper_nouns,
    proper_noun_pin_section,
)


def test_extracts_the_exact_names_from_the_reported_bug():
    instruction = "open WhatsApp and search Rakesh and say hello to him done"
    assert extract_probable_proper_nouns(instruction) == ["WhatsApp", "Rakesh"]


def test_does_not_treat_sentence_initial_capitalization_as_a_proper_noun():
    assert "Open" not in extract_probable_proper_nouns("Open notepad and write today's date")


def test_returns_empty_list_when_no_proper_nouns_present():
    instruction = "open notepad and write today's date"
    assert extract_probable_proper_nouns(instruction) == []
    assert proper_noun_pin_section(instruction) == ""


def test_pin_section_tells_model_to_copy_verbatim_not_guess():
    section = proper_noun_pin_section("search for Bhawesh on whatsapp")
    assert "Bhawesh" in section
    assert "verbatim" in section.lower()
    assert "do not retype" in section.lower() or "guess" in section.lower()


def test_pin_section_deduplicates_repeated_names():
    assert proper_noun_pin_section("message Rakesh then message Rakesh again").count("Rakesh") == 1


def test_local_actor_prompt_includes_exact_spelling_pin():
    prompt = actor_user_prompt("open WhatsApp and search Rakesh and say hello to him", "local", plan="1. open_app")
    assert "Rakesh" in prompt
    assert "EXACT SPELLINGS" in prompt


def test_browser_actor_prompt_includes_exact_spelling_pin():
    prompt = actor_user_prompt("search for Priyal on the site", "browser", plan="")
    assert "Priyal" in prompt
    assert "EXACT SPELLINGS" in prompt


def test_skips_words_that_start_a_later_sentence_and_the_pronoun_i():
    """A polluted transcript once pinned 'Done', 'I' and 'Go' as names."""
    instruction = (
        "Open WhatsApp and search Rakesh and say hello to him. Done. Done. "
        "I read done. Go and eat done."
    )
    assert extract_probable_proper_nouns(instruction) == ["WhatsApp", "Rakesh"]
