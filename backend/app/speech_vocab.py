"""
Helping Whisper hear the words this user actually says.

Added 2026-09-20 after a real run. "Emma, open Chrome, search Myntra, done"
came back as:

    M R Open Chrome Search Mentra Done

Three separate faults in one sentence:

  "Emma"   -> "M R"     the wake word is spoken at the very start of the
                        recording, half clipped, and Whisper guessed letters.
                        The stripper only knew the spelling "emma", so the
                        guess survived into the instruction.
  "Myntra" -> "Mentra"  a proper noun Whisper has no reason to know. Speech
                        models take a PROMPT of likely words, and naming the
                        shops and apps this agent drives fixes most of these.
  "done"   -> "Done"    the stop word is spoken into the recording on purpose,
                        so it has to come back off the end.

Whatever survives is what the planner acts on, and it acted on "Mentra" - it
searched Google for a word that does not exist. Hearing correctly is cheaper
than recovering afterwards.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

# The names this agent hears most: what it can open, search and be told to use.
# Sent to Whisper as a prompt, which is how a speech model is told what kind of
# words to expect, and used afterwards to repair near misses.
KNOWN_NAMES = (
    # the agent itself
    "Emma",
    # shopping
    "Myntra", "Flipkart", "Amazon", "Ajio", "Meesho", "Nykaa", "Zomato", "Swiggy",
    "BigBasket", "Blinkit",
    # messaging and mail
    "WhatsApp", "Gmail", "Outlook", "Telegram", "Slack", "Discord", "Instagram",
    "LinkedIn", "Twitter", "Facebook", "Snapchat",
    # browsers, tools, assistants
    "Chrome", "Google", "YouTube", "ChatGPT", "Gemini", "Claude", "Copilot",
    "Notepad", "Spotify", "Excel", "PowerPoint", "Outlook", "GitHub", "Reddit",
    "Netflix", "Hotstar", "Paytm", "PhonePe", "Zerodha", "Canva", "Figma",
    # words the agent's own instructions use
    "Downloads", "Desktop", "Documents", "screenshot", "bookmark", "playlist",
)

# Everyday words that must never be "corrected" into a brand name.
_PROTECTED = {
    "centre", "center", "mantra", "member", "monitor", "meeting", "morning",
    "message", "manager", "number", "folder", "matter", "mother", "another",
    "gemini",  # already correct
}

# How the wake word comes back when it is clipped at the start of a recording.
WAKE_MISHEARINGS = {
    "emma", "ema", "amma", "imma", "m r", "mr", "m", "amr", "em", "emma emma",
    "hey emma", "hi emma", "ok emma", "okay emma", "hello emma", "ama", "umma",
    "emir", "amar", "ammar", "enma", "elma",
}

# How the stop word comes back at the end of one.
STOP_MISHEARINGS = {
    "done", "dawn", "don", "dun", "doon", "down", "ton", "dome", "done done",
    "i'm done", "im done", "that's it", "thats it", "finish", "finished", "stop",
}

_WORD = re.compile(r"[A-Za-z']+")


def whisper_prompt(extra_names: str = "") -> str:
    """The hint given to Whisper before it transcribes.

    A speech model uses this as context for what it is about to hear, which is
    what turns "Mentra" back into "Myntra". Keep it a plain list of names: it
    is a prompt, not an instruction, and a sentence here can be transcribed
    into the result.
    """
    names = list(KNOWN_NAMES)
    names += [n.strip() for n in (extra_names or "").split(",") if n.strip()]
    return "Words that may appear: " + ", ".join(dict.fromkeys(names)) + "."


def _close(word: str, name: str) -> float:
    return SequenceMatcher(None, word.lower(), name.lower()).ratio()


def correct_names(text: str, extra_names: str = "") -> str:
    """Repair near misses of names the agent knows: 'Mentra' -> 'Myntra'.

    Only words long enough to be a name are touched, and only when they are
    not ordinary English: "mantra" and "meeting" stay as they are.
    """
    if not text:
        return text
    names = list(KNOWN_NAMES) + [n.strip() for n in (extra_names or "").split(",") if n.strip()]

    def repair(match: re.Match) -> str:
        word = match.group(0)
        if len(word) < 5 or word.lower() in _PROTECTED:
            return word
        best, score = "", 0.0
        for name in names:
            if abs(len(name) - len(word)) > 2:
                continue
            ratio = _close(word, name)
            if ratio > score:
                best, score = name, ratio
        if score >= 0.8 and best.lower() != word.lower():
            return best
        return word

    return _WORD.sub(repair, text)


def _leading_tokens(text: str, count: int = 3) -> list[str]:
    return text.split()[:count]


def strip_wake_word(text: str, variants: set[str] | None = None) -> str:
    """Remove the wake word from the front, however badly it was heard.

    The recording starts with the user saying "Emma", usually clipped, so
    Whisper returns anything from "Emma" to "M R" to "Amar". Up to two leading
    tokens are dropped while they still look like the wake word.
    """
    known = {v.lower() for v in (variants or set())} | WAKE_MISHEARINGS
    words = text.strip().split()
    for _ in range(3):
        if not words:
            break
        first = words[0].strip(".,!?").lower()
        two = " ".join(w.strip(".,!?").lower() for w in words[:2])
        if two in known:
            words = words[2:]
            continue
        if first in known or (len(first) <= 2 and len(words) > 1):
            words = words[1:]
            continue
        # A single near-miss of the wake word: "emmah", "emmar", "amma".
        if len(first) >= 3 and any(_close(first, v) >= 0.75 for v in known if len(v) >= 3):
            words = words[1:]
            continue
        break
    return " ".join(words)


def strip_stop_word(text: str, variants: set[str] | None = None) -> str:
    """Remove the stop word from the end, however badly it was heard."""
    known = {v.lower() for v in (variants or set())} | STOP_MISHEARINGS
    words = text.strip().split()
    for _ in range(4):
        if not words:
            break
        last = words[-1].strip(".,!?").lower()
        two = " ".join(w.strip(".,!?").lower() for w in words[-2:])
        if two in known:
            words = words[:-2]
            continue
        if last in known:
            words = words[:-1]
            continue
        if len(last) >= 3 and any(_close(last, v) >= 0.8 for v in known if len(v) >= 3):
            words = words[:-1]
            continue
        break
    return " ".join(words)


def clean_instruction(text: str, wake_variants: set[str] | None = None,
                      stop_variants: set[str] | None = None, extra_names: str = "") -> str:
    """Everything above, in the order it has to happen."""
    spoken = (text or "").strip()
    spoken = strip_wake_word(spoken, wake_variants)
    spoken = strip_stop_word(spoken, stop_variants)
    spoken = correct_names(spoken, extra_names)
    return re.sub(r"\s+", " ", spoken).strip(" .,")
