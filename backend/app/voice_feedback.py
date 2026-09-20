"""
Rating the last task by voice.

The windows show 👍 / 👎 under every result, but a hands-free task may finish
while no window is open. Saying the rating works the same way:

    "Emma … good job … done"                       -> 👍
    "Emma … wrong, you opened my own Chrome … done" -> 👎 with that note
    "Emma … feedback bad, use install_app … done"   -> 👎 with that note

(Emma is the wake word; the listener removes it before this module sees the text.)

The rating goes to the most recently finished task (from any window, the wake
word or REST) if it finished within the last 15 minutes, and it is stored
exactly like a click on 👍 / 👎: the note is shown to the planner next time it
plans something similar.

An utterance counts as feedback only when it STARTS with a rating phrase that
is followed by a pause (punctuation), the end, or words that begin an
explanation ("because", "you ..."). "Correct the spelling in notes.txt" and
"wrong number on the invoice, fix it" stay tasks.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

FEEDBACK_WINDOW_SECONDS = 15 * 60

_POSITIVE = (
    "good job", "great job", "nice job", "well done", "nice work", "good work", "great work",
    "perfect", "excellent", "correct", "that's right", "that is right", "that was right",
    "that's correct", "that is correct", "that was correct", "you did it right", "thumbs up",
    "it worked", "that worked", "shabash", "bahut badhiya", "bahut accha", "sahi hai", "bilkul sahi",
)
_NEGATIVE = (
    "wrong", "that's wrong", "that is wrong", "that was wrong", "you did it wrong", "not correct",
    "incorrect", "not right", "thumbs down", "bad job", "it didn't work", "it did not work",
    "that didn't work", "that did not work", "you failed", "galat", "yeh galat hai", "ye galat hai",
)
# Phrases are tried longest first, so "not correct" wins over "correct".
_PHRASES = sorted([(p, 1) for p in _POSITIVE] + [(p, -1) for p in _NEGATIVE], key=lambda item: -len(item[0]))
_LEAD_IN = re.compile(r"^(?:\s*(?:okay|ok|so|hey|hmm|no|nope|yes|yeah)\b[\s,.!]*)+")
# What may follow a rating phrase: a pause, the end, or the start of an explanation.
_AFTER_PHRASE = re.compile(
    r"\s*(?:[,.!;:?\-]|$)"
    r"|\s+(?:because|but|you|next time|this time|i wanted|i asked|i said|i told you|"
    r"it should|it was|it opened|it did|it used|it sent|it typed)\b"
)
_EXPLICIT = re.compile(r"^\s*feedback\b[\s:,.\-]*")
_EXPLICIT_RATING = re.compile(
    r"(?P<up>thumbs up|good|great|positive|correct|right|yes)\b|(?P<down>thumbs down|bad|negative|wrong|incorrect|no)\b"
)


@dataclass(frozen=True)
class VoiceFeedback:
    rating: int  # 1, -1, or 0 when "feedback" was said without a rating
    note: str


def parse_voice_feedback(text: str) -> VoiceFeedback | None:
    """The rating in a spoken sentence, or None when the sentence is a task."""
    original = text or ""
    # Same length as the original, so positions map back and the note keeps its capitals.
    lowered = original.replace("’", "'").lower()
    start = _LEAD_IN.match(lowered).end() if _LEAD_IN.match(lowered) else 0
    rest = lowered[start:]

    explicit = _EXPLICIT.match(rest)
    if explicit:
        pos = start + explicit.end()
        rating_match = _EXPLICIT_RATING.match(lowered, pos)
        if not rating_match:
            return VoiceFeedback(0, "")
        rating = 1 if rating_match.group("up") else -1
        return VoiceFeedback(rating, _clean_note(original[rating_match.end():]))

    for phrase, rating in _PHRASES:
        if not rest.startswith(phrase):
            continue
        after = start + len(phrase)
        if not _AFTER_PHRASE.match(lowered, after):
            continue
        return VoiceFeedback(rating, _clean_note(original[after:]))
    return None


def _clean_note(text: str) -> str:
    return text.strip().lstrip(",.!;:?- ").strip()[:300]


async def record_voice_feedback(feedback: VoiceFeedback, now: float | None = None) -> dict[str, Any]:
    """Store a spoken rating on the most recently finished task."""
    from app.agent.runner import last_finished_task
    from app.agent.services import brain

    if feedback.rating == 0:
        return {
            "applied": False, "queued": False, "task_id": "", "rating": 0,
            "message": "Say 'feedback good' or 'feedback bad', then what went wrong.",
        }
    last = last_finished_task()
    now = time.time() if now is None else now
    if last is None or now - last["finished_at"] > FEEDBACK_WINDOW_SECONDS:
        return {
            "applied": False, "queued": False, "task_id": "", "rating": feedback.rating,
            "message": "There is no task from the last 15 minutes to rate.",
        }
    result = await brain.record_feedback(last["task_id"], feedback.rating, feedback.note)
    result["task_id"] = last["task_id"]
    result["instruction"] = last["instruction"]
    return result
