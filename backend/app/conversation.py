"""
Telling a question apart from a job, and answering the questions.

Added 2026-09-20. Everything said to Emma used to become a TASK: planned,
acted on with tools, verified, learned from. Ask her "what did you just do?"
and she would open a browser and start working. A person asked a question
answers it; only a job gets done.

    "open myntra and filter under 1000"   -> TASK      (plan, act, verify)
    "what did you just do?"               -> QUESTION  (answer, out loud)
    "how does the wake word work?"        -> QUESTION
    "send shrey an email"                 -> TASK

Deciding costs nothing in the ordinary case: the wording of a question is
distinctive, so a small set of patterns settles most of it, and the model is
asked only when the patterns disagree with each other. A question is answered
in one call, in one or two sentences, because it is going to be SPOKEN - a
paragraph that reads well is exhausting to listen to.

When a question needs something only the computer can tell (what is in a
folder, how many unread emails), it is NOT a question for this purpose: it
needs tools, so it goes back to the task pipeline. `classify` errs that way -
a job wrongly answered is useless, while a question wrongly done as a job is
merely slow.
"""

from __future__ import annotations

import re

import structlog

from app.utils.llm import chat

logger = structlog.get_logger(__name__)

# Plain question openings.
_ASKS = re.compile(
    r"^\s*(?:what|what's|why|how|when|where|who|which|whose|whom|"
    r"is|are|was|were|do|does|did|am|can|could|will|would|have|has|"
    r"tell me|explain|describe)\b",
    re.IGNORECASE,
)

# Words that mean work. "What is the cheapest shirt on Myntra" is a job.
_DOES_WORK = re.compile(
    r"\b(open|launch|start|run|click|type|search|find|download|upload|"
    r"send|email|mail|write|create|make|delete|remove|move|copy|install|play|buy|order|"
    r"add to cart|filter|book|fill|sign in|log in|navigate|go to|visit|scroll|check)\b",
    re.IGNORECASE,
)

# The past tense turns a work word into a question about what already happened:
# "did you open chrome" asks, "open chrome" orders.
_PAST = re.compile(r"\b(did|have you|had|was|were|were you)\b", re.IGNORECASE)

# About Emma herself - always answerable by talking.
_ABOUT_EMMA = re.compile(
    # "are you listening", "did you send it"
    r"^\s*(?:are|can|do|did|have|has|will|could|would)\s+(?:you|your)\b"
    # "what did you just do", "how do you work" - an auxiliary sits in between
    r"|^\s*(?:what|who|how|why|when)\s+(?:did|do|does|are|can|have|has|will|would|could)\s+(?:you|your)\b"
    r"|\byourself\b|\bwhat can you do\b|\bwho are you\b",
    re.IGNORECASE,
)

# About the thing that just happened, which the agent can answer from memory.
_ABOUT_LAST = re.compile(
    r"\b(that|this|it|the task|just now|last one|previous)\b", re.IGNORECASE
)

# General knowledge or how-things-work, with no subject on this computer.
_GENERAL = re.compile(
    r"^\s*(?:what is|what are|what's|how does|how do|how can|why is|why are|why does|"
    r"tell me about|explain|describe|define)\b",
    re.IGNORECASE,
)

# Talk that is not a question and not a job: saying hello, thanking her,
# saying goodbye. A real run sent "hello what's up" through the whole planning
# pipeline and came back asking what the user wanted done with the phrase.
# One greeting, or several in a row, optionally addressing her by name:
# "hello", "hey emma", "hello whatsup", "thanks emma". A real run sent
# "hello what's up" through the whole planning pipeline, and the agent came
# back asking what the user wanted done with the phrase.
_GREETING = (
    r"(?:hi+|hey+|hello+|yo|sup|namaste|emma|"
    r"good\s+(?:morning|afternoon|evening|night)|"
    r"what'?s?\s*up|whats?up|wassup|"
    r"how(?:'s|\s+is|\s+are)\s+(?:it\s+going|you|things|u)|how\s+r\s+u|"
    r"thanks?(?:\s+you)?|thank\s+you|thx|ty|"
    r"bye|goodbye|see\s+you|nice|cool|okay|ok|great|awesome)"
)
_SMALL_TALK = re.compile(rf"^\s*(?:{_GREETING}[\s,!.?]*)+$", re.IGNORECASE)

QUESTION = "question"
TASK = "task"
UNSURE = "unsure"


def classify(instruction: str) -> str:
    """QUESTION, TASK, or UNSURE when only the model can tell.

    Wording alone cannot settle every case. "What did Priya say about
    tomorrow?" is phrased as a question but needs her mail opened and read, so
    it is work; "what did you just do?" is the same shape and is not. Those go
    to `route`, which asks the model.
    """
    text = (instruction or "").strip()
    if not text:
        return TASK

    if _SMALL_TALK.match(text):
        return QUESTION

    asks = bool(_ASKS.search(text)) or text.endswith("?")
    works = bool(_DOES_WORK.search(text))
    past = bool(_PAST.search(text))

    # An order is an order, however politely it opens: "can you open whatsapp".
    if works and not past:
        return TASK
    if _ABOUT_EMMA.search(text):
        return QUESTION
    if asks and _ABOUT_LAST.search(text):
        return QUESTION
    if asks and _GENERAL.search(text) and not works:
        return QUESTION
    if asks:
        return UNSURE
    return TASK


_ROUTER_SYSTEM = (
    "Decide whether the user wants the assistant to DO something on their computer, "
    "or is ASKING something it can answer by talking.\n"
    "Answer with one word: TASK or QUESTION.\n"
    "TASK: anything needing the browser, files, apps, or the user's own data - including "
    "questions whose answer must be looked up ('what did Priya say about tomorrow' means "
    "reading her mail, so TASK).\n"
    "QUESTION: about the assistant itself, about what it just did, general knowledge, or "
    "anything answerable in a sentence without opening anything.\n"
    "When unsure, answer TASK."
)


async def route(instruction: str, conversation: bool = False) -> str:
    """QUESTION or TASK, asking the model only when the wording is ambiguous.

    `conversation` is set by the talking microphone. It does not change what a
    clear instruction does - "open Myntra" still opens Myntra - but when the
    wording could go either way, someone in the middle of a conversation
    expects an answer rather than a browser window.
    """
    decided = classify(instruction)
    if decided != UNSURE:
        return decided
    try:
        response = await chat(
            "actor_fast",
            [{"role": "system", "content": _ROUTER_SYSTEM},
             {"role": "user", "content": instruction}],
            max_tokens=5,
            temperature=0.0,
        )
        word = (response.choices[0].message.content or "").strip().upper()
    except Exception as e:
        logger.warning("conversation_route_failed", error=str(e)[:160])
        return QUESTION if conversation else TASK
    if word.startswith("QUESTION"):
        return QUESTION
    return TASK


ANSWER_SYSTEM = (
    "You are Emma, a voice assistant running on the user's own Windows computer. "
    "You are ANSWERING A QUESTION out loud, not performing a task.\n"
    "Rules:\n"
    "- Answer in one or two short sentences. This is spoken aloud, so no lists, "
    "no markdown, no headings, no code.\n"
    "- Speak naturally, the way a person answers a question in a room.\n"
    "- If you genuinely do not know, or the answer needs you to look at the computer "
    "or the internet, say so in one sentence and offer to go and do it.\n"
    "- Never invent what you did, what you found, or what is on the user's computer.\n"
    "- If the user is just greeting you or thanking you, greet them back in a few words "
    "and offer to help. Do not explain what you are or list what you can do."
)


async def answer(instruction: str, recent: str = "") -> str:
    """One spoken-length answer to a question."""
    messages = [{"role": "system", "content": ANSWER_SYSTEM}]
    if recent:
        messages.append({
            "role": "system",
            "content": f"What you did most recently, for questions about it:\n{recent[:800]}",
        })
    messages.append({"role": "user", "content": instruction})

    try:
        response = await chat("actor_fast", messages, max_tokens=160, temperature=0.3)
        text = (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("conversation_answer_failed", error=str(e)[:200])
        return "Sorry, I could not think of an answer just now."

    # A model asked for two sentences sometimes writes five; spoken, that is a
    # lecture. Keep the first three at most.
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    return " ".join(parts[:3]).strip() or text
