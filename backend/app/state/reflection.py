"""
LLM-powered reflection (Level 3 learning).

Replaces the keyword templates in BrainMemory._generate_reflection with real
reasoning, in two places:

1. Before a task, `plan_strategy` reads the most similar past experiences
   (from semantic_memory) and writes a short STRATEGY / AVOID note for the
   planner: what worked on those tasks, what failed and why.
2. After a task, `reflect_on_outcome` looks at what actually happened — the
   tools used, the error, the final report, and similar past attempts — and
   writes the lesson stored with the experience. When a similar earlier
   attempt failed and this one worked, it is asked what changed.

Both are one short routed LLM call (role "reflection", see app.utils.llm), so
they share the Groq key pool, rate-limit handling and Gemini fallback with the
rest of the agent. Verified against this project's Groq account:
gpt-oss-120b with reasoning_effort="low" returned a clean STRATEGY / AVOID note
in 0.77 s.

Streaming: `plan_strategy(..., soft_deadline=...)` streams the reply and stops
reading at the deadline, so the planner can start with whatever part of the
strategy has arrived instead of waiting for the whole note.

Neither call can break a task. Each has a timeout, and on any failure the
caller gets a deterministic fallback built from the stored data instead.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import structlog

from app.config import get_settings
from app.state.semantic_memory import Experience, anti_skills, summarize_strategies, tool_sequence

logger = structlog.get_logger(__name__)

CompletionFn = Callable[..., Awaitable[Any]]

_THINK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


@dataclass
class StrategyAdvice:
    """Pre-task guidance distilled from similar past experiences."""

    text: str
    from_llm: bool
    experience_ids: list[str] = field(default_factory=list)
    partial: bool = False


@dataclass
class OutcomeReflection:
    """Post-task lesson. Field names match brain.Reflection on purpose."""

    what_succeeded: str
    what_failed: str
    root_cause: str
    what_to_try_next: str
    from_llm: bool = True
    confidence: float = 0.85

    def as_lesson(self) -> str:
        parts = []
        if self.what_succeeded and self.what_succeeded.lower() != "none":
            parts.append(f"Worked: {self.what_succeeded}")
        if self.what_failed and self.what_failed.lower() != "none":
            parts.append(f"Failed: {self.what_failed}")
        if self.root_cause and self.root_cause.lower() != "none":
            parts.append(f"Why: {self.root_cause}")
        if self.what_to_try_next and self.what_to_try_next.lower() != "none":
            parts.append(f"Next time: {self.what_to_try_next}")
        return " | ".join(parts)


_STRATEGY_SYSTEM = (
    "You are the long-term memory of a computer automation agent. You are given a new "
    "task and past tasks that are similar in meaning, with their outcomes. Write a short, "
    "concrete strategy for the new task based ONLY on that evidence. Reuse approaches that "
    "worked, name approaches that failed so they are avoided, and adapt names, apps and "
    "sites to the new task. When a flow is marked as confirmed by the user, learn from its "
    "shape and adapt it — do not copy it blindly. When the user said what went wrong, make "
    "avoiding exactly that the first bullet. Never invent tools. Plain text, no preamble."
)

_OUTCOME_SYSTEM = (
    "You analyse one finished run of a computer automation agent and extract the lesson "
    "that should be remembered for similar future tasks. Be specific to what actually "
    "happened in the tool log and report; do not speculate beyond it. Reply with ONE JSON "
    'object and nothing else: {"what_succeeded": str, "what_failed": str, '
    '"root_cause": str, "what_to_try_next": str}. Use "None" for a field that does not apply.'
)


def _strip_think(text: str) -> str:
    return _THINK.sub("", text or "").strip()


def _normalize_instruction(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def _format_experience(i: int, exp: Experience) -> str:
    outcome = "WORKED" if exp.effective_success else "FAILED"
    if exp.user_disputed:
        outcome = "USER SAID IT DID NOT WORK"
    if exp.user_confirmed:
        outcome = "USER SAID IT WORKED"
    lines = [
        f"PAST TASK {i} ({outcome}, similarity {exp.similarity:.2f}"
        + (f", user feedback {exp.feedback:+d}" if exp.feedback else "")
        + f"): {exp.instruction}",
        f"  steps: {exp.strategy_signature}",
    ]
    if exp.steps_detail:
        lines.append("  what it did: " + " -> ".join(exp.steps_detail[:8]))
    if exp.feedback_comment:
        lines.append(f'  the user said: "{exp.feedback_comment[:200]}"')
    if exp.lesson:
        lines.append(f"  lesson: {exp.lesson[:400]}")
    if exp.error and not exp.effective_success:
        lines.append(f"  error: {exp.error[:200]}")
    return "\n".join(lines)


def confirmed_flow_block(experiences: list[Experience]) -> str:
    """The steps of a run the user gave a 👍, as something to learn from."""
    confirmed = [e for e in experiences if e.verified and e.steps_detail]
    if not confirmed:
        return ""
    best = confirmed[0]
    return (
        "\nA FLOW THE USER CONFIRMED AS CORRECT on a similar task — learn from its shape and adapt "
        "the names, paths and details to the new task:\n"
        f'  "{best.instruction[:120]}": ' + " -> ".join(best.steps_detail[:10]) + "\n"
    )


def user_complaints_block(experiences: list[Experience]) -> str:
    """What the user said was wrong with past attempts (👎 notes)."""
    complaints = [e for e in experiences if e.feedback_comment and not e.effective_success]
    if not complaints:
        return ""
    lines = "\n".join(f'  - "{e.feedback_comment[:200]}" (on: {e.instruction[:70]})' for e in complaints[:3])
    return "\nWHAT THE USER SAID WENT WRONG BEFORE — do not repeat it:\n" + lines + "\n"


def fallback_strategy(experiences: list[Experience]) -> str:
    """Deterministic strategy note used when the LLM is unavailable."""
    lines: list[str] = []
    ranked = summarize_strategies(experiences)
    if ranked:
        lines.append("STRATEGY (from similar past tasks, no LLM available):")
        for signature, wins, total in ranked[:3]:
            lines.append(f"- Approach {signature}: worked {wins}/{total} time(s)")
    anti = anti_skills(experiences)
    if anti:
        lines.append("ANTI-SKILLS (kept failing on similar tasks, never use):")
        for signature, wins, total in anti[:2]:
            lines.append(f"- {signature}: worked {wins}/{total}")
    confirmed = confirmed_flow_block(experiences)
    if confirmed:
        lines.append(confirmed.strip())
    complaints = user_complaints_block(experiences)
    if complaints:
        lines.append(complaints.strip())
    avoid = [e for e in experiences if not e.effective_success and (e.lesson or e.error)]
    if avoid:
        lines.append("AVOID:")
        for exp in avoid[:2]:
            reason = exp.lesson or exp.error
            lines.append(f'- "{exp.instruction[:70]}" failed: {reason[:160]}')
    return "\n".join(lines)


def _format_steps(trajectory: list[dict[str, Any]], limit: int) -> list[str]:
    steps = []
    for i, step in enumerate((trajectory or [])[:limit]):
        if not isinstance(step, dict):
            continue
        mark = "ok" if step.get("success") else "FAILED"
        detail = step.get("error") or step.get("output") or ""
        steps.append(
            f"{i + 1}. {step.get('action_type')}({str(step.get('input_value') or '')[:80]}) "
            f"-> {mark} {str(detail)[:120]}"
        )
    return steps


def _parse_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = _strip_think(text)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


class ReflectionEngine:
    def __init__(self, completion_fn: CompletionFn | None = None) -> None:
        self._completion_fn = completion_fn
        self._strategy_cache: dict[str, tuple[float, StrategyAdvice]] = {}

    # ------------------------------------------------------------------
    # LLM plumbing
    # ------------------------------------------------------------------

    def _messages(self, system: str, user: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    async def _complete(self, system: str, user: str, max_tokens: int) -> str:
        from app.utils.llm import chat

        response = await chat(
            "reflection",
            self._messages(system, user),
            max_tokens=max_tokens,
            temperature=0.2,
            timeout=get_settings().reflection_timeout_seconds,
            completion_fn=self._completion_fn,
        )
        return _strip_think(response.choices[0].message.content or "")

    async def _complete_streaming(
        self, system: str, user: str, max_tokens: int, soft_deadline: float
    ) -> tuple[str, bool]:
        """Stream a reply, stop reading at `soft_deadline` (time.monotonic()).

        Returns (text received so far, whether the reply finished).
        """
        from app.utils.llm import chat

        stream = await chat(
            "reflection",
            self._messages(system, user),
            max_tokens=max_tokens,
            temperature=0.2,
            timeout=get_settings().reflection_timeout_seconds,
            completion_fn=self._completion_fn,
            stream=True,
        )
        parts: list[str] = []
        complete = False
        iterator = stream.__aiter__()
        while True:
            remaining = soft_deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                chunk = await asyncio.wait_for(iterator.__anext__(), timeout=remaining)
            except StopAsyncIteration:
                complete = True
                break
            except asyncio.TimeoutError:
                break
            choices = getattr(chunk, "choices", None) or []
            delta = getattr(choices[0], "delta", None) if choices else None
            piece = getattr(delta, "content", None) if delta is not None else None
            if piece:
                parts.append(piece)
        return _strip_think("".join(parts)), complete

    # ------------------------------------------------------------------
    # Pre-task strategy
    # ------------------------------------------------------------------

    async def plan_strategy(
        self,
        instruction: str,
        experiences: list[Experience],
        soft_deadline: float | None = None,
    ) -> StrategyAdvice | None:
        if not experiences:
            return None
        settings = get_settings()
        ids = [e.task_id for e in experiences]
        key = "|".join([
            _normalize_instruction(instruction),
            *sorted(f"{e.task_id}:{e.feedback}:{int(e.user_disputed)}" for e in experiences),
        ])
        cached = self._strategy_cache.get(key)
        if cached and time.monotonic() - cached[0] < settings.reflection_cache_ttl_seconds:
            logger.info("strategy_cache_hit")
            return cached[1]

        if not settings.llm_reflection_enabled:
            return StrategyAdvice(fallback_strategy(experiences), from_llm=False, experience_ids=ids)

        ranked = summarize_strategies(experiences)
        approach_lines = "\n".join(f"- {sig}: worked {wins}/{total}" for sig, wins, total in ranked)
        anti = anti_skills(experiences)
        anti_lines = (
            "ANTI-SKILLS (approaches that kept failing on similar tasks — never use them):\n"
            + "\n".join(f"- {sig}: worked {wins}/{total}" for sig, wins, total in anti)
            + "\n"
        ) if anti else ""
        user = (
            f"NEW TASK: {instruction}\n\n"
            + "\n".join(_format_experience(i + 1, e) for i, e in enumerate(experiences))
            + f"\n\nAPPROACH SUCCESS RATES ACROSS THESE TASKS:\n{approach_lines}\n"
            + anti_lines
            + confirmed_flow_block(experiences)
            + user_complaints_block(experiences)
            + "\nWrite exactly:\nSTRATEGY:\n- up to 4 short bullets\n"
            "AVOID:\n- up to 2 short bullets (name every anti-skill)"
        )
        try:
            if soft_deadline is not None:
                text, complete = await self._complete_streaming(_STRATEGY_SYSTEM, user, 500, soft_deadline)
                partial = not complete
            else:
                text, partial = await self._complete(_STRATEGY_SYSTEM, user, max_tokens=500), False
            if "STRATEGY" not in text.upper():
                raise ValueError("strategy reply missing STRATEGY section")
            advice = StrategyAdvice(text=text[:1500], from_llm=True, experience_ids=ids, partial=partial)
        except Exception as e:
            logger.warning("strategy_reflection_failed", error=str(e)[:200])
            advice = StrategyAdvice(fallback_strategy(experiences), from_llm=False, experience_ids=ids)

        if not advice.partial:
            self._strategy_cache[key] = (time.monotonic(), advice)
        return advice

    # ------------------------------------------------------------------
    # Post-task reflection
    # ------------------------------------------------------------------

    async def reflect_on_outcome(
        self,
        *,
        instruction: str,
        trajectory: list[dict[str, Any]],
        success: bool,
        error: str | None,
        raw_result: str,
        similar: list[Experience] | None = None,
        previous_attempt: dict[str, Any] | None = None,
    ) -> OutcomeReflection | None:
        """One LLM call that turns this run into a lesson. None on any failure.

        previous_attempt: {"trajectory", "error"} of this same task's failed
        first attempt when this run was the automatic retry. The model diffs
        the two runs, so a retry that worked teaches what changed.
        """
        if not get_settings().llm_reflection_enabled:
            return None

        steps = _format_steps(trajectory, limit=20)

        comparison = ""
        if previous_attempt:
            first_steps = _format_steps(previous_attempt.get("trajectory") or [], limit=12)
            comparison += (
                f"\nTHIS SAME TASK'S FIRST ATTEMPT FAILED; THIS RETRY {'SUCCEEDED' if success else 'ALSO FAILED'}.\n"
                f"First attempt error: {str(previous_attempt.get('error') or 'none')[:300]}\n"
                f"First attempt tool log:\n{chr(10).join(first_steps) or '(no tool calls)'}\n"
                + (
                    "Compare the two tool logs: what_succeeded must name what this retry did differently "
                    "that made it work, and what_to_try_next must say what to do FIRST next time.\n"
                    if success
                    else "Say what both attempts got wrong and what to try instead.\n"
                )
            )
        prior_failures = [e for e in (similar or []) if not e.effective_success]
        if success and prior_failures:
            comparison += (
                "\nEARLIER SIMILAR ATTEMPTS THAT FAILED (explain what this run did differently):\n"
                + "\n".join(_format_experience(i + 1, e) for i, e in enumerate(prior_failures[:2]))
            )

        user = (
            f"TASK: {instruction}\n"
            f"OUTCOME: {'SUCCESS' if success else 'FAILURE'}\n"
            f"ERROR: {(error or 'none')[:300]}\n"
            f"TOOL LOG:\n{chr(10).join(steps) or '(no tool calls captured)'}\n"
            f"FINAL REPORT (truncated):\n{(raw_result or '')[:1200]}"
            f"{comparison}"
        )
        try:
            text = await self._complete(_OUTCOME_SYSTEM, user, max_tokens=500)
            data = _parse_json_object(text)
            if not data:
                raise ValueError("reflection reply was not a JSON object")
            return OutcomeReflection(
                what_succeeded=str(data.get("what_succeeded") or "None")[:300],
                what_failed=str(data.get("what_failed") or "None")[:300],
                root_cause=str(data.get("root_cause") or "")[:300],
                what_to_try_next=str(data.get("what_to_try_next") or "")[:300],
                from_llm=True,
                confidence=0.9 if success else 0.85,
            )
        except Exception as e:
            logger.warning("outcome_reflection_failed", error=str(e)[:200])
            return None


def trajectory_tools(trajectory: list[dict[str, Any]]) -> list[str]:
    """Collapsed tool list for a trajectory (re-exported for callers)."""
    return tool_sequence(trajectory)


_engine: ReflectionEngine | None = None


def get_reflection_engine() -> ReflectionEngine:
    global _engine
    if _engine is None:
        _engine = ReflectionEngine()
    return _engine
