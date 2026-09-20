"""
Explainability: "why I did what I did", shown to the user after every task.

Built only from facts the run recorded — the tools actually called, the past
experiences that were recalled, the strategy that was followed, and whether a
retry happened. No extra LLM call, so it adds no latency and cannot invent a
rationale the agent did not really use.
"""

from __future__ import annotations

from typing import Any

from app.state.semantic_memory import anti_skills, tool_sequence


def _step_label(step: dict[str, Any]) -> str:
    mark = "✓" if step.get("success") else "✗"
    target = str(step.get("input_value") or "").strip()
    if target in ("", "{}"):
        return f"{mark} {step.get('action_type')}"
    return f"{mark} {step.get('action_type')} ({target[:40]})"


def _strategy_bullets(strategy: str, limit: int = 3) -> list[str]:
    bullets = [line.strip() for line in strategy.splitlines() if line.strip().startswith(("-", "•", "*"))]
    return [b.lstrip("-•* ").strip() for b in bullets[:limit]]


def build_explanation(
    *,
    trajectory: list[dict[str, Any]],
    success: bool,
    experiences: list[Any],
    strategy: str,
    strategy_from_llm: bool,
    retried: bool,
    first_attempt_error: str,
    retry_blocked_reason: str,
    error: str,
) -> str:
    lines: list[str] = []

    steps = [s for s in trajectory or [] if isinstance(s, dict)]
    if steps:
        shown = " → ".join(_step_label(s) for s in steps[:12])
        more = f" … and {len(steps) - 12} more" if len(steps) > 12 else ""
        lines.append(f"What I did: {shown}{more}")
    else:
        lines.append("What I did: no tool actions completed.")

    if experiences:
        lines.append(
            f"Why: this matched {len(experiences)} similar past task(s), so I reused what worked and avoided what failed:"
        )
        for exp in experiences[:3]:
            if exp.user_confirmed:
                verdict = "you said it worked"
            elif exp.effective_success:
                verdict = "worked"
            else:
                verdict = "you said it did not work" if exp.user_disputed else "failed"
            thumbs = " 👍" if exp.feedback > 0 else (" 👎" if exp.feedback < 0 else "")
            match = (
                f"its lesson is {exp.similarity:.0%} relevant" if exp.matched_on == "lesson"
                else f"{exp.similarity:.0%} similar"
            )
            lines.append(f'• "{exp.instruction[:70]}" ({verdict}, {exp.age_text()}, {match}){thumbs}')
        confirmed = next((exp for exp in experiences if getattr(exp, "verified", False)), None)
        if confirmed is not None:
            lines.append(f'I learned the shape of this from the run you approved: "{confirmed.instruction[:70]}".')
        for exp in experiences[:3]:
            if exp.feedback_comment and not exp.effective_success:
                lines.append(f'I avoided what you reported last time: "{exp.feedback_comment[:120]}"')
                break
        used = " -> ".join(tool_sequence(steps)) if steps else ""
        for signature, wins, total in anti_skills(experiences)[:2]:
            if signature != used:
                lines.append(f"Avoided: {signature} — it worked only {wins} of {total} times on similar tasks.")
        bullets = _strategy_bullets(strategy)
        if bullets:
            source = "" if strategy_from_llm else " (from stored results, reflection model unavailable)"
            lines.append(f"Strategy I followed{source}: " + "; ".join(bullets))
    else:
        lines.append(
            "Why: nothing similar was in my memory yet, so I planned from scratch. "
            "This run is now stored, so similar tasks will reuse what happened here."
        )

    if retried:
        lines.append(f"My first attempt failed ({first_attempt_error[:140]}), so I retried once with a changed plan.")
    elif not success and retry_blocked_reason:
        lines.append(f"I did not retry automatically because {retry_blocked_reason}.")

    if not success and error:
        lines.append(f"What went wrong: {error[:220]}")

    return "\n".join(lines)
