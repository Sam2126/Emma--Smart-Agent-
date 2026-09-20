"""
Learn node — records the outcome and builds the explanation.

Awaited (fast, local):
  * episodic task log
  * local-computer domain tip (local tasks)
  * SQL learning: domain memory, skills, failure patterns, brain stats

Background (one LLM call, so it must not delay the user's result):
  * LLM reflection on what happened, compared with similar past attempts
  * the experience written to semantic memory with that lesson

The user's 👍 / 👎 can arrive before the background write finishes; it is
queued and applied when the experience lands (semantic_memory.py).
"""

from __future__ import annotations

from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig

from app.agent.context import run_context, schedule_background
from app.agent.explain import build_explanation
from app.agent.services import brain, episodic_logger
from app.agent.state import AgentState, TaskStatus
from app.agent.toolkit import LOCAL_SCOPE

logger = structlog.get_logger(__name__)


async def learn_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    run = run_context(config)
    task_id = state["task_id"]
    instruction = state["task_instruction"]
    scope = state.get("scope", "browser")
    domain = state.get("domain") or ("local" if scope == LOCAL_SCOPE else "general")
    success = bool(state.get("success"))
    trajectory = state.get("trajectory", [])
    report = state.get("final_report", "")
    error = state.get("error", "")
    duration = run.elapsed()
    run.notify("learning", "📚 Saving what I learned...", 0.96)

    try:
        await episodic_logger.finalize_task(
            task_id=task_id,
            success=success,
            final_summary=(report or error)[:500],
            duration_seconds=duration,
            **({} if success else {"error": error or "Task failed"}),
        )
    except Exception as e:
        logger.warning("episodic_finalize_failed", error=str(e)[:200])

    if scope == LOCAL_SCOPE:
        try:
            from app.state.domain_memory import DomainMemoryStore

            tip = f"TASK: {instruction[:150]} -> {'DONE' if success else 'FAILED'}. REPORT: {(report or error)[:400]}"
            # The run is counted once, in brain.learn_from_task below.
            await DomainMemoryStore().record_domain_learning(
                "local", site_name="Local Computer", general_tips=tip, success=success, count_run=False
            )
        except Exception as e:
            logger.warning("local_learning_failed", error=str(e)[:200])

    try:
        await brain.learn_from_task(
            task_id=task_id,
            instruction=instruction,
            domain=domain,
            trajectory=trajectory,
            success=success,
            error=None if success else (error or "Task failed"),
            duration_seconds=duration,
            raw_result=report,
            persist_template_reflection=False,
        )
    except Exception as e:
        logger.warning("brain_post_learning_failed", error=str(e)[:200])

    if not trajectory:
        # Nothing was done, so there is no approach to remember. Found
        # 2026-09-16: a stray microphone recording was stored as a successful
        # experience with an empty tool list.
        logger.info("no_steps_nothing_to_learn", task_id=task_id)
        # Kept aside: if the user rates this run, the rating and note are stored.
        brain.note_unstored_run(
            task_id=task_id, instruction=instruction, domain=domain, scope=scope,
            success=success, error=error, report=report,
        )
        return _result(state, run, duration, success, error, report)

    schedule_background(
        brain.learn_semantic(
            task_id=task_id,
            instruction=instruction,
            domain=domain,
            scope=scope,
            trajectory=trajectory,
            success=success,
            error=None if success else error,
            raw_result=report,
            similar=state.get("experiences", []),
            # After an automatic retry, reflection diffs the failed first
            # attempt against this run: "what changed?"
            previous_attempt=(
                {
                    "trajectory": state.get("first_attempt_trajectory", []),
                    "error": state.get("first_attempt_error", ""),
                }
                if state.get("retried") else None
            ),
        ),
        name=f"learn:{task_id}",
    )

    return _result(state, run, duration, success, error, report)


def _result(state: AgentState, run, duration: float, success: bool, error: str, report: str) -> dict[str, Any]:
    """The task's outcome: the explanation, the duration and the final status."""
    trajectory = state.get("trajectory", [])
    explanation = build_explanation(
        trajectory=trajectory,
        success=success,
        experiences=state.get("experiences", []),
        strategy=state.get("strategy", ""),
        strategy_from_llm=bool(state.get("strategy_from_llm")),
        retried=bool(state.get("retried")),
        first_attempt_error=state.get("first_attempt_error", ""),
        retry_blocked_reason=state.get("retry_blocked_reason", ""),
        error=error,
    )
    run.notify(
        "completed" if success else "failed",
        "✅ Done" if success else "❌ Could not finish",
        1.0,
        details=(report or error)[:200],
    )
    return {
        "explanation": explanation,
        "duration_seconds": duration,
        "status": TaskStatus.COMPLETED.value if success else TaskStatus.FAILED.value,
    }
