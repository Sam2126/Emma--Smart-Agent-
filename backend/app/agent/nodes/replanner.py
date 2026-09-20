"""
Replan node — automatic retry with strategy mutation.

Reached only when the first attempt failed AND retrying is safe (verify node).
It reflects on why the attempt failed, stores that failure as an experience
right away (it is real evidence of an approach that does not work), asks the
planner for a new plan that avoids it, and hands back to the actor.

Example of what this is for:
  attempt 1: search box -> type query -> Enter -> blocked by a popup
  attempt 2: dismiss the popup first -> search -> success
and the lesson "dismiss popups on this site before searching" is remembered.
"""

from __future__ import annotations

from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig

from app.agent.context import run_context, schedule_background
from app.agent.nodes.planner import make_plan
from app.agent.prompts import retry_context_prompt
from app.agent.services import brain
from app.agent.state import AgentState, TaskStatus
from app.state.reflection import OutcomeReflection

logger = structlog.get_logger(__name__)


async def replan_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    run = run_context(config)
    run.notify("replanning", "🔁 First attempt failed, retrying with a different strategy...", 0.5)

    instruction = state["task_instruction"]
    trajectory = state.get("trajectory", [])
    error = state.get("error", "")
    report = state.get("final_report", "")
    experiences = state.get("experiences", [])

    reflection = await brain.reflection.reflect_on_outcome(
        instruction=instruction,
        trajectory=trajectory,
        success=False,
        error=error,
        raw_result=report,
        similar=experiences,
    )
    if reflection is None:
        template = brain._generate_reflection(
            task=instruction, trajectory=trajectory, success=False, error=error, raw_result=report
        )
        reflection = OutcomeReflection(
            what_succeeded=template.what_succeeded,
            what_failed=template.what_failed,
            root_cause=template.root_cause,
            what_to_try_next=template.what_to_try_next,
            from_llm=False,
            confidence=template.confidence,
        )

    task_id = state["task_id"]
    # A first attempt that ran no tool is not stored, like any run without steps
    # (learner). Found 2026-09-16: "I have no tool to find the latest PowerPoint"
    # from an attempt that never tried was saved as a lesson, and a lesson like
    # that is recalled for similar tasks and talks the agent into refusing again.
    if trajectory:
        schedule_background(
            brain.learn_semantic(
                task_id=f"{task_id}#attempt1",
                instruction=instruction,
                domain=state.get("domain", ""),
                scope=state.get("scope", "browser"),
                trajectory=trajectory,
                success=False,
                error=error,
                raw_result=report,
                reflection=reflection,
                similar=experiences,
            ),
            name=f"learn:{task_id}#attempt1",
        )

    steps = " -> ".join(
        f"{s.get('action_type')}({'ok' if s.get('success') else 'failed'})" for s in trajectory[:10]
    )
    retry_context = retry_context_prompt(error or report[:300], reflection.as_lesson(), steps)
    plan = await make_plan(state, retry_context=retry_context)
    logger.info("replanned_after_failure", task_id=task_id, lesson=reflection.as_lesson()[:200])

    return {
        "attempt": 2,
        "retried": True,
        "first_attempt_error": error,
        "first_attempt_trajectory": trajectory,
        "retry_context": retry_context,
        "plan": plan,
        "trajectory": [],
        "final_report": "",
        "failure_kind": "",
        "error": "",
        "success": False,
        "retry_blocked_reason": "",
        "status": TaskStatus.ACTING.value,
    }
