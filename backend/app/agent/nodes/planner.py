"""
Planner node — turns the instruction, memory and strategy into a short plan.

A planning failure never fails the task: the actor can still work from the
instruction alone, so an empty plan is passed on and the run continues.
"""

from __future__ import annotations

from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig

from app.agent.context import run_context
from app.agent.prompts import planner_system_prompt, planner_user_prompt
from app.agent.state import AgentState, TaskStatus
from app.agent.toolkit import strip_think
from app.utils.llm import chat

logger = structlog.get_logger(__name__)


async def make_plan(state: AgentState, retry_context: str = "") -> str:
    scope = state.get("scope", "browser")
    messages = [
        {"role": "system", "content": planner_system_prompt(scope)},
        {
            "role": "user",
            "content": planner_user_prompt(
                state["task_instruction"],
                scope,
                brain_context=state.get("brain_context", ""),
                strategy=state.get("strategy", ""),
                retry_context=retry_context,
            ),
        },
    ]
    try:
        response = await chat("planner", messages, max_tokens=1200, temperature=0.1, timeout=60)
        return strip_think(response.choices[0].message.content)
    except Exception as e:
        logger.warning("planner_failed", error=str(e)[:300])
        return ""


async def plan_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    run = run_context(config)
    run.notify("planning", "🗺️ Planning the steps...", 0.12)
    plan = await make_plan(state)
    if plan:
        run.notify("planning", "🗺️ Plan ready", 0.18, details=plan[:400])
    logger.info("plan_ready", task_id=state.get("task_id"), chars=len(plan))
    return {"plan": plan, "status": TaskStatus.ACTING.value}
