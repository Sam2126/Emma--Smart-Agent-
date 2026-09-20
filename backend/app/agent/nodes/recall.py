"""
Recall node — everything the planner needs before the first step.

Runs concurrently, because none of these depend on each other:
  * SQL memory: domain selectors, learned skills, failure patterns
    (for local tasks: the tips learned from past operations on this computer)
  * semantic memory: past experiences similar in meaning to this instruction
  * the tool set for this task's scope
  * the browser page's starting state (browser tasks only; local tasks must
    never wake the browser)

Then, when similar experiences exist, the reflection engine writes a
STRATEGY / AVOID note. It streams, and the planner gets whatever has arrived
by STRATEGY_GRACE_SECONDS, so a slow reflection never stalls the task.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig

from app.agent.context import run_context
from app.agent.services import brain
from app.agent.state import AgentState, TaskStatus
from app.agent.toolkit import LOCAL_SCOPE, build_toolset, initial_tool_schemas

logger = structlog.get_logger(__name__)

STRATEGY_GRACE_SECONDS = 4.0


async def _sql_context(instruction: str, scope: str) -> str:
    if scope == LOCAL_SCOPE:
        try:
            from app.state.domain_memory import DomainMemoryStore

            memory = await DomainMemoryStore().get_domain_memory("local")
            learned = memory.get("general_tips") or ""
            if learned:
                # New tips are appended at the end, so keep the LAST 2000
                # chars — otherwise the oldest lessons always win.
                return f"--- LEARNED FROM PAST LOCAL OPERATIONS ---\n{learned[-2000:]}\n--- END ---"
        except Exception as e:
            logger.warning("local_memory_recall_failed", error=str(e)[:200])
        return ""
    try:
        return await brain.recall_for_task(instruction)
    except Exception as e:
        logger.warning("brain_recall_failed", error=str(e)[:200])
        return "No prior experience available."


async def _initial_page_state(scope: str) -> dict[str, Any]:
    if scope == LOCAL_SCOPE:
        return {}
    try:
        from app.browser.page_state import extract_page_state
        from app.browser.registry import get_browser_controller

        page = await get_browser_controller().get_active_page()
        return asdict(await extract_page_state(page))
    except Exception as e:
        logger.info("initial_page_state_unavailable", error=str(e)[:150])
        return {}


async def recall_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    run = run_context(config)
    instruction = state["task_instruction"]
    scope = state.get("scope", "browser")
    run.notify("thinking", "🧠 Recalling past experience...", 0.05)

    domain = "local" if scope == LOCAL_SCOPE else brain._extract_domain_from_instruction(instruction)

    sql_context, experiences, toolset, initial_state = await asyncio.gather(
        _sql_context(instruction, scope),
        brain.recall_semantic(instruction, exclude_task_id=state["task_id"]),
        asyncio.to_thread(build_toolset, scope),
        _initial_page_state(scope),
    )
    run.toolset = toolset
    run.tool_schemas, run.browser_tools_deferred = initial_tool_schemas(toolset, scope, instruction)

    strategy, from_llm, partial = "", False, False
    if experiences:
        run.notify("thinking", f"🧠 Found {len(experiences)} similar past task(s), building a strategy...", 0.08)
        advice = await brain.reflection.plan_strategy(
            instruction, experiences, soft_deadline=time.monotonic() + STRATEGY_GRACE_SECONDS
        )
        if advice:
            strategy, from_llm, partial = advice.text, advice.from_llm, advice.partial

    logger.info(
        "recall_complete",
        task_id=state["task_id"],
        experiences=len(experiences),
        strategy_chars=len(strategy),
        strategy_from_llm=from_llm,
        strategy_partial=partial,
        tools=len(run.tool_schemas),
        browser_tools_deferred=run.browser_tools_deferred,
        elapsed=round(run.elapsed(), 2),
    )
    return {
        "domain": domain,
        "brain_context": sql_context,
        "experiences": experiences,
        "strategy": strategy,
        "strategy_from_llm": from_llm,
        "strategy_partial": partial,
        "initial_page_state": initial_state,
        "status": TaskStatus.PLANNING.value,
    }
