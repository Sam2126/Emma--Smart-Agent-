"""
Actor node — the native function-calling loop that actually does the work.

Each turn the model either calls a tool or writes the final report. Tool calls
run through toolkit.execute_tool_call, every call becomes a structured
trajectory step (this replaces CrewAI's event bus), and a progress line is
streamed to the UI before each call.

Guards:
  * a wall-clock deadline for the whole task (task_timeout_seconds)
  * a cap on tool steps per scope (actor_max_iterations_local / _browser)
  * two-tier models: routine turns use the fast model; the first turn, any turn
    right after a failed step, and every fifth turn use the action model. If the
    fast model errors, the next turn escalates instead of failing the task.
  * up to two nudges when the model returns an empty reply
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig

from app.agent.context import run_context
from app.agent.prompts import EMAIL_RULES, actor_system_prompt, actor_user_prompt
from app.agent.state import AgentState, FailureKind, TaskStatus
from app.agent.toolkit import (
    LOCAL_SCOPE,
    SendTracker,
    build_toolset,
    compact_messages,
    describe_action,
    describe_result,
    execute_tool_call,
    parse_arguments,
    strip_think,
    BROWSER_TOOL_NAMES,
    EMAIL_TOOL_NAMES,
    USE_BROWSER_TOOL,
    USE_EMAIL_TOOL,
    enable_tool_group,
    initial_tool_schemas,
    instruction_needs_email,
)
from app.config import get_settings
from app.utils.llm import actor_role_for_turn, chat, role_is_rate_limited

logger = structlog.get_logger(__name__)

ACTOR_MAX_OUTPUT_TOKENS = 2048
_MIN_OUTPUT_TOKENS = 1024
# Characters per token for these prompts, on the safe side: Groq's own counts
# for this agent's requests came to about 6.3-6.6 characters per token.
_CHARS_PER_TOKEN = 6.0


def output_token_budget(messages: list[dict[str, Any]], schemas: list[dict[str, Any]] | None) -> int:
    """max_tokens for one actor call: 2,048, or less when the request would pass Groq's per-request limit.

    Found 2026-09-17: an email task that also needed the browser asked Groq for
    8,573 tokens (input plus the 2,048 reserved for the answer); anything above
    8,000 is refused outright ("Request too large"), on every key.
    """
    limit = get_settings().groq_request_token_limit
    if limit <= 0:
        return ACTOR_MAX_OUTPUT_TOKENS
    estimate = (len(json.dumps(messages, default=str, ensure_ascii=False)) + len(json.dumps(schemas or []))) / _CHARS_PER_TOKEN
    return int(max(_MIN_OUTPUT_TOKENS, min(ACTOR_MAX_OUTPUT_TOKENS, limit - 150 - estimate)))


_NUDGE = (
    "Continue: call the next tool. If every part of the request is already done, "
    "reply with the final plain-text report instead."
)


async def act_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    run = run_context(config)
    settings = get_settings()
    scope = state.get("scope", "browser")
    instruction = state["task_instruction"]
    attempt = state.get("attempt", 1)

    if not run.toolset:
        run.toolset = await asyncio.to_thread(build_toolset, scope)
        run.tool_schemas, run.browser_tools_deferred = initial_tool_schemas(run.toolset, scope, instruction)

    max_tool_steps = (
        settings.actor_max_iterations_local if scope == LOCAL_SCOPE else settings.actor_max_iterations_browser
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": actor_system_prompt(
            scope, include_email=any(s["function"]["name"] in EMAIL_TOOL_NAMES for s in run.tool_schemas)
            or instruction_needs_email(instruction))},
        {"role": "user", "content": actor_user_prompt(instruction, scope, state.get("plan", ""), state.get("retry_context", ""))},
    ]

    trajectory: list[dict[str, Any]] = []
    final_report = ""
    failure_kind = FailureKind.NONE.value
    error = ""
    last_failed = False
    empty_replies = 0
    last_tool: str | None = None
    sends = SendTracker(instruction)
    blocked_repeats = 0
    base_progress = 0.2 if attempt == 1 else 0.55
    run.notify("acting", "⚙️ Working on it..." if attempt == 1 else "⚙️ Trying again with a new plan...", base_progress)

    turn = 0
    while True:
        if len(trajectory) >= max_tool_steps:
            failure_kind = FailureKind.STEP_LIMIT.value
            error = f"Reached the limit of {max_tool_steps} tool steps before finishing."
            break
        remaining = run.seconds_left()
        if remaining <= 3:
            failure_kind = FailureKind.TIMEOUT.value
            error = (
                f"Task timed out after {settings.task_timeout_seconds}s — step budget exceeded. "
                "The agent could not complete the task within the allowed time."
            )
            break

        role = actor_role_for_turn(turn, last_failed, last_tool)
        if role == "actor_fast" and role_is_rate_limited("actor_fast"):
            # Every key is cooling for the fast model; Groq limits each model
            # separately, so use the action model instead of waiting.
            role = "actor"
        turn += 1
        try:
            to_send = compact_messages(messages)
            response = await chat(
                role,
                to_send,
                tools=run.tool_schemas,
                max_tokens=output_token_budget(to_send, run.tool_schemas),
                temperature=0.0,
                timeout=min(remaining, 120),
            )
        except asyncio.TimeoutError:
            failure_kind = FailureKind.TIMEOUT.value
            error = f"The model did not respond in time (task budget {settings.task_timeout_seconds}s)."
            break
        except Exception as e:
            if role == "actor_fast":
                logger.warning("fast_actor_failed_escalating", error=str(e)[:200])
                last_failed = True
                continue
            failure_kind = FailureKind.EXCEPTION.value
            error = f"LLM call failed: {str(e)[:300]}"
            break

        choices = getattr(response, "choices", None) or []
        message = choices[0].message if choices else None
        tool_calls = list(getattr(message, "tool_calls", None) or []) if message is not None else []

        if not tool_calls:
            content = strip_think(getattr(message, "content", None) if message is not None else "")
            if content:
                final_report = content
                break
            empty_replies += 1
            if empty_replies > 2:
                failure_kind = FailureKind.EXCEPTION.value
                error = "The model returned empty replies instead of acting."
                break
            messages.append({"role": "user", "content": _NUDGE})
            last_failed = True
            continue
        empty_replies = 0

        calls = []
        for i, tc in enumerate(tool_calls):
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", "") or ""
            arguments = getattr(fn, "arguments", None)
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments or {})
            calls.append((getattr(tc, "id", None) or f"call_{turn}_{i}", name, arguments))

        messages.append({
            "role": "assistant",
            "content": getattr(message, "content", None) or None,
            "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
                for call_id, name, arguments in calls
            ],
        })

        for call_id, name, arguments in calls:
            args, _ = parse_arguments(arguments)
            progress = base_progress + min(0.3, 0.03 * (len(trajectory) + 1))
            if name == USE_BROWSER_TOOL:
                if run.browser_tools_deferred:
                    run.tool_schemas, added = enable_tool_group(
                        run.tool_schemas, run.toolset, BROWSER_TOOL_NAMES, USE_BROWSER_TOOL)
                    run.browser_tools_deferred = False
                    content = "Browser tools enabled: " + ", ".join(added) + "."
                else:
                    content = "Browser tools are already available."
                run.notify("acting", "🌐 Enabling browser tools", progress)
                messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": content})
                last_tool = name
                continue
            if name == USE_EMAIL_TOOL:
                run.tool_schemas, added = enable_tool_group(run.tool_schemas, run.toolset, EMAIL_TOOL_NAMES, USE_EMAIL_TOOL)
                content = (
                    ("Email tools enabled: " + ", ".join(added) + ".\n" + EMAIL_RULES) if added
                    else "Email tools are already available."
                )
                run.notify("acting", "📧 Enabling email tools", progress)
                messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": content})
                last_tool = name
                continue
            already_sent = sends.duplicate_of(name, args)
            if already_sent is not None:
                blocked_repeats += 1
                logger.info("duplicate_send_blocked", task_id=state.get("task_id"), text=already_sent[:80])
                run.notify("acting", f"🛑 Not sending '{already_sent[:50]}' again — it was already sent", progress)
                messages.append({
                    "role": "tool", "tool_call_id": call_id, "name": name,
                    "content": (
                        f"REFUSED: '{already_sent}' was already sent earlier in this task. "
                        "Doing it again would send a duplicate message. Do not retry this. "
                        "Write the final report now."
                    ),
                })
                continue
            run.notify("acting", describe_action(name, args), progress)
            result = await execute_tool_call(run.toolset, name, arguments)
            trajectory.append(result.record)
            output = result.output[:6000]
            if result.record["success"]:
                output += sends.record(name, args)
            messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": output})
            last_failed = not result.record["success"]
            last_tool = name
            if last_failed:
                run.notify("acting", f"⚠️ {name} did not work: {result.output[:140]}", progress)
            else:
                outcome = describe_result(name, result.output)
                if outcome:
                    run.notify("acting", outcome, progress)

        if blocked_repeats >= 2 and sends.sent:
            # The model keeps trying to resend: the work is done, stop here.
            sent_list = ", ".join(f"'{text}'" for text in sends.sent.values())
            final_report = (
                f"Completed the requested steps and sent {sent_list} exactly once. "
                "Stopped repeated attempts to send it again."
            )
            break

    logger.info(
        "act_complete",
        task_id=state.get("task_id"),
        attempt=attempt,
        tool_steps=len(trajectory),
        failure_kind=failure_kind or None,
        report_chars=len(final_report),
    )
    return {
        "trajectory": trajectory,
        "final_report": final_report,
        "failure_kind": failure_kind,
        "error": error,
        "status": TaskStatus.VERIFYING.value,
    }
