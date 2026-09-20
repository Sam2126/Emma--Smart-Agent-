"""
Verify node — decides whether the task really succeeded, never trusting the
actor's own report on its own.

Local tasks: the report must not describe a failure, AND the tools actually
used must match what was asked (an in-app request that only launched the app
is incomplete, whatever the report claims).

Browser tasks: independent verification against the live page — site or
generic rules, the text judge, and a vision judge on a screenshot of the final
page (see verifier/engine.py).

It also decides whether an automatic retry is allowed (see toolkit.retry_block_reason).
"""

from __future__ import annotations

import base64
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig

from app.agent.context import run_context
from app.agent.state import AgentState, FailureKind, TaskStatus
from app.agent.toolkit import (
    LOCAL_SCOPE,
    local_completion_shortfall,
    report_indicates_failure,
    retry_block_reason,
)
from app.config import get_settings

logger = structlog.get_logger(__name__)


def _action_summary(trajectory: list[dict[str, Any]], report: str) -> str:
    steps = "\n".join(
        f"{i + 1}. {s.get('action_type')}({str(s.get('input_value') or '')[:60]}) -> "
        f"{'ok' if s.get('success') else 'FAILED'}"
        for i, s in enumerate(trajectory[:25])
    )
    return f"{steps}\n\nAGENT REPORT:\n{report[:600]}"


async def _verify_browser(state: AgentState, report: str, trajectory: list[dict[str, Any]]) -> tuple[bool, str]:
    settings = get_settings()
    try:
        from app.browser.page_state import PageState, extract_page_state
        from app.browser.registry import get_browser_controller

        page = await get_browser_controller().get_active_page()
        final_state = await extract_page_state(page)
    except Exception as e:
        # No page to inspect: judge from the run itself. The report must not
        # describe a failure and at least one step must have worked.
        logger.warning("final_page_state_unavailable", error=str(e)[:200])
        passed = (
            bool(report)
            and any(s.get("success") for s in trajectory)
            and not report_indicates_failure(report[:400], trajectory)
        )
        return passed, "page state unavailable; judged from the run's own tool results"

    screenshot_b64 = None
    if settings.vision_verification_enabled:
        try:
            shot = await page.screenshot(type="jpeg", quality=60, full_page=False, timeout=10000)
            screenshot_b64 = base64.b64encode(shot).decode()
        except Exception as e:
            logger.info("verification_screenshot_failed", error=str(e)[:150])

    initial = state.get("initial_page_state") or {}
    initial_state = PageState(**initial) if initial else None

    from app.verifier.engine import VerificationEngine

    result = await VerificationEngine().verify_task(
        task_type="general",
        task_instruction=state["task_instruction"],
        final_state=final_state,
        action_summary=_action_summary(trajectory, report),
        initial_state=initial_state,
        screenshot_b64=screenshot_b64,
    )
    logger.info(
        "independent_ground_truth_verification",
        passed=result.passed,
        confidence=result.confidence,
        tier=result.tier.value,
        summary=result.summary,
    )
    return result.passed, result.summary


async def verify_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    run = run_context(config)
    settings = get_settings()
    scope = state.get("scope", "browser")
    instruction = state["task_instruction"]
    trajectory = state.get("trajectory", [])
    report = state.get("final_report", "")
    failure_kind = state.get("failure_kind", "")
    error = state.get("error", "")
    attempt = state.get("attempt", 1)
    run.notify("verifying", "🔎 Checking the result...", 0.9 if attempt == 1 else 0.93)

    summary = ""
    if failure_kind:
        success = False
        summary = error
    elif scope == LOCAL_SCOPE:
        success = bool(report) and not report_indicates_failure(report, trajectory)
        shortfall = local_completion_shortfall(instruction, trajectory)
        if success and shortfall:
            success = False
            failure_kind = FailureKind.INCOMPLETE.value
            error = f"Marked incomplete: {shortfall}"
            report = f"{report}\n\n[COMPLETION CHECK] Marked INCOMPLETE: {shortfall}"
            logger.info("local_task_marked_incomplete", task_id=state.get("task_id"), reason=shortfall)
        elif not success:
            failure_kind = FailureKind.VERIFICATION.value
            error = error or "The final report describes a failure."
        summary = "local: report and tool-usage check"
    else:
        success, summary = await _verify_browser(state, report, trajectory)
        if not success:
            failure_kind = FailureKind.VERIFICATION.value
            error = error or f"Verification failed: {summary}"

    blocked = ""
    if not success:
        if attempt >= 2:
            blocked = "it had already retried once"
        elif not settings.auto_retry_on_failure:
            blocked = "automatic retry is turned off"
        else:
            blocked = retry_block_reason(trajectory, failure_kind) or ""

    return {
        "success": success,
        "failure_kind": "" if success else failure_kind,
        "error": "" if success else error,
        "final_report": report,
        "verification_summary": summary,
        "retry_blocked_reason": blocked,
        "status": TaskStatus.LEARNING.value if (success or blocked) else TaskStatus.REPLANNING.value,
    }
