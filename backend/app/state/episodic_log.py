"""
Episodic log — records task trajectories for later analysis and learning.

Every completed task (success or failure) gets logged with its full trajectory:
instruction → plan → actions → verification results → outcome.

Phase 1 will use this data to build skill memory (Qdrant) and eventually
train LoRA adapters (Phase 4).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import structlog

from app.state.database import get_session
from app.state.models import TaskRecord, StepRecord, VerificationRecord

logger = structlog.get_logger(__name__)


class EpisodicLogger:
    """
    Logs complete task trajectories to the database.

    Usage:
        logger = EpisodicLogger()
        await logger.log_task(task_id, instruction, ...)
        await logger.log_step(task_id, step_index, ...)
        await logger.finalize_task(task_id, success, ...)
    """

    async def log_task_start(
        self,
        task_id: str,
        instruction: str,
        task_type: str = "",
        site: str = "amazon.in",
        plan: list[dict] | None = None,
    ) -> None:
        """Log the start of a new task."""
        try:
            session = await get_session()
            async with session.begin():
                record = TaskRecord(
                    id=task_id,
                    instruction=instruction,
                    task_type=task_type,
                    site=site,
                    status="planning",
                    plan_json=json.dumps(plan) if plan else None,
                )
                session.add(record)

            logger.info("task_logged_start", task_id=task_id)

        except Exception as e:
            logger.error("task_log_start_failed", task_id=task_id, error=str(e))

    async def log_step(
        self,
        task_id: str,
        step_index: int,
        action_type: str,
        selector_used: str = "",
        input_value: str = "",
        success: bool = False,
        error: str | None = None,
        url_before: str = "",
        url_after: str = "",
        page_state_before: dict | None = None,
        page_state_after: dict | None = None,
        verification_tier: str = "",
        verification_passed: bool = False,
        verification_confidence: float = 0.0,
        verification_summary: str = "",
        verification_details: dict | None = None,
    ) -> None:
        """Log a single action step within a task."""
        try:
            session = await get_session()
            async with session.begin():
                step = StepRecord(
                    task_id=task_id,
                    step_index=step_index,
                    action_type=action_type,
                    selector_used=selector_used,
                    input_value=input_value,
                    success=success,
                    error=error,
                    url_before=url_before,
                    url_after=url_after,
                    page_state_before_json=(
                        json.dumps(_truncate_state(page_state_before))
                        if page_state_before else None
                    ),
                    page_state_after_json=(
                        json.dumps(_truncate_state(page_state_after))
                        if page_state_after else None
                    ),
                )
                session.add(step)
                await session.flush()  # Get the step ID

                # Log verification if provided
                if verification_tier:
                    verification = VerificationRecord(
                        step_id=step.id,
                        tier=verification_tier,
                        passed=verification_passed,
                        confidence=verification_confidence,
                        summary=verification_summary,
                        details_json=(
                            json.dumps(verification_details)
                            if verification_details else None
                        ),
                    )
                    session.add(verification)

            logger.debug(
                "step_logged",
                task_id=task_id,
                step_index=step_index,
                success=success,
            )

        except Exception as e:
            logger.error("step_log_failed", task_id=task_id, step_index=step_index, error=str(e))

    async def finalize_task(
        self,
        task_id: str,
        success: bool,
        status: str = "",
        error: str | None = None,
        final_summary: str = "",
        duration_seconds: float | None = None,
    ) -> None:
        """Mark a task as completed (success or failure)."""
        try:
            session = await get_session()
            async with session.begin():
                from sqlalchemy import select
                result = await session.execute(
                    select(TaskRecord).where(TaskRecord.id == task_id)
                )
                record = result.scalar_one_or_none()

                if record:
                    record.status = status or ("completed" if success else "failed")
                    record.error = error
                    record.final_summary = final_summary
                    record.completed_at = datetime.now(timezone.utc)
                    record.duration_seconds = duration_seconds

            logger.info(
                "task_finalized",
                task_id=task_id,
                success=success,
                duration=f"{duration_seconds:.1f}s" if duration_seconds else "?",
            )

        except Exception as e:
            logger.error("task_finalize_failed", task_id=task_id, error=str(e))

    async def get_task_history(self, limit: int = 20) -> list[dict]:
        """Retrieve recent task records for display in the extension."""
        try:
            session = await get_session()
            async with session.begin():
                from sqlalchemy import select
                result = await session.execute(
                    select(TaskRecord)
                    .order_by(TaskRecord.created_at.desc())
                    .limit(limit)
                )
                records = result.scalars().all()

                return [
                    {
                        "id": r.id,
                        "instruction": r.instruction,
                        "task_type": r.task_type,
                        "site": r.site,
                        "status": r.status,
                        "created_at": r.created_at.isoformat() if r.created_at else None,
                        "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                        "duration_seconds": r.duration_seconds,
                        "final_summary": r.final_summary,
                    }
                    for r in records
                ]

        except Exception as e:
            logger.error("get_history_failed", error=str(e))
            return []


def _truncate_state(state: dict) -> dict:
    """Truncate page state for storage (remove large fields)."""
    if not state:
        return {}

    truncated = {**state}
    # Keep URL, title, extracted_values; truncate a11y tree and text
    if "accessibility_tree" in truncated:
        tree = truncated["accessibility_tree"]
        truncated["accessibility_tree"] = tree[:30] if isinstance(tree, list) else []
    if "visible_text" in truncated:
        text = truncated["visible_text"]
        truncated["visible_text"] = text[:500] if isinstance(text, str) else ""
    return truncated
