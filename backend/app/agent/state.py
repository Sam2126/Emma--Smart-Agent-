"""
State schema for the LangGraph agent engine.

One AgentState flows through every node of the graph:

    recall -> plan -> act -> verify -> (replan -> act) -> learn

Nodes return partial updates; LangGraph merges them. Callbacks and other
per-run objects that must not live in the state (the status callback, the tool
set, the deadline) travel in the run context instead — see agent/context.py.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, TypedDict


class TaskStatus(str, Enum):
    """Current status of the task in the agent loop."""
    PLANNING = "planning"
    PERCEIVING = "perceiving"
    ACTING = "acting"
    VERIFYING = "verifying"
    REPLANNING = "replanning"
    LEARNING = "learning"
    WAITING_CONFIRMATION = "waiting_confirmation"
    COMPLETED = "completed"
    FAILED = "failed"


class FailureKind(str, Enum):
    NONE = ""
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    STEP_LIMIT = "step_limit"
    VERIFICATION = "verification"
    INCOMPLETE = "incomplete"


class AgentState(TypedDict, total=False):
    # --- Task identity ---
    task_id: str
    task_instruction: str
    scope: str                     # "browser" (extension) | "local" (Local Agent UI, wake word)
    domain: str

    # --- Recall (pre-flight) ---
    brain_context: str             # SQL memory: domain selectors, skills, failure patterns
    experiences: list[Any]         # semantic_memory.Experience, most similar first
    strategy: str                  # STRATEGY / AVOID note from similar experiences
    strategy_from_llm: bool
    strategy_partial: bool
    initial_page_state: dict[str, Any]

    # --- Planning / acting ---
    plan: str
    attempt: int                   # 1, or 2 after an automatic retry
    retry_context: str
    trajectory: list[dict[str, Any]]
    final_report: str

    # --- Outcome ---
    success: bool
    failure_kind: str              # FailureKind value
    error: str
    verification_summary: str
    retry_blocked_reason: str
    retried: bool
    first_attempt_error: str
    first_attempt_trajectory: list[dict[str, Any]]  # compared with the retry when learning

    # --- Output ---
    status: str                    # TaskStatus value
    explanation: str
    duration_seconds: float


def create_initial_state(
    task_instruction: str,
    task_id: str | None = None,
    scope: str = "browser",
) -> AgentState:
    """Initial state for a new task.

    Args:
        task_instruction: The user instruction (typed, or a voice transcript).
        task_id: Optional task id; generated when omitted.
        scope: "browser" for extension tasks (web only) or "local" for the
            Local Agent UI and wake word (this computer plus the browser).
    """
    return AgentState(
        task_id=task_id or str(uuid.uuid4()),
        task_instruction=task_instruction,
        scope=scope,
        domain="",
        brain_context="",
        experiences=[],
        strategy="",
        strategy_from_llm=False,
        strategy_partial=False,
        initial_page_state={},
        plan="",
        attempt=1,
        retry_context="",
        trajectory=[],
        final_report="",
        success=False,
        failure_kind=FailureKind.NONE.value,
        error="",
        verification_summary="",
        retry_blocked_reason="",
        retried=False,
        first_attempt_error="",
        first_attempt_trajectory=[],
        status=TaskStatus.PLANNING.value,
        explanation="",
        duration_seconds=0.0,
    )
