"""
WebSocket message protocol — defines all messages between extension and backend.

Uses Pydantic models for validation and serialization. Every message has a
`type` field for routing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Message types
# =============================================================================

class MessageType(str, Enum):
    """All supported WebSocket message types."""
    # Extension / Local UI -> Backend
    TASK_SUBMIT = "task_submit"
    CONFIRMATION_RESPONSE = "confirmation_response"
    TASK_CANCEL = "task_cancel"
    VOICE_TASK = "voice_task"
    TASK_FEEDBACK = "task_feedback"

    # Backend → Extension
    STATUS_UPDATE = "status_update"
    CONFIRMATION_REQUEST = "confirmation_request"
    TASK_COMPLETE = "task_complete"
    VOICE_TRANSCRIPT = "voice_transcript"
    FEEDBACK_RECORDED = "feedback_recorded"
    ERROR = "error"
    CONNECTED = "connected"


# =============================================================================
# Extension → Backend messages
# =============================================================================

class TaskSubmitMessage(BaseModel):
    """User submits a new task via the extension (scope=browser) or the
    Local Agent UI (scope=local)."""
    type: str = MessageType.TASK_SUBMIT.value
    instruction: str
    scope: str = "browser"
    # Came from the conversation microphone (see VoiceTaskMessage).
    conversation: bool = False  # "browser" | "local"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class VoiceTaskMessage(BaseModel):
    """Voice-recorded task: audio (base64) to transcribe into an instruction."""
    type: str = MessageType.VOICE_TASK.value
    audio_base64: str
    mime: str = "audio/webm"
    scope: str = "browser"  # "browser" | "local"
    # True when it came from the conversation microphone rather than the task
    # one. Anything clearly a job is still carried out; the difference is what
    # happens with a sentence that could be either, which in a conversation is
    # answered rather than acted on.
    conversation: bool = False
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ConfirmationResponseMessage(BaseModel):
    """User confirms or rejects an irreversible action."""
    type: str = MessageType.CONFIRMATION_RESPONSE.value
    task_id: str
    confirmed: bool


class TaskCancelMessage(BaseModel):
    """User cancels a running task."""
    type: str = MessageType.TASK_CANCEL.value
    task_id: str


class TaskFeedbackMessage(BaseModel):
    """User rates a finished task: 👍 (+1) or 👎 (-1)."""
    type: str = MessageType.TASK_FEEDBACK.value
    task_id: str
    rating: int
    comment: str = ""

    @field_validator("rating")
    @classmethod
    def _rating_is_up_or_down(cls, value: int) -> int:
        if value not in (1, -1):
            raise ValueError("rating must be 1 (👍) or -1 (👎)")
        return value


# =============================================================================
# Backend → Extension messages
# =============================================================================

class VoiceTranscriptMessage(BaseModel):
    """Transcript of a voice task, echoed back so the UI can show it."""
    type: str = MessageType.VOICE_TRANSCRIPT.value
    text: str
    scope: str = "browser"


class StatusUpdateMessage(BaseModel):
    """Progress update sent to the extension during task execution."""
    type: str = MessageType.STATUS_UPDATE.value
    task_id: str
    status: str  # TaskStatus value
    current_step: str = ""  # Human-readable current step description
    current_step_index: int = 0
    total_steps: int = 0
    progress: float = 0.0  # 0.0 to 1.0
    details: str = ""  # Additional context


class ConfirmationRequestMessage(BaseModel):
    """Ask the user to confirm an irreversible action."""
    type: str = MessageType.CONFIRMATION_REQUEST.value
    task_id: str
    action_description: str
    details: dict[str, Any] = Field(default_factory=dict)
    # Product info for display
    product_name: str = ""
    price: str = ""
    quantity: int = 1


class TaskCompleteMessage(BaseModel):
    """Task has finished (success or failure)."""
    type: str = MessageType.TASK_COMPLETE.value
    task_id: str
    success: bool
    summary: str
    error: str = ""
    duration_seconds: float = 0.0
    # "Why I did this": the steps taken and the past tasks that shaped them.
    explanation: str = ""
    # True when the first attempt failed and the agent retried with a new plan.
    retried: bool = False


class FeedbackRecordedMessage(BaseModel):
    """Acknowledges a 👍 / 👎 so the UI can confirm it was saved."""
    type: str = MessageType.FEEDBACK_RECORDED.value
    task_id: str
    rating: int
    applied: bool
    queued: bool = False
    message: str = ""


class ErrorMessage(BaseModel):
    """Error message from backend."""
    type: str = MessageType.ERROR.value
    message: str
    task_id: str = ""


class ConnectedMessage(BaseModel):
    """Sent when extension first connects."""
    type: str = MessageType.CONNECTED.value
    message: str = "Connected to Self-Improving Agent backend"
    version: str = "0.3.0"


# =============================================================================
# Message parsing
# =============================================================================

def parse_incoming_message(
    data: dict,
) -> TaskSubmitMessage | VoiceTaskMessage | ConfirmationResponseMessage | TaskCancelMessage | TaskFeedbackMessage | None:
    """
    Parse an incoming WebSocket message from the extension.

    Args:
        data: Parsed JSON dict from the WebSocket.

    Returns:
        Typed message object, or None if the type is unknown.
    """
    msg_type = data.get("type")

    match msg_type:
        case MessageType.TASK_SUBMIT.value:
            return TaskSubmitMessage(**data)
        case MessageType.VOICE_TASK.value:
            return VoiceTaskMessage(**data)
        case MessageType.CONFIRMATION_RESPONSE.value:
            return ConfirmationResponseMessage(**data)
        case MessageType.TASK_CANCEL.value:
            return TaskCancelMessage(**data)
        case MessageType.TASK_FEEDBACK.value:
            return TaskFeedbackMessage(**data)
        case _:
            return None
