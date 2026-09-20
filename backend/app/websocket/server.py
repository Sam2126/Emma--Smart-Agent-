"""
WebSocket server — handles communication between the Chrome extension, the
Local Agent UI and the backend.

Runs alongside the FastAPI server on a separate port. Manages:
- task submissions (typed and voice) -> the LangGraph agent engine
- streaming progress: every phase and every tool call as a status update
- human confirmation before irreversible actions (checkout, payment)
- 👍 / 👎 feedback on finished tasks, which weights future recall
- task cancellation
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import structlog
import websockets
from websockets.asyncio.server import ServerConnection

from app.agent.state import TaskStatus
from app.websocket.protocol import (
    ConfirmationRequestMessage,
    ConfirmationResponseMessage,
    ConnectedMessage,
    ErrorMessage,
    FeedbackRecordedMessage,
    StatusUpdateMessage,
    TaskCancelMessage,
    TaskCompleteMessage,
    TaskFeedbackMessage,
    TaskSubmitMessage,
    VoiceTaskMessage,
    VoiceTranscriptMessage,
    parse_incoming_message,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Scope resolution helpers
# ---------------------------------------------------------------------------

# Phrases that force "local" scope regardless of the caller's default scope
_LOCAL_FORCE_PHRASES = (
    "on this device",
    "on my device",
    "local device",
    "use this device",
    "this computer",
    "my computer",
    "locally",
    "on this machine",
    "on my machine",
    "on this pc",
    "on my pc",
)

# Phrases/patterns that force "browser" / Chrome scope
_BROWSER_FORCE_PHRASES = (
    "in chrome",
    "use chrome",
    "open chrome",
    "in the browser",
    "use the browser",
    "on the browser",
    "in browser",
    "open browser",
    "search online",
    "search the web",
    "google it",
    "go to website",
    "go to the website",
    "open website",
    "visit website",
    "open url",
    "go to url",
    "browse to",
    "navigate to",
    "http://",
    "https://",
    ".com",
    ".org",
    ".net",
    ".io",
)

# Files and folders that live on this laptop. A task naming one keeps the local
# tools even when it also says "open chrome", because the local scope has the
# browser tools as well and can do both halves in a single run. Found
# 2026-09-16: "open chrome search gemini and add the last ppt from downloads and
# ask him to create a summary" matched "open chrome" first, so it ran in browser
# scope, whose 12 tools cannot look in Downloads. The agent reported — correctly
# for the tools it was given — that it had no way to find the file, and did
# nothing. Routing one of these to local is safe: local is the wider scope, and
# rule 5 below already sends everything unrecognised there.
_LOCAL_RESOURCE_PHRASES = (
    "downloads",
    "download folder",
    "my files",
    "local file",
    "local folder",
    "file explorer",
    "desktop",
    "documents",
    "attach",
    "upload",
    "from my computer",
    "from my laptop",
    "from my pc",
    "ppt",
    "pdf",
    "docx",
    "xlsx",
    "csv",
    "resume",
    "screenshot",
    ".txt",
    ".zip",
    "last file",
    "latest file",
    "recent file",
    # Email is done with the email tools (local scope). An address such as
    # rakesh@gmail.com also contains ".com", which would otherwise force the
    # browser scope, where there is no email tool. "mail" also covers email,
    # e-mail, gmail and mailbox.
    "mail",
    "inbox",
)

# Task categories that are clearly local (file system, apps, system)
_LOCAL_TASK_PHRASES = (
    "open notepad",
    "open calculator",
    "open file",
    "open folder",
    "create file",
    "create folder",
    "delete file",
    "delete folder",
    "rename file",
    "rename folder",
    "move file",
    "copy file",
    "list files",
    "show files",
    "show folders",
    "in notepad",
    "in calculator",
    "in file explorer",
    "file explorer",
    "task manager",
    "control panel",
    "settings app",
    "write to file",
    "read the file",
    "read file",
    "take a screenshot",
    "clipboard",
    "type on keyboard",
    "press key",
    "run command",
    "open terminal",
    "open cmd",
    "open powershell",
    "open vs code",
    "open vscode",
    "open excel",
    "open word",
    "open powerpoint",
    "play music",
    "play video",
    "set volume",
    "adjust volume",
    "set brightness",
    "wifi",
    "bluetooth",
    "install app",
    "uninstall app",
)


def _resolve_voice_scope(transcript: str, caller_scope: str) -> str:
    """
    Determine the best scope ('local' or 'browser') for a voice command.

    Priority rules (applied in order):
      1. If the transcript contains an **explicit local-device phrase**, return 'local'.
      2. If it names a file or folder on this laptop, return 'local' — even when it
         also names the browser, because only the local scope can do both halves.
      3. If the transcript contains an **explicit browser/Chrome phrase**, return 'browser'.
      4. If caller_scope is already 'local' (i.e. request came from Local Agent UI), return 'local'.
      5. If the instruction looks like a clearly local OS/app task, return 'local'.
      6. Otherwise default to 'local' — try local first, let the agent escalate if needed.
         (The local scope has browser tools too.)
    """
    lower = transcript.lower()

    for phrase in _LOCAL_FORCE_PHRASES:
        if phrase in lower:
            return "local"

    for phrase in _LOCAL_RESOURCE_PHRASES:
        if phrase in lower:
            return "local"

    for phrase in _BROWSER_FORCE_PHRASES:
        if phrase in lower:
            return "browser"

    if caller_scope == "local":
        return "local"

    for phrase in _LOCAL_TASK_PHRASES:
        if phrase in lower:
            return "local"

    return "local"


async def _ensure_browser_quietly() -> None:
    """Bring up the browser lazily; failures are logged, never fatal."""
    try:
        from app.main import ensure_browser_connection

        await ensure_browser_connection()
    except Exception as e:
        logger.warning("lazy_browser_bringup_failed", error=str(e)[:150])


class WebSocketServer:
    """
    WebSocket server for extension / Local Agent UI <-> backend communication.

    Manages connected clients and routes messages to the agent engine.
    """

    def __init__(self) -> None:
        self._clients: set[ServerConnection] = set()
        self._running_tasks: dict[str, asyncio.Task] = {}
        # task_id -> future resolved by the user's confirmation response
        self._pending_confirmations: dict[str, asyncio.Future] = {}
        # Progress messages go out as tasks of their own. The event loop keeps
        # only weak references to tasks, so these are held until they finish.
        self._send_tasks: set[asyncio.Task] = set()

    async def start(self, host: str, port: int) -> None:
        """Start the WebSocket server."""
        logger.info("websocket_server_starting", host=host, port=port)

        async with websockets.serve(
            self._handle_connection,
            host,
            port,
            ping_interval=20,
            ping_timeout=20,
            max_size=32 * 1024 * 1024,  # voice recordings arrive as base64
        ):
            logger.info("websocket_server_started", host=host, port=port)
            await asyncio.Future()  # Run forever

    async def _handle_connection(self, websocket: ServerConnection) -> None:
        """Handle a new WebSocket connection."""
        self._clients.add(websocket)
        client_id = id(websocket)
        logger.info("client_connected", client_id=client_id)
        # The browser is NOT brought up just because a client connected.
        # Chrome only starts when a browser-scoped task actually needs it.

        await self._send(websocket, ConnectedMessage())

        try:
            async for raw_message in websocket:
                try:
                    data = json.loads(raw_message)
                    await self._handle_message(websocket, data)
                except json.JSONDecodeError:
                    logger.warning("invalid_json", message=str(raw_message)[:100])
                    await self._send(websocket, ErrorMessage(message="Invalid JSON"))
                except Exception as e:
                    logger.error("message_handling_error", error=str(e))
                    await self._send(websocket, ErrorMessage(message=str(e)))
        except websockets.exceptions.ConnectionClosed:
            logger.info("client_disconnected", client_id=client_id)
        finally:
            self._clients.discard(websocket)

    async def _handle_message(self, websocket: ServerConnection, data: dict) -> None:
        """Route an incoming message to the appropriate handler."""
        message = parse_incoming_message(data)

        if message is None:
            logger.warning("unknown_message_type", type=data.get("type"))
            await self._send(websocket, ErrorMessage(message=f"Unknown message type: {data.get('type')}"))
            return

        match message:
            case TaskSubmitMessage():
                await self._handle_task_submit(websocket, message)
            case VoiceTaskMessage():
                await self._handle_voice_task(websocket, message)
            case ConfirmationResponseMessage():
                await self._handle_confirmation(websocket, message)
            case TaskCancelMessage():
                await self._handle_cancel(websocket, message)
            case TaskFeedbackMessage():
                await self._handle_feedback(websocket, message)

    async def _handle_task_submit(self, websocket: ServerConnection, message: TaskSubmitMessage) -> None:
        """Start a task on the agent engine."""
        instruction = message.instruction.strip()
        if not instruction:
            await self._send(websocket, ErrorMessage(message="Empty instruction"))
            return

        logger.info("task_submitted", instruction=instruction)

        # Local Agent always sends scope="local"; the extension sends "browser".
        # Extension text tasks get the same smart routing as voice tasks.
        raw_scope = getattr(message, "scope", "browser") or "browser"
        scope = _resolve_voice_scope(instruction, raw_scope)
        if scope != raw_scope:
            logger.info("task_scope_overridden", raw=raw_scope, resolved=scope, instruction=instruction[:80])

        task_id = str(uuid.uuid4())
        await self._send(websocket, StatusUpdateMessage(
            task_id=task_id,
            status=TaskStatus.PLANNING.value,
            current_step="Planning task...",
            progress=0.0,
        ))

        task = asyncio.create_task(
            self._run_agent(websocket, task_id, instruction, scope, getattr(message, "conversation", False))
        )
        self._running_tasks[task_id] = task

    async def _handle_voice_task(self, websocket: ServerConnection, message: VoiceTaskMessage) -> None:
        """Decode voice audio, transcribe via Groq Whisper, then run as a task."""
        import base64

        from app.config import get_settings
        from app.speech_vocab import clean_instruction, whisper_prompt
        from app.voice import guess_extension, transcribe_audio

        try:
            audio = base64.b64decode(message.audio_base64)
        except Exception as e:
            await self._send(websocket, ErrorMessage(message=f"Bad audio payload: {e}"))
            return

        logger.info("voice_task_received", bytes=len(audio), scope=message.scope)
        try:
            # Without a language the model guesses, and it guesses badly on a
            # short phrase: a real run turned spoken English into Korean
            # ("hello what's up" came back as "바로 뭐죠"). The vocabulary is
            # the same one the wake word listener uses, so "Myntra" survives
            # here too.
            transcript = await transcribe_audio(
                audio,
                filename=guess_extension(message.mime),
                language="en",
                prompt=whisper_prompt(get_settings().speech_extra_names),
            )
            transcript = clean_instruction(
                transcript, extra_names=get_settings().speech_extra_names
            ) or transcript
        except Exception as e:
            await self._send(websocket, ErrorMessage(message=f"Voice transcription failed: {e}"))
            return

        if not transcript:
            await self._send(websocket, ErrorMessage(message="Voice heard, but no speech recognized."))
            return

        logger.info("voice_transcribed", instruction=transcript)
        from app.voice_feedback import parse_voice_feedback, record_voice_feedback

        feedback = parse_voice_feedback(transcript)
        if feedback is not None:
            # A spoken "good job" / "wrong, ..." rates the last task instead of running one.
            await self._send(websocket, VoiceTranscriptMessage(text=transcript, scope=message.scope))
            result = await record_voice_feedback(feedback)
            logger.info("voice_feedback", task_id=result.get("task_id", ""), rating=feedback.rating, via="mic")
            if result.get("task_id"):
                await self.broadcast(FeedbackRecordedMessage(
                    task_id=result["task_id"],
                    rating=feedback.rating,
                    applied=bool(result.get("applied")),
                    queued=bool(result.get("queued")),
                    message=result.get("message", ""),
                ))
            else:
                await self._send(websocket, ErrorMessage(message=result.get("message", "Feedback could not be saved.")))
            return

        resolved_scope = _resolve_voice_scope(transcript, message.scope)
        logger.info("voice_scope_resolved", raw_scope=message.scope, resolved=resolved_scope, transcript=transcript[:80])

        await self._send(websocket, VoiceTranscriptMessage(text=transcript, scope=resolved_scope))
        await self._handle_task_submit(websocket, TaskSubmitMessage(
            instruction=transcript, scope=resolved_scope, conversation=message.conversation,
        ))

    async def _run_agent(self, websocket: ServerConnection, task_id: str, instruction: str, scope: str,
                         conversation: bool = False) -> None:
        """Run one task on the agent engine and stream its progress."""
        from app.agent.runner import AgentRunner

        # Browser tasks pre-ensure the browser. Local tasks must not wake any
        # browser: a browser tool used later brings Chrome up lazily itself.
        if scope == "browser":
            await _ensure_browser_quietly()

        def _on_status(update: dict[str, Any]) -> None:
            self._send_soon(websocket, StatusUpdateMessage(
                task_id=task_id,
                status=update.get("status", "acting"),
                current_step=update.get("current_step", "Processing..."),
                progress=update.get("progress", 0.5),
                details=update.get("details", ""),
            ))

        async def _confirm(action_description: str, details: dict[str, Any]) -> bool:
            return await self._request_confirmation(websocket, task_id, action_description, details)

        runner = AgentRunner(status_callback=_on_status, confirm_callback=_confirm)
        try:
            result = await runner.run_task(instruction=instruction, task_id=task_id, scope=scope,
                                           conversation=conversation)
            success = result.get("success", False)
            await self._send(websocket, TaskCompleteMessage(
                task_id=task_id,
                success=success,
                summary=result.get("summary") or ("Task completed" if success else "Task failed"),
                error=result.get("error", ""),
                duration_seconds=result.get("duration_seconds", 0.0),
                explanation=result.get("explanation", ""),
                retried=bool(result.get("retried")),
            ))
            logger.info("task_finished", task_id=task_id, success=success, duration=f"{result.get('duration_seconds', 0.0):.1f}s")
        except asyncio.CancelledError:
            logger.info("task_run_cancelled", task_id=task_id)
            raise
        except Exception as e:
            logger.error("agent_execution_error", task_id=task_id, error=str(e))
            await self._send(websocket, TaskCompleteMessage(
                task_id=task_id,
                success=False,
                summary="Task failed during agent execution",
                error=str(e),
            ))
        finally:
            self._running_tasks.pop(task_id, None)
            pending = self._pending_confirmations.pop(task_id, None)
            if pending and not pending.done():
                pending.set_result(False)

    async def _request_confirmation(
        self,
        websocket: ServerConnection,
        task_id: str,
        action_description: str,
        details: dict[str, Any],
    ) -> bool:
        """Show the confirmation dialog and wait for the user's answer."""
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_confirmations[task_id] = future
        logger.info("confirmation_requested", task_id=task_id, action=action_description)
        await self._send(websocket, StatusUpdateMessage(
            task_id=task_id,
            status=TaskStatus.WAITING_CONFIRMATION.value,
            current_step="⚠️ Waiting for your confirmation",
            progress=0.6,
            details=action_description,
        ))
        await self._send(websocket, ConfirmationRequestMessage(
            task_id=task_id,
            action_description=f"The agent wants to: {action_description}. This cannot be undone.",
            details={k: str(v) for k, v in (details or {}).items()},
        ))
        try:
            return bool(await future)
        finally:
            self._pending_confirmations.pop(task_id, None)

    async def _handle_confirmation(self, websocket: ServerConnection, message: ConfirmationResponseMessage) -> None:
        """Deliver the user's answer to the task that is waiting for it."""
        logger.info("confirmation_received", task_id=message.task_id, confirmed=message.confirmed)
        future = self._pending_confirmations.get(message.task_id)
        if future is None or future.done():
            await self._send(websocket, ErrorMessage(
                message="There is no action waiting for confirmation for this task.",
                task_id=message.task_id,
            ))
            return
        future.set_result(bool(message.confirmed))

    async def _handle_cancel(self, websocket: ServerConnection, message: TaskCancelMessage) -> None:
        """Handle task cancellation request."""
        task_id = message.task_id
        pending = self._pending_confirmations.pop(task_id, None)
        if pending and not pending.done():
            pending.set_result(False)

        task = self._running_tasks.get(task_id)
        if task:
            task.cancel()
            self._running_tasks.pop(task_id, None)
            await self._send(websocket, TaskCompleteMessage(task_id=task_id, success=False, summary="Task cancelled by user"))
            logger.info("task_cancelled", task_id=task_id)
        else:
            await self._send(websocket, ErrorMessage(message=f"No running task with ID: {task_id}", task_id=task_id))

    async def _handle_feedback(self, websocket: ServerConnection, message: TaskFeedbackMessage) -> None:
        """Store the user's 👍 / 👎 on a finished task."""
        from app.agent.services import brain

        result = await brain.record_feedback(message.task_id, message.rating, message.comment)
        logger.info("task_feedback", task_id=message.task_id, rating=message.rating, applied=result.get("applied"))
        await self._send(websocket, FeedbackRecordedMessage(
            task_id=message.task_id,
            rating=message.rating,
            applied=bool(result.get("applied")),
            queued=bool(result.get("queued")),
            message=result.get("message", ""),
        ))

    def _send_soon(self, websocket: ServerConnection, message: Any) -> None:
        """Send without waiting, keeping the send task alive until it has run."""
        task = asyncio.get_running_loop().create_task(self._send(websocket, message))
        self._send_tasks.add(task)
        task.add_done_callback(self._send_tasks.discard)

    async def _send(self, websocket: ServerConnection, message: Any) -> None:
        """Send a message to a connected client."""
        try:
            await websocket.send(json.dumps(message.model_dump()))
        except websockets.exceptions.ConnectionClosed:
            logger.warning("send_failed_connection_closed")
        except Exception as e:
            logger.error("send_failed", error=str(e))

    async def broadcast(self, message: Any) -> None:
        """Broadcast a message to all connected clients."""
        for client in self._clients.copy():
            await self._send(client, message)
