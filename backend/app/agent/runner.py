"""
AgentRunner — runs one task through the LangGraph engine.

Replaces CrewRunner. Every entry point uses it: the WebSocket server
(extension and Local Agent UI, typed and voice), the wake-word listener, the
REST /task endpoint and the eval runner.

    runner = AgentRunner(status_callback=..., confirm_callback=...)
    result = await runner.run_task(instruction, task_id, scope)

status_callback receives {"status", "current_step", "progress", "details"} for
every phase and every tool call (streaming progress). confirm_callback, when
given, is awaited as confirm(action_description, details) -> bool before any
irreversible browser action; without it such actions are refused.
"""

from __future__ import annotations

import asyncio
import time
import weakref
from typing import Any, Awaitable, Callable

import structlog

from app.agent.context import RunContext, StatusCallback, track_run, untrack_run
from app.agent.graph import compile_agent_graph
from app.agent.services import brain, episodic_logger
from app.agent.state import create_initial_state
from app.config import get_settings
from app.utils.confirmation import Confirmer, reset_confirmer, set_confirmer

logger = structlog.get_logger(__name__)

ConfirmCallback = Callable[[str, dict[str, Any]], Awaitable[bool]]

_graph = None

# One task at a time. Tasks drive the same browser tab and the same keyboard and
# mouse, so two at once corrupt each other. Found 2026-09-15: a typed task
# started while a wake-word task was still verifying, and both drove the same
# page. A new task waits (with a status line) for the running one to finish.
# One lock per event loop, so each test's loop gets its own.
_task_lanes: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = weakref.WeakKeyDictionary()


def _task_lane() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lane = _task_lanes.get(loop)
    if lane is None:
        lane = _task_lanes[loop] = asyncio.Lock()
    return lane


# The most recently finished task, whichever window or the wake word started
# it: a spoken rating ("Emma, good job, done") applies to it.
_last_finished: dict[str, Any] | None = None


def last_finished_task() -> dict[str, Any] | None:
    return dict(_last_finished) if _last_finished else None


def _remember_finished(task_id: str, instruction: str, success: bool) -> None:
    global _last_finished
    _last_finished = {
        "task_id": task_id,
        "instruction": instruction,
        "success": success,
        "finished_at": time.time(),
    }


def get_graph():
    global _graph
    if _graph is None:
        _graph = compile_agent_graph()
    return _graph


class AgentRunner:
    def __init__(
        self,
        status_callback: StatusCallback | None = None,
        confirm_callback: ConfirmCallback | None = None,
    ) -> None:
        self.status_callback = status_callback
        self.confirm_callback = confirm_callback
        # What the last task did, so "what did you just do?" has an answer.
        self._last_summary = ""

    async def run_task(self, instruction: str, task_id: str, scope: str = "browser",
                       conversation: bool = False) -> dict[str, Any]:
        current = track_run()
        try:
            lane = _task_lane()
            if lane.locked():
                logger.info("task_waiting_for_previous_task", task_id=task_id)
                if self.status_callback:
                    try:
                        self.status_callback({
                            "status": "planning",
                            "current_step": "⏳ Waiting for the current task to finish…",
                            "progress": 0.0,
                            "details": "",
                        })
                    except Exception:
                        pass
            async with lane:
                return await self._run_task(instruction, task_id, scope, conversation)
        finally:
            untrack_run(current)

    async def _answer_if_question(
        self, instruction: str, task_id: str, run: Any, conversation: bool = False
    ) -> dict[str, Any] | None:
        """Reply out loud when the user asked something; None when work is wanted."""
        from app.conversation import QUESTION, answer, route

        if await route(instruction, conversation=conversation) != QUESTION:
            return None

        reply = await answer(instruction, recent=self._last_summary)
        logger.info("question_answered", task_id=task_id, chars=len(reply))
        try:
            from app.tts import Priority, say

            say(reply, priority=Priority.ANSWER)
        except Exception as e:
            logger.debug("tts_say_failed", error=str(e)[:120])

        if self.status_callback:
            try:
                self.status_callback({
                    "status": "answered", "current_step": reply, "progress": 1.0, "details": "",
                })
            except Exception:
                pass
        return {
            "task_id": task_id,
            "success": True,
            "summary": reply,
            "error": "",
            "explanation": "You asked a question, so I answered it instead of doing a task.",
            "duration_seconds": run.elapsed(),
            "steps": 0,
            "retried": False,
            "answered_question": True,
        }

    async def _run_task(self, instruction: str, task_id: str, scope: str,
                        conversation: bool = False) -> dict[str, Any]:
        settings = get_settings()
        run = RunContext(
            status_callback=self.status_callback,
            deadline=time.monotonic() + settings.task_timeout_seconds,
        )
        logger.info("agent_task_started", task_id=task_id, scope=scope, instruction=instruction)

        # A question is answered, not performed. Asked "what did you just do?",
        # the agent used to plan, open a browser and start working.
        answered = await self._answer_if_question(instruction, task_id, run, conversation)
        if answered is not None:
            return answered

        try:
            await episodic_logger.log_task_start(task_id=task_id, instruction=instruction)
        except Exception as e:
            logger.warning("episodic_log_start_failed", error=str(e)[:200])

        confirmer = (
            Confirmer(self.confirm_callback, settings.human_confirmation_timeout_seconds)
            if self.confirm_callback else None
        )
        token = set_confirmer(confirmer)
        try:
            final = await get_graph().ainvoke(
                create_initial_state(instruction, task_id=task_id, scope=scope),
                config={"configurable": {"run": run}, "recursion_limit": 30},
            )
        except asyncio.CancelledError:
            try:
                await episodic_logger.finalize_task(
                    task_id=task_id, success=False, error="Cancelled by user", duration_seconds=run.elapsed()
                )
            except Exception:
                pass
            raise
        except Exception as e:
            return await self._crashed(task_id, instruction, scope, run, e)
        finally:
            reset_confirmer(token)

        success = bool(final.get("success"))
        report = final.get("final_report") or ""
        error = final.get("error") or ""
        result = {
            "task_id": task_id,
            "success": success,
            "summary": report or (error if not success else "Task completed"),
            "error": "" if success else error,
            "duration_seconds": final.get("duration_seconds") or run.elapsed(),
            "domain": final.get("domain", ""),
            "explanation": final.get("explanation", ""),
            "retried": bool(final.get("retried")),
            "trajectory": final.get("trajectory", []),
        }
        # Kept so a spoken "what did you just do?" has something true to say.
        outcome = "worked" if success else "failed"
        self._last_summary = f"Task: {instruction}\nOutcome: {outcome}\n{report or error}"[:900]
        _remember_finished(task_id, instruction, success)
        logger.info(
            "agent_task_finished",
            task_id=task_id,
            success=success,
            retried=result["retried"],
            steps=len(result["trajectory"]),
            duration=f"{result['duration_seconds']:.1f}s",
        )
        return result

    async def _crashed(self, task_id: str, instruction: str, scope: str, run: RunContext, exc: Exception) -> dict[str, Any]:
        """The graph itself raised. Record it like any failed task."""
        logger.error("agent_task_crashed", task_id=task_id, error=str(exc)[:500])
        duration = run.elapsed()
        domain = "local" if scope == "local" else brain._extract_domain_from_instruction(instruction)
        try:
            await episodic_logger.finalize_task(task_id=task_id, success=False, error=str(exc), duration_seconds=duration)
        except Exception:
            pass
        try:
            await brain.learn_from_task(
                task_id=task_id, instruction=instruction, domain=domain, trajectory=[],
                success=False, error=str(exc), duration_seconds=duration, raw_result="",
            )
        except Exception:
            pass
        run.notify("failed", f"Failed: {exc}", 1.0)
        _remember_finished(task_id, instruction, False)
        return {
            "task_id": task_id,
            "success": False,
            "summary": f"Task failed: {exc}",
            "error": str(exc),
            "duration_seconds": duration,
            "domain": domain,
            "explanation": "",
            "retried": False,
            "trajectory": [],
        }
