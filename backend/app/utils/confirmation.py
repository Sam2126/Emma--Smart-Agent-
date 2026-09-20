"""
Human confirmation channel for irreversible actions.

When a browser tool is about to place an order, pay or check out, it asks the
user instead of acting blindly. The agent engine installs a Confirmer for the
running task as a context variable; `asyncio.to_thread` copies context into the
tool's worker thread, so the tool reads it there with `get_confirmer()`.

A Confirmer wraps an async callback supplied by whoever started the task. The
WebSocket server's callback shows the extension's confirmation dialog and waits
for the user's answer. Tasks started without a UI (REST, wake word) have no
Confirmer, and the tools refuse irreversible actions — the same behaviour the
safety gate always had.
"""

from __future__ import annotations

import asyncio
import contextvars
from typing import Any, Awaitable, Callable

import structlog

logger = structlog.get_logger(__name__)

ConfirmFn = Callable[[str, dict[str, Any]], Awaitable[bool]]


class Confirmer:
    def __init__(self, fn: ConfirmFn, timeout_seconds: float = 120.0) -> None:
        self._fn = fn
        self._timeout = timeout_seconds

    async def request(self, action_description: str, details: dict[str, Any] | None = None) -> bool:
        """Ask the user. Anything other than an explicit yes counts as no."""
        try:
            answer = await asyncio.wait_for(self._fn(action_description, details or {}), self._timeout)
            return bool(answer)
        except asyncio.TimeoutError:
            logger.warning("confirmation_timed_out", action=action_description, seconds=self._timeout)
            return False
        except Exception as e:
            logger.warning("confirmation_failed", action=action_description, error=str(e)[:200])
            return False


_current: contextvars.ContextVar[Confirmer | None] = contextvars.ContextVar("agent_confirmer", default=None)


def set_confirmer(confirmer: Confirmer | None) -> contextvars.Token:
    return _current.set(confirmer)


def reset_confirmer(token: contextvars.Token) -> None:
    _current.reset(token)


def get_confirmer() -> Confirmer | None:
    return _current.get()
