"""
Registry for the application's main asyncio event loop.

Browser tools must run on the loop that owns the Playwright connection, while
the agent executes tools in worker threads. Tools find that loop here instead
of importing app.main, which would pull the whole web app into every tool
import (and into tests).
"""

from __future__ import annotations

import asyncio

_main_loop: asyncio.AbstractEventLoop | None = None


def set_main_loop(loop: asyncio.AbstractEventLoop | None) -> None:
    global _main_loop
    _main_loop = loop


def get_main_loop() -> asyncio.AbstractEventLoop | None:
    return _main_loop
