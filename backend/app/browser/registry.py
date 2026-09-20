"""
Process-wide registry for the shared BrowserController.

main.py registers the controller at startup; browser tools and the agent
engine look it up here. (It used to live in the old LangGraph perceiver node.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.browser.controller import BrowserController

_browser_controller: "BrowserController | None" = None


def set_browser_controller(controller: "BrowserController | None") -> None:
    global _browser_controller
    _browser_controller = controller


def get_browser_controller() -> "BrowserController":
    if _browser_controller is None:
        raise RuntimeError("Browser controller not initialized. Call set_browser_controller first.")
    return _browser_controller


def has_browser_controller() -> bool:
    return _browser_controller is not None
