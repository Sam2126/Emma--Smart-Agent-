"""
Atomic browser action executors.

Each function performs a single, discrete browser action (click, type, navigate,
etc.) and returns an ActionResult describing what happened. Actions include
random delays to mimic human behavior and avoid bot detection.
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import structlog
from playwright.async_api import Page, TimeoutError as PlaywrightTimeout

from app.config import get_settings

logger = structlog.get_logger(__name__)


class ActionType(str, Enum):
    """All supported atomic browser actions."""
    CLICK = "click"
    TYPE = "type"
    NAVIGATE = "navigate"
    SCROLL = "scroll"
    WAIT_FOR = "wait_for"
    PRESS_KEY = "press_key"
    SELECT_OPTION = "select_option"
    GET_TEXT = "get_text"
    HOVER = "hover"
    GO_BACK = "go_back"


@dataclass
class ActionResult:
    """Result of executing a single browser action."""
    action_type: ActionType
    success: bool
    selector: str = ""
    input_value: str = ""
    error: str | None = None
    # The URL after the action executed
    url_after: str = ""
    # Any extracted text (for GET_TEXT actions)
    extracted_text: str = ""
    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)


async def _human_delay() -> None:
    """Insert a random delay to mimic human interaction speed."""
    settings = get_settings()
    delay = random.uniform(settings.action_delay_min, settings.action_delay_max)
    await asyncio.sleep(delay)


async def execute_action(
    page: Page,
    action_type: ActionType,
    selector: str = "",
    input_value: str = "",
    timeout: int = 10000,
) -> ActionResult:
    """
    Execute a single browser action on the given page.

    Routes to the appropriate action handler based on action_type.
    """
    logger.info(
        "executing_action",
        action=action_type.value,
        selector=selector,
        input_value=input_value[:50] if input_value else "",
    )

    # Add human-like delay before each action
    await _human_delay()

    try:
        match action_type:
            case ActionType.CLICK:
                return await _click(page, selector, timeout)
            case ActionType.TYPE:
                return await _type_text(page, selector, input_value, timeout)
            case ActionType.NAVIGATE:
                return await _navigate(page, input_value, timeout)
            case ActionType.SCROLL:
                return await _scroll(page, input_value)
            case ActionType.WAIT_FOR:
                return await _wait_for(page, selector, timeout)
            case ActionType.PRESS_KEY:
                return await _press_key(page, input_value)
            case ActionType.SELECT_OPTION:
                return await _select_option(page, selector, input_value, timeout)
            case ActionType.GET_TEXT:
                return await _get_text(page, selector, timeout)
            case ActionType.HOVER:
                return await _hover(page, selector, timeout)
            case ActionType.GO_BACK:
                return await _go_back(page)
            case _:
                return ActionResult(
                    action_type=action_type,
                    success=False,
                    error=f"Unknown action type: {action_type}",
                )
    except PlaywrightTimeout as e:
        logger.warning("action_timeout", action=action_type.value, selector=selector)
        return ActionResult(
            action_type=action_type,
            success=False,
            selector=selector,
            input_value=input_value,
            error=f"Timeout after {timeout}ms: {str(e)}",
            url_after=page.url,
        )
    except Exception as e:
        logger.error("action_failed", action=action_type.value, error=str(e))
        return ActionResult(
            action_type=action_type,
            success=False,
            selector=selector,
            input_value=input_value,
            error=str(e),
            url_after=page.url,
        )


# =============================================================================
# Individual action implementations
# =============================================================================

# Time limits for finding a click target. Found 2026-09-15: Playwright's
# Locator.count() has no timeout, and the fallback strategies called it on every
# frame of the page. On chatgpt.com (several iframes) one click spent 121 s
# looking for an element and another 37 s, while the task stalled.
_CLICK_LOCATE_BUDGET_S = 6.0
_COUNT_TIMEOUT_S = 1.5
_MAX_FRAMES_SEARCHED = 5
_CSS_CHARS = set("#.[]>:=*~()'\"")
_GAME_SELECTORS = ("#game-iframe", "iframe", "canvas", "#gamePageMainContainer", "#game-overlay")


def _text_hint(selector: str) -> str:
    """The visible text a selector refers to, or "" for a pure CSS selector.

    'a:has-text("ChatGPT")' -> 'ChatGPT', 'text=Sign in' -> 'Sign in',
    'Add to cart' -> 'Add to cart', 'a[href*="chatgpt.com"]' -> ''. The text
    strategies used to run on CSS selectors as well, searching the page for the
    literal string 'a[href*="chatgpt.com"]'.
    """
    s = (selector or "").strip()
    match = re.search(r""":has-text\((["'])(.+?)\1\)""", s)
    if match:
        return match.group(2).strip()
    if s.startswith("text="):
        return s[5:].strip().strip("'\"")
    if s and not any(ch in _CSS_CHARS for ch in s):
        return s
    return ""


async def _count(locator, timeout: float = _COUNT_TIMEOUT_S) -> int:
    """Locator.count() with a timeout; 0 when it errors or takes too long."""
    try:
        return await asyncio.wait_for(locator.count(), timeout)
    except Exception:
        return 0


async def _first_present(factories: list[Callable[[], Any]], deadline: float):
    """The first candidate that matches, preferring a visible match, within the deadline."""
    hidden_match = None
    for make in factories:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            candidate = make()
        except Exception:
            continue
        try:
            visible = candidate.filter(visible=True).first
            if await _count(visible, min(_COUNT_TIMEOUT_S, remaining)) > 0:
                return visible
        except Exception:
            pass
        if hidden_match is None and await _count(candidate.first, min(_COUNT_TIMEOUT_S, max(remaining, 0.1))) > 0:
            hidden_match = candidate.first
    return hidden_match


async def _click(page: Page, selector: str, timeout: int) -> ActionResult:
    """
    Click an element, finding it quickly or failing fast.

    Flow (all lookups share a 6 s budget, each check at most 1.5 s):
      1. The selector as given (CSS / Playwright syntax)
      2. For text selectors only: text match, role (button, link, tab), href slug,
         image alt text, aria-label
      3. Canvas / iframe / game containers for play-style targets
      4. Up to 5 child frames (embedded apps, games)
    Visible matches win over hidden ones. Then: normal click -> force click ->
    JS dispatch, each with a short timeout.
    """
    t_start = time.monotonic()
    deadline = t_start + _CLICK_LOCATE_BUDGET_S
    text = _text_hint(selector)

    main: list[Callable[[], Any]] = [lambda: page.locator(selector)]
    if text:
        main.append(lambda: page.get_by_text(text, exact=False))
        for role in ("button", "link", "tab"):
            main.append(lambda role=role: page.get_by_role(role, name=text))
        if len(text) >= 3:
            slug = text.lower().replace(" ", "-").replace(":", "").replace("'", "")
            for css in (
                f"a[href*='{slug}']",
                f"img[alt*='{text}' i]",
                f"[aria-label*='{text}' i]",
                f"a:has(img[alt*='{text}' i])",
            ):
                main.append(lambda css=css: page.locator(css))
    wants_game = any(kw in selector.lower() for kw in ("canvas", "screen", "iframe", "game", "container")) or any(
        kw in text.lower() for kw in ("start", "play", "game", "continue")
    )
    if wants_game:
        for css in _GAME_SELECTORS:
            main.append(lambda css=css: page.locator(css))

    locator = await _first_present(main, deadline)

    if locator is None and time.monotonic() < deadline:
        frames: list[Callable[[], Any]] = []
        try:
            child_frames = [f for f in page.frames if f != page.main_frame and not f.is_detached()]
        except Exception:
            child_frames = []
        for frame in child_frames[:_MAX_FRAMES_SEARCHED]:
            if text:
                frames.append(lambda frame=frame: frame.get_by_text(text, exact=False))
            frames.append(lambda frame=frame: frame.locator(selector))
        locator = await _first_present(frames, deadline)

    if locator is None:
        elapsed = time.monotonic() - t_start
        logger.debug(
            "element_not_found",
            selector=selector,
            elapsed_ms=f"{elapsed * 1000:.0f}ms",
        )
        return ActionResult(
            action_type=ActionType.CLICK,
            success=False,
            selector=selector,
            error=f"Element '{selector}' not found on page — skipped.",
            url_after=page.url,
        )

    # ── Attempt click (visible → force → JS dispatch) ──
    try:
        await locator.wait_for(state="visible", timeout=min(timeout, 3000))
        await locator.scroll_into_view_if_needed(timeout=2000)
        await locator.click(timeout=min(timeout, 3000))
    except Exception as primary_err:
        logger.debug(
            "primary_click_failed_attempting_fallbacks",
            selector=selector,
            error=str(primary_err)[:80],
        )
        # Force click (bypasses subtle overlay interceptions)
        try:
            await locator.click(force=True, timeout=1500)
        except Exception:
            # JS dispatch click on closest clickable ancestor. Without a timeout
            # this waited up to 30 s for the element.
            try:
                await locator.evaluate(
                    "el => (el.closest('button, a, label, input, [role=\"button\"], [role=\"checkbox\"], div') || el).click()",
                    timeout=1500,
                )
            except Exception:
                return ActionResult(
                    action_type=ActionType.CLICK,
                    success=False,
                    selector=selector,
                    error=f"Click failed on '{selector}': {str(primary_err)[:120]}",
                    url_after=page.url,
                )

    # Wait briefly for any navigation or DOM update triggered by the click
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=2000)
    except Exception:
        pass

    elapsed = time.monotonic() - t_start
    logger.debug("click_succeeded", selector=selector, elapsed_ms=f"{elapsed * 1000:.0f}ms")

    return ActionResult(
        action_type=ActionType.CLICK,
        success=True,
        selector=selector,
        url_after=page.url,
    )


async def _type_text(page: Page, selector: str, text: str, timeout: int) -> ActionResult:
    """
    Universally types text into any field: standard inputs, textareas,
    rich text editors, contenteditable containers, and code editors (Monaco, CodeMirror, Ace).
    """
    locator = page.locator(selector).first
    try:
        if await _count(locator) == 0:
            # Quick fallback to any visible search or text input on the page
            for cand_sel in ['input[type="search"]', 'input[type="text"]', 'input:not([type])', 'textarea', '[contenteditable="true"]']:
                cand = page.locator(cand_sel).first
                if await _count(cand) > 0 and await cand.is_visible():
                    locator = cand
                    break
    except Exception:
        pass

    try:
        await locator.wait_for(state="attached", timeout=min(timeout, 2000))
        await locator.scroll_into_view_if_needed(timeout=2000)
        await locator.click(timeout=min(timeout, 2000))
    except Exception:
        return ActionResult(
            action_type=ActionType.TYPE,
            success=False,
            selector=selector,
            input_value=text,
            error=f"No input element found for '{selector}' on the page.",
            url_after=page.url,
        )

    await asyncio.sleep(0.1)
    typed_successfully = False

    # Strategy 1: Standard input / textarea fill
    try:
        await locator.fill(text, timeout=2000)
        typed_successfully = True
    except Exception:
        pass

    # Strategy 2: Focus + Select All + Keyboard insert_text
    if not typed_successfully:
        try:
            await locator.focus(timeout=2000)
            import sys
            mod_key = "Meta" if sys.platform == "darwin" else "Control"
            await page.keyboard.press(f"{mod_key}+A")
            await page.keyboard.press("Backspace")
            await asyncio.sleep(0.05)
            await page.keyboard.insert_text(text)
            typed_successfully = True
        except Exception as kb_err:
            logger.debug("keyboard_insert_failed", error=str(kb_err))

    # Strategy 3: Direct JS injection into CodeMirror / Ace / contenteditable.
    # Playwright passes the element as the function's FIRST argument and the
    # value as the second. This used to be called as ([el, val]) with
    # [locator, text], which always threw (a Locator cannot be sent to the page),
    # so this fallback never ran.
    if not typed_successfully:
        try:
            js_inject = """
            (el, val) => {
                if (el.isContentEditable) {
                    el.innerText = val;
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                    return true;
                }
                if (el.CodeMirror) {
                    el.CodeMirror.setValue(val);
                    return true;
                }
                const cmEl = el.closest('.CodeMirror') || el.querySelector('.CodeMirror');
                if (cmEl && cmEl.CodeMirror) {
                    cmEl.CodeMirror.setValue(val);
                    return true;
                }
                if (window.monaco && window.monaco.editor) {
                    const editors = window.monaco.editor.getEditors();
                    if (editors.length > 0) {
                        editors[0].setValue(val);
                        return true;
                    }
                }
                return false;
            }
            """
            success_js = await locator.evaluate(js_inject, text, timeout=2000)
            if success_js:
                typed_successfully = True
        except Exception:
            pass

    if not typed_successfully:
        raise RuntimeError(
            f"Could not type into element '{selector}' using standard, keyboard, or editor protocols."
        )

    return ActionResult(
        action_type=ActionType.TYPE,
        success=True,
        selector=selector,
        input_value=text,
        url_after=page.url,
    )


async def _navigate(page: Page, url: str, timeout: int) -> ActionResult:
    """Navigate to a URL.

    Succeeds once the site has answered (the navigation committed), then gives
    the page up to `timeout` ms to build its DOM. Found 2026-09-15: chatgpt.com
    took longer than 10 s to reach DOMContentLoaded, which counted as a failed
    navigation, so the agent re-opened the same URL three times (35 s) while
    the browser was already on the site. A page that is still loading is now
    reported as reached, with metadata still_loading=True.
    """
    await page.goto(url, wait_until="commit", timeout=max(timeout, 15000))
    still_loading = False
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=timeout)
    except PlaywrightTimeout:
        still_loading = True

    return ActionResult(
        action_type=ActionType.NAVIGATE,
        success=True,
        input_value=url,
        url_after=page.url,
        metadata={"still_loading": still_loading},
    )


async def _scroll(page: Page, direction: str) -> ActionResult:
    """Scroll the page. direction = 'up' | 'down' | number of pixels."""
    try:
        pixels = int(direction)
    except ValueError:
        pixels = -500 if direction.lower() == "up" else 500

    await page.evaluate(f"window.scrollBy(0, {pixels})")
    await asyncio.sleep(0.5)  # Let content load after scroll

    return ActionResult(
        action_type=ActionType.SCROLL,
        success=True,
        input_value=direction,
        url_after=page.url,
    )


async def _wait_for(page: Page, selector: str, timeout: int) -> ActionResult:
    """Wait for an element to become visible."""
    locator = page.locator(selector).first
    await locator.wait_for(state="visible", timeout=timeout)

    return ActionResult(
        action_type=ActionType.WAIT_FOR,
        success=True,
        selector=selector,
        url_after=page.url,
    )


async def _press_key(page: Page, key: str) -> ActionResult:
    """Press a keyboard key (e.g., 'Enter', 'Escape', 'Tab')."""
    await page.keyboard.press(key)
    await asyncio.sleep(0.3)

    return ActionResult(
        action_type=ActionType.PRESS_KEY,
        success=True,
        input_value=key,
        url_after=page.url,
    )


async def _select_option(page: Page, selector: str, value: str, timeout: int) -> ActionResult:
    """Select a dropdown option by value or label."""
    locator = page.locator(selector).first
    await locator.wait_for(state="visible", timeout=timeout)
    await locator.select_option(value, timeout=timeout)

    return ActionResult(
        action_type=ActionType.SELECT_OPTION,
        success=True,
        selector=selector,
        input_value=value,
        url_after=page.url,
    )


async def _get_text(page: Page, selector: str, timeout: int) -> ActionResult:
    """Extract the text content of an element."""
    locator = page.locator(selector).first
    await locator.wait_for(state="visible", timeout=timeout)
    text = await locator.inner_text(timeout=timeout)

    return ActionResult(
        action_type=ActionType.GET_TEXT,
        success=True,
        selector=selector,
        extracted_text=text.strip(),
        url_after=page.url,
    )


async def _hover(page: Page, selector: str, timeout: int) -> ActionResult:
    """Hover over an element."""
    locator = page.locator(selector).first
    await locator.wait_for(state="visible", timeout=timeout)
    await locator.hover(timeout=timeout)

    return ActionResult(
        action_type=ActionType.HOVER,
        success=True,
        selector=selector,
        url_after=page.url,
    )


async def _go_back(page: Page) -> ActionResult:
    """Navigate back in browser history."""
    await page.go_back(wait_until="domcontentloaded")

    return ActionResult(
        action_type=ActionType.GO_BACK,
        success=True,
        url_after=page.url,
    )


# =============================================================================
# Interrupt detection (CAPTCHA, login, modals)
# =============================================================================

async def detect_interrupts(page: Page, interrupt_selectors: dict[str, str]) -> dict[str, bool]:
    """
    Check the current page for known interrupts (CAPTCHA, login, location modal).

    Args:
        interrupt_selectors: Dict of {interrupt_name: css_selector}.

    Returns:
        Dict of {interrupt_name: is_present} for any detected interrupts.
    """
    detected: dict[str, bool] = {}

    for name, selector in interrupt_selectors.items():
        try:
            is_visible = await page.locator(selector).first.is_visible(timeout=1000)
            if is_visible:
                detected[name] = True
                logger.warning("interrupt_detected", type=name, selector=selector)
        except Exception:
            pass

    return detected
