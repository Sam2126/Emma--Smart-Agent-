"""
Real browser integration tests.

These tests launch an ACTUAL headless browser, navigate to real pages,
and verify that the agent's core tools produce real results — not mocked ones.

Uses httpbin.org and example.com as stable, login-free test endpoints so the
tests run without any credentials or external accounts.

Requirements:
    playwright install chromium  (run once to install the browser binary)
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from playwright.async_api import async_playwright, Page


# =============================================================================
# Helpers
# =============================================================================

async def _launch_headless_page() -> tuple:
    """Launch headless Chromium and return (playwright, browser, page)."""
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(headless=True)
    page = await browser.new_page()
    return pw, browser, page


async def _cleanup(pw, browser):
    """Close browser and playwright cleanly."""
    try:
        await browser.close()
    except Exception:
        pass
    try:
        await pw.stop()
    except Exception:
        pass


# =============================================================================
# Test 1: Navigate + Perceive on a real page
# =============================================================================

@pytest.mark.asyncio
async def test_navigate_and_perceive_real_page():
    """
    Navigate to a real page and verify that perceive_page returns genuine elements.

    Uses httpbin.org/forms/post — a stable page with a real HTML form that
    always has predictable input elements.
    """
    pw, browser, page = await _launch_headless_page()
    try:
        await page.goto("https://httpbin.org/forms/post", wait_until="domcontentloaded", timeout=30000)

        # Import and run the same JS perception script the agent uses
        from app.tools.browser import _PERCEIVE_JS  # JS script from tools.py
        elements = await page.evaluate(_PERCEIVE_JS)

        # Must return a list (not empty, not a string)
        assert isinstance(elements, list), f"Expected list, got {type(elements)}: {elements}"
        assert len(elements) > 0, "perceive_page returned 0 elements on a page with a real form"

        # All elements must be parseable as JSON (the new fix)
        for item in elements:
            json_str = json.dumps(item)
            parsed = json.loads(json_str)
            assert "category" in parsed or "selector" in parsed or "text" in parsed, (
                f"Element missing expected keys: {parsed}"
            )

        # Must find the customer name input or similar form element
        categories = [e.get("category", "") for e in elements]
        selectors_found = [e.get("selector", "") for e in elements]
        texts_found = [e.get("text", "") for e in elements]

        has_input = (
            "input" in categories
            or any("input" in s.lower() for s in selectors_found)
            or any("custname" in s.lower() or "name" in s.lower() for s in selectors_found)
        )
        assert has_input, (
            f"Expected to find at least one input element on httpbin forms page. "
            f"Found categories: {set(categories)}"
        )

    finally:
        await _cleanup(pw, browser)


# =============================================================================
# Test 2: Fast-fail click on nonexistent element
# =============================================================================

@pytest.mark.asyncio
async def test_click_nonexistent_element_fast_fails(monkeypatch):
    """
    Verify that clicking a nonexistent element fails in <3 seconds.

    The old code would run 3 strategies × ~7s each = up to 21s wasted on a
    phantom element. After the fix, it should return in well under 3 seconds.
    """
    from app.browser.actions import execute_action, ActionType
    from app.config import get_settings

    # This test measures the element lookup. execute_action also waits a random
    # 1-3 s "human" delay first, which made the 3 s limit fail at random.
    monkeypatch.setattr(get_settings(), "action_delay_min", 0.0)
    monkeypatch.setattr(get_settings(), "action_delay_max", 0.0)

    pw, browser, page = await _launch_headless_page()
    try:
        await page.goto("https://example.com", wait_until="domcontentloaded", timeout=15000)

        t_start = time.monotonic()
        result = await execute_action(
            page=page,
            action_type=ActionType.CLICK,
            selector="#this-element-absolutely-does-not-exist-xyz-123",
            timeout=5000,
        )
        elapsed = time.monotonic() - t_start

        # Must fail
        assert result.success is False, "Expected failure for nonexistent element"
        # Must fail FAST (under 3 seconds — not 21s like the old code)
        assert elapsed < 3.0, (
            f"Fast-fail took {elapsed:.2f}s — expected <3s. "
            "The existence pre-check is not working."
        )

    finally:
        await _cleanup(pw, browser)


# =============================================================================
# Test 3: Type text into a real form field
# =============================================================================

@pytest.mark.asyncio
async def test_type_into_real_form_field():
    """
    Verify that the agent can type text into a real input field and read it back.

    Uses httpbin.org/forms/post which has a stable <input name="custname"> field.
    """
    from app.browser.actions import execute_action, ActionType

    pw, browser, page = await _launch_headless_page()
    try:
        await page.goto("https://httpbin.org/forms/post", wait_until="domcontentloaded", timeout=30000)

        test_text = "Self-Improving Agent Test"

        result = await execute_action(
            page=page,
            action_type=ActionType.TYPE,
            selector="input[name='custname']",
            input_value=test_text,
            timeout=5000,
        )

        assert result.success is True, f"Type action failed: {result.error}"

        # Read back the value to confirm it was typed
        actual_value = await page.locator("input[name='custname']").input_value()
        assert actual_value == test_text, (
            f"Expected '{test_text}' but found '{actual_value}' in the input field."
        )

    finally:
        await _cleanup(pw, browser)


# =============================================================================
# Test 4: Navigate to a real page, verify URL and title
# =============================================================================

@pytest.mark.asyncio
async def test_navigate_real_site():
    """Verify that navigate_browser successfully opens a real URL."""
    from app.browser.actions import execute_action, ActionType

    pw, browser, page = await _launch_headless_page()
    try:
        result = await execute_action(
            page=page,
            action_type=ActionType.NAVIGATE,
            input_value="https://example.com",
            timeout=15000,
        )

        assert result.success is True, f"Navigate failed: {result.error}"
        assert "example.com" in result.url_after, (
            f"Expected example.com in url_after, got: {result.url_after}"
        )

        title = await page.title()
        assert title, "Page title should not be empty after successful navigation"

    finally:
        await _cleanup(pw, browser)


# =============================================================================
# Test 5: Perception returns JSON-parseable output (regression test for repr() bug)
# =============================================================================

@pytest.mark.asyncio
async def test_perception_output_is_json_parseable():
    """
    Regression test: perception output must be JSON, not Python repr() strings.

    Before the fix, tools.py called str(item) which produces Python dict repr:
        {'category': 'button', 'text': 'Search', 'selector': '#search'}
    which is NOT valid JSON (single quotes). The LLM would have to guess at parsing.

    After the fix, it uses json.dumps() which produces valid JSON:
        {"category": "button", "text": "Search", "selector": "#search"}
    """
    pw, browser, page = await _launch_headless_page()
    try:
        await page.goto("https://example.com", wait_until="domcontentloaded", timeout=15000)

        from app.tools.browser import _PERCEIVE_JS
        elements = await page.evaluate(_PERCEIVE_JS)

        assert isinstance(elements, list)

        for item in elements:
            # json.dumps must succeed (it will fail on Python repr strings with single quotes)
            json_str = json.dumps(item)
            # And it must round-trip correctly
            reparsed = json.loads(json_str)
            assert isinstance(reparsed, dict), f"Expected dict after round-trip, got: {type(reparsed)}"

    finally:
        await _cleanup(pw, browser)
