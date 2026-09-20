"""
Speed and robustness fixes from the 2026-09-15 run log, where a Google search
took 118 s and a ChatGPT task stalled for minutes:

  * clicks: bounded element lookup, text strategies only for text selectors
  * navigation: a slow page counts as reached instead of failed
  * Gemini: an overloaded model is skipped for a while
  * verification: the vision judge only runs when it can change the result
  * page state: accessibility tree from Playwright's ARIA snapshot
  * Chrome port probe, quiet /health logs
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time

import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeout

from app.browser import actions
from app.browser.page_state import PageState, _parse_aria_snapshot
from app.config import get_settings
from app.utils import llm
from app.verifier import engine as engine_module
from app.verifier.engine import VerificationEngine, VerificationTier
from app.verifier.llm_judge import LLMJudgment
from tests.conftest import llm_message


# =============================================================================
# Clicks
# =============================================================================

@pytest.mark.parametrize("selector,expected", [
    ('a:has-text("ChatGPT – OpenAI")', "ChatGPT – OpenAI"),
    ("button:has-text('Add to Cart')", "Add to Cart"),
    ("text=Sign in", "Sign in"),
    ("Add to cart", "Add to cart"),
    ('a[href*="chatgpt.com"]', ""),
    ("#nav-search-submit-button", ""),
    ("input[name='q']", ""),
])
def test_text_hint(selector, expected):
    assert actions._text_hint(selector) == expected


class _Locator:
    """Minimal stand-in for a Playwright locator."""

    def __init__(self, count: int, visible: int | None = None, delay: float = 0.0):
        self.n = count
        self.visible_n = count if visible is None else visible
        self.delay = delay
        self.first = self

    async def count(self) -> int:
        await asyncio.sleep(self.delay)
        return self.n

    def filter(self, visible: bool = False) -> "_Locator":
        return _Locator(self.visible_n, delay=self.delay)


async def test_count_gives_up_instead_of_hanging():
    started = time.monotonic()
    assert await actions._count(_Locator(1, delay=5), timeout=0.1) == 0
    assert time.monotonic() - started < 1


async def test_a_visible_match_wins_over_a_hidden_one():
    hidden, shown = _Locator(1, visible=0), _Locator(1, visible=1)
    found = await actions._first_present([lambda: hidden, lambda: shown], time.monotonic() + 5)
    assert found is not None and found.visible_n == 1


async def test_a_hidden_match_is_used_when_nothing_is_visible():
    hidden = _Locator(1, visible=0)
    assert await actions._first_present([lambda: hidden], time.monotonic() + 5) is hidden


async def test_lookup_stops_at_the_deadline():
    slow = [lambda: _Locator(0, delay=0.3) for _ in range(20)]
    started = time.monotonic()
    assert await actions._first_present(slow, time.monotonic() + 0.5) is None
    assert time.monotonic() - started < 2


# =============================================================================
# Navigation
# =============================================================================

class _SlowPage:
    url = "https://chatgpt.com/"

    def __init__(self) -> None:
        self.gotos: list[tuple[str, int]] = []

    async def goto(self, url, wait_until, timeout):
        self.gotos.append((wait_until, timeout))

    async def wait_for_load_state(self, state, timeout):
        raise PlaywrightTimeout("Timeout 10000ms exceeded.")


class _CheckPage:
    def __init__(self, url: str, title: str, clears_after: int | None = None):
        self._url, self._title, self.clears_after, self.polls = url, title, clears_after, 0

    @property
    def url(self) -> str:
        return "https://chatgpt.com/" if self.clears_after is not None and self.polls > self.clears_after else self._url

    async def title(self) -> str:
        self.polls += 1
        return "ChatGPT" if self.clears_after is not None and self.polls > self.clears_after else self._title


async def test_bot_check_is_waited_out_when_it_clears(monkeypatch):
    from app.tools import browser

    monkeypatch.setattr(browser.asyncio, "sleep", _no_sleep)
    page = _CheckPage("https://chatgpt.com/?__cf_chl_rt_tk=abc", "Just a moment...", clears_after=2)
    assert await browser._bot_check_remains(page, wait_seconds=30) is False


async def test_bot_check_that_stays_is_reported(monkeypatch):
    from app.tools import browser

    page = _CheckPage("https://chatgpt.com/?__cf_chl_rt_tk=abc", "Just a moment...")
    started = time.monotonic()
    assert await browser._bot_check_remains(page, wait_seconds=0.2) is True
    assert time.monotonic() - started < 3
    assert await browser._bot_check_remains(_CheckPage("https://www.google.com/", "Google"), wait_seconds=5) is False


async def _no_sleep(seconds):
    return None


async def test_a_slow_page_counts_as_reached():
    page = _SlowPage()
    result = await actions._navigate(page, "https://chatgpt.com", 10000)
    assert result.success is True
    assert result.metadata["still_loading"] is True
    assert page.gotos == [("commit", 15000)]


# =============================================================================
# Page state
# =============================================================================

def test_aria_snapshot_is_parsed_into_interactive_nodes():
    snapshot = """- banner:
  - link "Gmail":
    - /url: https://mail.google.com
  - img "Google"
- search:
  - combobox "Search" [expanded=false]
  - button "Google Search"
- heading "ChatGPT – OpenAI" [level=3]
- text: Some paragraph
- checkbox "Remember me" [checked]
- button "Sign in" [disabled]
"""
    nodes = _parse_aria_snapshot(snapshot)
    assert {"role": "link", "name": "Gmail"} in nodes
    assert {"role": "combobox", "name": "Search"} in nodes
    assert {"role": "button", "name": "Google Search"} in nodes
    assert {"role": "heading", "name": "ChatGPT – OpenAI", "level": 3} in nodes
    assert {"role": "checkbox", "name": "Remember me", "checked": True} in nodes
    assert {"role": "button", "name": "Sign in", "disabled": True} in nodes
    assert not any(node["role"] in ("banner", "search", "text") for node in nodes)
    assert "[link]" in PageState(accessibility_tree=nodes).to_llm_context()


# =============================================================================
# Gemini circuit breaker
# =============================================================================

@pytest.fixture
def fresh_breaker():
    llm._reset_for_tests()
    yield
    llm._reset_for_tests()


def test_overloaded_gemini_model_is_skipped_for_a_while(fresh_breaker):
    high_demand = RuntimeError('503 {"message": "This model is currently experiencing high demand"}')
    assert llm.mark_gemini_unavailable("gemini/busy", high_demand) == 120
    assert llm.mark_gemini_unavailable("gemini/slow", asyncio.TimeoutError()) == 120
    assert llm.mark_gemini_unavailable("gemini/limited", RuntimeError("429 RESOURCE_EXHAUSTED quota")) == 300
    assert llm.mark_gemini_unavailable("gemini/bad", ValueError("400 invalid argument")) == 0
    assert not llm.gemini_model_available("gemini/busy")
    assert llm.gemini_model_available("gemini/bad")
    assert llm.order_by_availability(["gemini/busy", "gemini/ok", "gemini/slow"]) == [
        "gemini/ok", "gemini/busy", "gemini/slow",
    ]


async def test_vision_skips_a_cooling_gemini_model(monkeypatch, fresh_breaker):
    import litellm

    from app.utils import vision

    s = get_settings()
    monkeypatch.setattr(s, "gemini_api_key", "k")
    monkeypatch.setattr(s, "gemini_vision_model", "gemini/busy,gemini/ok")
    llm.mark_gemini_unavailable("gemini/busy", RuntimeError("503 high demand"))
    tried: list[str] = []

    async def fake(**kwargs):
        tried.append(kwargs["model"])
        return llm_message("a search results page")

    monkeypatch.setattr(litellm, "acompletion", fake)
    assert await vision._describe_with_gemini("aGk=", "what?", "image/png", 100) == "a search results page"
    assert tried == ["gemini/ok"]


async def test_a_slow_vision_model_times_out_and_the_next_one_answers(monkeypatch, fresh_breaker):
    import litellm

    from app.utils import vision

    s = get_settings()
    monkeypatch.setattr(s, "gemini_api_key", "k")
    monkeypatch.setattr(s, "gemini_vision_model", "gemini/slow,gemini/ok")
    monkeypatch.setattr(vision, "GEMINI_VISION_TIMEOUT_SECONDS", 0.05)
    tried: list[str] = []

    async def fake(**kwargs):
        tried.append(kwargs["model"])
        if kwargs["model"] == "gemini/slow":
            await asyncio.sleep(1)
        return llm_message("a login form")

    monkeypatch.setattr(litellm, "acompletion", fake)
    assert await vision._describe_with_gemini("aGk=", "what?", "image/png", 100) == "a login form"
    assert tried == ["gemini/slow", "gemini/ok"]
    assert not llm.gemini_model_available("gemini/slow")


# =============================================================================
# Verification: vision only when it can change the outcome
# =============================================================================

class _Judge:
    def __init__(self, text: LLMJudgment, vision: LLMJudgment | None = None, vision_delay: float = 0.0):
        self.text, self.vision, self.vision_delay = text, vision, vision_delay
        self.visual_calls = 0

    async def verify_task(self, instruction, state, summary):
        return self.text

    async def verify_task_visual(self, instruction, summary, screenshot):
        self.visual_calls += 1
        await asyncio.sleep(self.vision_delay)
        return self.vision


@pytest.fixture
def vision_on(monkeypatch):
    monkeypatch.setattr(get_settings(), "vision_verification_enabled", True)


async def _verify(judge: _Judge, instruction: str = "finish the form"):
    return await VerificationEngine(llm_judge=judge).verify_task(
        task_type="general",
        task_instruction=instruction,
        final_state=PageState(url="https://example.com/done", title="Done", visible_text="All set"),
        action_summary="1. click_element(#submit) -> ok",
        initial_state=None,
        screenshot_b64="aGk=",
    )


async def test_confident_text_pass_skips_the_vision_judge(vision_on):
    judge = _Judge(LLMJudgment(passed=True, confidence=0.95, reasoning="done"))
    result = await _verify(judge)
    assert result.passed and judge.visual_calls == 0
    assert "Vision: skipped (rules and text judge agree)" in result.summary


async def test_text_fail_asks_the_vision_judge(vision_on):
    judge = _Judge(LLMJudgment(passed=False, confidence=0.9, reasoning="no text"),
                   vision=LLMJudgment(passed=True, confidence=0.9, reasoning="result visible", source="vision"))
    result = await _verify(judge)
    assert judge.visual_calls == 1
    assert result.passed and result.tier == VerificationTier.VISION


async def test_unsure_text_pass_asks_the_vision_judge(vision_on):
    judge = _Judge(LLMJudgment(passed=True, confidence=0.6, reasoning="maybe"),
                   vision=LLMJudgment(passed=False, confidence=0.8, reasoning="popup covers it", source="vision"))
    result = await _verify(judge)
    assert judge.visual_calls == 1 and not result.passed


async def test_a_slow_vision_judge_is_cut_off(vision_on, monkeypatch):
    monkeypatch.setattr(engine_module, "VISION_JUDGE_TIMEOUT_SECONDS", 0.05)
    judge = _Judge(LLMJudgment(passed=True, confidence=0.5, reasoning="maybe"),
                   vision=LLMJudgment(passed=True, confidence=0.9, reasoning="ok"), vision_delay=1)
    started = time.monotonic()
    result = await _verify(judge)
    assert time.monotonic() - started < 1
    assert "Vision: timed out" in result.summary and result.passed


async def test_failed_rule_skips_the_vision_judge(vision_on):
    judge = _Judge(LLMJudgment(passed=True, confidence=0.5, reasoning="?"),
                   vision=LLMJudgment(passed=True, confidence=0.9, reasoning="ok"))
    result = await _verify(judge, instruction="search headphones on flipkart")
    assert judge.visual_calls == 0 and not result.passed
    assert "Vision: skipped (a rule already failed)" in result.summary


# =============================================================================
# Chrome port probe and quiet health checks
# =============================================================================

def test_port_probe_detects_listening_and_closed_ports():
    from app.browser.controller import _port_open

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        assert _port_open("127.0.0.1", port) is True
    finally:
        server.close()
    assert _port_open("127.0.0.1", port, timeout=0.3) is False


def test_health_checks_are_kept_out_of_the_access_log():
    from app.main import _QuietHealthChecks

    quiet = _QuietHealthChecks()

    def record(path: str) -> logging.LogRecord:
        return logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1,
                                 '%s - "%s %s HTTP/%s" %d', ("127.0.0.1:1", "GET", path, "1.1", 200), None)

    assert quiet.filter(record("/health")) is False
    assert quiet.filter(record("/local")) is True
