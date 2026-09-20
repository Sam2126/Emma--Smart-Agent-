"""
Generic, site-independent task verification rules.

Only Amazon has hand-written rules (rules/amazon.py). On every other site,
task verification rested entirely on an LLM reading page text. These rules read
what the instruction asked for and check concrete page evidence that holds on
any site:

  search  -> the query's words show up in the URL, title or page text
  go to X -> the browser actually reached that site
  cart    -> the cart counter went up (only when both pages expose one)
  always  -> the final page is not a login wall, CAPTCHA or error page,
             unless logging in was the task

Rules are only added when the instruction makes them meaningful, so a task
this module can't interpret gets no rules rather than wrong ones.
"""

from __future__ import annotations

import math
import re
from typing import Any
from urllib.parse import unquote_plus, urlparse

from app.browser.page_state import PageState
from app.verifier.rule_checks import CheckResult, NumericIncrementCheck, RuleCheck

_STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "of", "on", "in", "at", "to", "from", "with",
    "under", "below", "over", "above", "best", "good", "top", "some", "any", "me", "my",
    "please", "it", "its", "then", "now", "ok", "okay", "open", "search", "find", "show",
    "rs", "inr", "price", "cheap", "new",
}

_KNOWN_SITES = {
    "amazon": "amazon", "flipkart": "flipkart", "myntra": "myntra", "youtube": "youtube",
    "google": "google", "ebay": "ebay", "walmart": "walmart", "wikipedia": "wikipedia",
    "github": "github", "linkedin": "linkedin", "twitter": "twitter", "reddit": "reddit",
    "netflix": "netflix", "spotify": "spotify", "crazygames": "crazygames", "poki": "poki",
    "stackoverflow": "stackoverflow", "leetcode": "leetcode", "meesho": "meesho",
    "ajio": "ajio", "nykaa": "nykaa", "swiggy": "swiggy", "zomato": "zomato",
}

_DOMAIN_RE = re.compile(r"\b([a-z0-9][a-z0-9\-]*\.(?:com|in|org|net|io|co|dev|app|ai|co\.in|edu|gov))\b")

_LOGIN_URL_MARKERS = ("/login", "/signin", "/sign-in", "/ap/signin", "accounts.google.com", "/auth/", "/session/new")
_BLOCK_TEXT_MARKERS = (
    "sign in to continue", "log in to continue", "login to continue", "verify you are human",
    "are you a robot", "captcha", "access denied", "403 forbidden", "page not found",
    "this site can't be reached", "err_connection",
)


def infer_task_intent(instruction: str) -> dict[str, Any]:
    text = (instruction or "").lower()
    intent: dict[str, Any] = {
        "search_query": None,
        "target_site": None,
        "add_to_cart": bool(re.search(r"\badd\b.*\b(cart|basket|bag)\b", text)),
        "login_task": bool(re.search(r"\b(log ?in|sign ?in)\b", text)) and not re.search(r"\b(log ?in|sign ?in) (to continue|wall)\b", text),
    }

    m = re.search(
        r"\b(?:search|find|look up|look for)(?:\s+for)?\s+(.+?)(?=\s+(?:on|in|at|from|and|then|under|below|with)\b|[.,;!?]|$)",
        text,
    )
    if m:
        query = m.group(1).strip(" '\"")
        if query and query not in ("it", "that", "this", "tab"):
            intent["search_query"] = query

    domain = _DOMAIN_RE.search(text)
    if domain:
        intent["target_site"] = domain.group(1).split(".")[0]
    else:
        for name, keyword in _KNOWN_SITES.items():
            if re.search(rf"\b{name}\b", text):
                intent["target_site"] = keyword
                break
    return intent


def _significant_tokens(query: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    return [t for t in tokens if len(t) >= 2 and t not in _STOPWORDS] or tokens[:3]


class SearchQueryReflectedCheck(RuleCheck):
    """Most of the query's words appear in the URL, title or visible text."""

    def __init__(self, query: str) -> None:
        self.query = query
        self.tokens = _significant_tokens(query)

    @property
    def name(self) -> str:
        return "search_query_reflected"

    async def check(self, before, after: PageState, context=None) -> CheckResult:
        haystack = " ".join([
            unquote_plus(after.url or "").lower(),
            (after.title or "").lower(),
            (after.visible_text or "")[:6000].lower(),
        ])
        hits = [t for t in self.tokens if t in haystack]
        needed = max(1, math.ceil(len(self.tokens) * 0.5))
        return CheckResult(
            check_name=self.name,
            passed=len(hits) >= needed,
            expected=f"at least {needed} of {self.tokens}",
            actual=f"found {hits}",
            details=f"Search for '{self.query}' should be reflected on the results page.",
        )


class ReachedSiteCheck(RuleCheck):
    """The final page belongs to the site the instruction named."""

    def __init__(self, site_keyword: str) -> None:
        self.site = site_keyword

    @property
    def name(self) -> str:
        return "reached_requested_site"

    async def check(self, before, after: PageState, context=None) -> CheckResult:
        netloc = urlparse(after.url or "").netloc.lower()
        return CheckResult(
            check_name=self.name,
            passed=self.site in netloc,
            expected=f"host contains '{self.site}'",
            actual=netloc or "(no url)",
        )


class NotBlockedCheck(RuleCheck):
    """The final page is not a login wall, CAPTCHA or error page."""

    @property
    def name(self) -> str:
        return "not_blocked_by_login_or_error"

    async def check(self, before, after: PageState, context=None) -> CheckResult:
        url = (after.url or "").lower()
        text = f"{(after.title or '').lower()} {(after.visible_text or '')[:3000].lower()}"
        url_hit = next((m for m in _LOGIN_URL_MARKERS if m in url), None)
        text_hit = next((m for m in _BLOCK_TEXT_MARKERS if m in text), None)
        blocked = url_hit or text_hit
        return CheckResult(
            check_name=self.name,
            passed=not blocked,
            expected="a normal content page",
            actual=f"blocked by '{blocked}'" if blocked else "no login wall, CAPTCHA or error detected",
        )


def get_generic_task_rules(
    instruction: str,
    initial_state: PageState | None,
    final_state: PageState | None,
) -> list[RuleCheck]:
    intent = infer_task_intent(instruction)
    rules: list[RuleCheck] = []
    if intent["search_query"]:
        rules.append(SearchQueryReflectedCheck(intent["search_query"]))
    if intent["target_site"]:
        rules.append(ReachedSiteCheck(intent["target_site"]))
    if (
        intent["add_to_cart"]
        and initial_state is not None
        and final_state is not None
        and "cart_count" in (initial_state.extracted_values or {})
        and "cart_count" in (final_state.extracted_values or {})
    ):
        rules.append(NumericIncrementCheck("cart_count", min_increment=1))
    if not intent["login_task"]:
        rules.append(NotBlockedCheck())
    return rules
