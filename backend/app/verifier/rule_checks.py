"""
Rule-based verification checks.

These are the first (and most trustworthy) tier of the verification system.
Each check is a simple, deterministic function that inspects the page state
before and after an action to confirm expected changes occurred.

Rule-based checks are cheap, fast, and impossible to "fool" — unlike LLM
judgments, they either pass or fail based on concrete DOM evidence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import structlog

from app.browser.page_state import PageState

logger = structlog.get_logger(__name__)


@dataclass
class CheckResult:
    """Result of a single rule-based check."""
    check_name: str
    passed: bool
    expected: str = ""
    actual: str = ""
    details: str = ""


class RuleCheck(ABC):
    """Base class for all rule-based verification checks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this check."""
        ...

    @abstractmethod
    async def check(
        self,
        before: PageState | None,
        after: PageState,
        context: dict[str, Any] | None = None,
    ) -> CheckResult:
        """
        Run this check against page state(s).

        Args:
            before: Page state before the action (None for initial checks).
            after: Page state after the action.
            context: Additional context (e.g., expected product name, price).

        Returns:
            CheckResult with pass/fail and details.
        """
        ...


# =============================================================================
# Concrete Rule Checks
# =============================================================================


class URLContainsCheck(RuleCheck):
    """Check that the current URL contains an expected pattern."""

    def __init__(self, pattern: str) -> None:
        self._pattern = pattern

    @property
    def name(self) -> str:
        return f"url_contains:{self._pattern}"

    async def check(
        self,
        before: PageState | None,
        after: PageState,
        context: dict[str, Any] | None = None,
    ) -> CheckResult:
        passed = self._pattern in after.url
        return CheckResult(
            check_name=self.name,
            passed=passed,
            expected=f"URL contains '{self._pattern}'",
            actual=after.url,
            details="" if passed else f"Pattern '{self._pattern}' not found in URL",
        )


class URLChangedCheck(RuleCheck):
    """Check that the URL changed after an action."""

    @property
    def name(self) -> str:
        return "url_changed"

    async def check(
        self,
        before: PageState | None,
        after: PageState,
        context: dict[str, Any] | None = None,
    ) -> CheckResult:
        if before is None:
            return CheckResult(
                check_name=self.name,
                passed=True,
                details="No 'before' state — skipping URL change check",
            )

        passed = before.url != after.url
        return CheckResult(
            check_name=self.name,
            passed=passed,
            expected="URL should have changed",
            actual=f"Before: {before.url} → After: {after.url}",
        )


class ElementExistsCheck(RuleCheck):
    """Check that a specific element exists in the extracted values."""

    def __init__(self, key: str) -> None:
        self._key = key

    @property
    def name(self) -> str:
        return f"element_exists:{self._key}"

    async def check(
        self,
        before: PageState | None,
        after: PageState,
        context: dict[str, Any] | None = None,
    ) -> CheckResult:
        value = after.extracted_values.get(self._key)
        passed = value is not None and len(value.strip()) > 0
        return CheckResult(
            check_name=self.name,
            passed=passed,
            expected=f"'{self._key}' should exist in extracted values",
            actual=f"Value: {value!r}" if value else "Not found",
        )


class ElementTextContainsCheck(RuleCheck):
    """Check that an extracted value contains expected text."""

    def __init__(self, key: str, expected_text: str, case_sensitive: bool = False) -> None:
        self._key = key
        self._expected_text = expected_text
        self._case_sensitive = case_sensitive

    @property
    def name(self) -> str:
        return f"text_contains:{self._key}:{self._expected_text}"

    async def check(
        self,
        before: PageState | None,
        after: PageState,
        context: dict[str, Any] | None = None,
    ) -> CheckResult:
        value = after.extracted_values.get(self._key, "")

        if self._case_sensitive:
            passed = self._expected_text in value
        else:
            passed = self._expected_text.lower() in value.lower()

        return CheckResult(
            check_name=self.name,
            passed=passed,
            expected=f"'{self._key}' contains '{self._expected_text}'",
            actual=f"Value: {value!r}",
        )


class ValueChangedCheck(RuleCheck):
    """Check that a specific extracted value changed after an action."""

    def __init__(self, key: str) -> None:
        self._key = key

    @property
    def name(self) -> str:
        return f"value_changed:{self._key}"

    async def check(
        self,
        before: PageState | None,
        after: PageState,
        context: dict[str, Any] | None = None,
    ) -> CheckResult:
        if before is None:
            return CheckResult(
                check_name=self.name,
                passed=True,
                details="No 'before' state — skipping value change check",
            )

        before_val = before.extracted_values.get(self._key, "")
        after_val = after.extracted_values.get(self._key, "")
        passed = before_val != after_val

        return CheckResult(
            check_name=self.name,
            passed=passed,
            expected=f"'{self._key}' should have changed",
            actual=f"Before: {before_val!r} → After: {after_val!r}",
        )


class NumericIncrementCheck(RuleCheck):
    """Check that a numeric value incremented (e.g., cart count)."""

    def __init__(self, key: str, min_increment: int = 1) -> None:
        self._key = key
        self._min_increment = min_increment

    @property
    def name(self) -> str:
        return f"numeric_increment:{self._key}"

    async def check(
        self,
        before: PageState | None,
        after: PageState,
        context: dict[str, Any] | None = None,
    ) -> CheckResult:
        if before is None:
            return CheckResult(
                check_name=self.name,
                passed=True,
                details="No 'before' state — skipping increment check",
            )

        try:
            before_val = int(before.extracted_values.get(self._key, "0"))
            after_val = int(after.extracted_values.get(self._key, "0"))
            passed = (after_val - before_val) >= self._min_increment
        except ValueError:
            passed = False
            before_val = before.extracted_values.get(self._key, "?")
            after_val = after.extracted_values.get(self._key, "?")

        return CheckResult(
            check_name=self.name,
            passed=passed,
            expected=f"'{self._key}' incremented by at least {self._min_increment}",
            actual=f"Before: {before_val} → After: {after_val}",
        )


class TitleContainsCheck(RuleCheck):
    """Check that the page title contains expected text."""

    def __init__(self, expected_text: str) -> None:
        self._expected_text = expected_text

    @property
    def name(self) -> str:
        return f"title_contains:{self._expected_text}"

    async def check(
        self,
        before: PageState | None,
        after: PageState,
        context: dict[str, Any] | None = None,
    ) -> CheckResult:
        passed = self._expected_text.lower() in after.title.lower()
        return CheckResult(
            check_name=self.name,
            passed=passed,
            expected=f"Title contains '{self._expected_text}'",
            actual=f"Title: {after.title!r}",
        )


class VisibleTextContainsCheck(RuleCheck):
    """Check that the visible page text contains expected content."""

    def __init__(self, expected_text: str) -> None:
        self._expected_text = expected_text

    @property
    def name(self) -> str:
        return f"visible_text_contains:{self._expected_text}"

    async def check(
        self,
        before: PageState | None,
        after: PageState,
        context: dict[str, Any] | None = None,
    ) -> CheckResult:
        passed = self._expected_text.lower() in after.visible_text.lower()
        return CheckResult(
            check_name=self.name,
            passed=passed,
            expected=f"Page text contains '{self._expected_text}'",
            actual=f"Text length: {len(after.visible_text)} chars",
            details="" if passed else f"'{self._expected_text}' not found in visible text",
        )
