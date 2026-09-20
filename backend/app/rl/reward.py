"""
Reward Engine (Process Reward Model — PRM) for Browser Agent RL.

Implements dense step-level rewards and terminal outcome rewards strictly per
§6 of the RL Implementation Plan (v2):
- Rule-based milestones: URL milestones (+1.0), numeric deltas (+1.0),
  facet/filter applications (+1.0), target element existence (+0.5)
- Step cost for efficiency: -0.05 per step
- Terminal outcomes: PASS (+2.0), FAIL (-2.0)
- Anti-gaming penalties: spam clicks (-0.3, terminate after 3), fake search URLs (-1.0),
  checkout bypass attempts (-5.0, immediate abort), deprecated selectors (-1.0)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional
import structlog

from app.verifier.rule_checks import (
    CheckResult,
    URLContainsCheck,
    URLChangedCheck,
    ElementExistsCheck,
    NumericIncrementCheck,
)

logger = structlog.get_logger(__name__)

# Irreversible/financial action identifiers that require confirmation
IRREVERSIBLE_ACTION_TYPES = {
    "place_order",
    "confirm_payment",
    "checkout",
    "buy_now",
    "submit_order",
}

CHECKOUT_SELECTOR_PATTERNS = [
    r"buy[_-]?now",
    r"place[_-]?order",
    r"proceed[_-]?to[_-]?checkout",
    r"submit[_-]?payment",
]


@dataclass
class RewardBreakdown:
    """Detailed decomposition of step reward."""
    milestones: float = 0.0
    penalties: float = 0.0
    step_cost: float = -0.05
    terminal: float = 0.0
    total: float = 0.0
    details: list[str] = field(default_factory=list)
    gaming_flags: list[str] = field(default_factory=list)
    abort_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "milestones": round(self.milestones, 4),
            "penalties": round(self.penalties, 4),
            "step_cost": round(self.step_cost, 4),
            "terminal": round(self.terminal, 4),
            "total": round(self.total, 4),
            "details": self.details,
            "gaming_flags": self.gaming_flags,
            "abort_requested": self.abort_requested,
        }


def is_fake_search_url(action: dict[str, Any] | Any) -> bool:
    """Detect if an action tries to fake search navigation via URL query params."""
    action_type = getattr(action, "type", None) or (action.get("type") if isinstance(action, dict) else "")
    if action_type not in ("navigate", "goto", "open_url"):
        return False
    url = getattr(action, "url", None) or (action.get("url") if isinstance(action, dict) else "")
    if not url:
        url = getattr(action, "text", None) or (action.get("text") if isinstance(action, dict) else "")
    url_str = str(url).lower()
    # Query param patterns for search engines / e-commerce
    if re.search(r"/(s\?k=|search\?q=|search/\?keyword=)", url_str):
        return True
    return False


def is_checkout_action(action: dict[str, Any] | Any) -> bool:
    """Detect if an action attempts an irreversible checkout/payment."""
    action_type = (getattr(action, "type", None) or (action.get("type") if isinstance(action, dict) else "")).lower()
    if action_type in IRREVERSIBLE_ACTION_TYPES:
        return True
    selector = str(getattr(action, "selector", None) or (action.get("selector") if isinstance(action, dict) else "") or "").lower()
    for pat in CHECKOUT_SELECTOR_PATTERNS:
        if re.search(pat, selector):
            return True
    return False


def dense_reward(
    rules: list[CheckResult] | None = None,
    state_before: Any = None,
    state_after: Any = None,
    action: dict[str, Any] | Any = None,
    is_terminal: bool = False,
    terminal_pass: bool = False,
    terminal_fail: bool = False,
    consecutive_spam_clicks: int = 0,
    deprecated_selectors: set[str] | list[str] | None = None,
) -> RewardBreakdown:
    """
    Compute dense PRM reward and anti-gaming penalties for one step transition.

    Table:
      +1.0  URL milestone matched
      +1.0  Numeric delta (cart badge increment)
      +1.0  Facet/filter applied (element appears in DOM)
      +0.5  Target element exists
      -0.05 Step cost (efficiency)
      +2.0  Terminal PASS
      -2.0  Terminal FAIL

    Anti-gaming:
      -0.3  Spam-click repeat (terminate after 3)
      -1.0  Fake search-URL navigation
      -5.0  Checkout/payment without confirmation (aborts rollout)
      -1.0  Reuse of deprecated selector (strike count >= 3)
    """
    rb = RewardBreakdown()
    rules = rules or []
    deprecated_selectors = set(deprecated_selectors or [])

    # 1. Step Cost (time pressure -> efficiency)
    rb.step_cost = -0.05

    # 2. Rule Check Milestones
    for check in rules:
        if not check.passed:
            continue
        cname = check.check_name.lower()
        if "url_contains" in cname or "url_changed" in cname:
            rb.milestones += 1.0
            rb.details.append(f"+1.0 URL milestone matched ({check.check_name})")
        elif "numeric_increment" in cname:
            rb.milestones += 1.0
            rb.details.append(f"+1.0 Numeric delta increment ({check.check_name})")
        elif "facet" in cname or "filter" in cname:
            rb.milestones += 1.0
            rb.details.append(f"+1.0 Facet/filter applied ({check.check_name})")
        elif "element_exists" in cname or "text_contains" in cname:
            rb.milestones += 0.5
            rb.details.append(f"+0.5 Target element exists ({check.check_name})")

    # 3. State-level deltas if states are provided
    if state_before is not None and state_after is not None:
        # Check cart count increase if extracted_values available
        before_extracted = getattr(state_before, "extracted_values", {}) or {}
        after_extracted = getattr(state_after, "extracted_values", {}) or {}
        if isinstance(before_extracted, dict) and isinstance(after_extracted, dict):
            cart_before = int(before_extracted.get("cart_count", 0) or 0)
            cart_after = int(after_extracted.get("cart_count", 0) or 0)
            if cart_after > cart_before and not any("numeric_increment" in d for d in rb.details):
                rb.milestones += 1.0
                rb.details.append(f"+1.0 Cart count incremented ({cart_before} -> {cart_after})")

        # Check facet filter appearance
        if getattr(state_after, "url", "") != getattr(state_before, "url", ""):
            after_url = str(getattr(state_after, "url", "")).lower()
            if any(k in after_url for k in ["filter", "facet", "rh=", "p_n_"]):
                if not any("facet" in d.lower() for d in rb.details):
                    rb.milestones += 1.0
                    rb.details.append("+1.0 Filter/facet URL pattern matched")

    # 4. Anti-Gaming Violations
    if action:
        # 4a. Checkout bypass attempt
        if is_checkout_action(action):
            has_confirmed = (action.get("confirmed") if isinstance(action, dict) else getattr(action, "confirmed", False))
            if not has_confirmed:
                rb.penalties -= 5.0
                rb.abort_requested = True
                rb.gaming_flags.append("gaming: unauthorized_checkout_attempt")
                rb.details.append("-5.0 Checkout attempted without safety confirmation (ABORT)")

        # 4b. Fake search-URL navigation
        if is_fake_search_url(action):
            rb.penalties -= 1.0
            rb.gaming_flags.append("gaming: fake_search_url")
            rb.details.append("-1.0 Fake search-URL navigation penalty")

        # 4c. Deprecated selector penalty
        sel = getattr(action, "selector", None) or (action.get("selector") if isinstance(action, dict) else "")
        if sel and sel in deprecated_selectors:
            rb.penalties -= 1.0
            rb.gaming_flags.append("gaming: deprecated_selector_reuse")
            rb.details.append(f"-1.0 Deprecated selector reuse penalty ({sel})")

    # 4d. Spam click penalty
    if consecutive_spam_clicks > 0:
        penalty = -0.3 * consecutive_spam_clicks
        rb.penalties += penalty
        rb.gaming_flags.append(f"gaming: spam_clicks_repeat_{consecutive_spam_clicks}")
        rb.details.append(f"{penalty:.1f} Spam-click repeat penalty (streak={consecutive_spam_clicks})")
        if consecutive_spam_clicks >= 3:
            rb.abort_requested = True
            rb.details.append("Spam-click repeat limit reached (>=3) -> terminate rollout")

    # 5. Terminal Verdict
    if is_terminal:
        if terminal_pass:
            rb.terminal += 2.0
            rb.details.append("+2.0 Terminal PASS outcome")
        elif terminal_fail:
            rb.terminal -= 2.0
            rb.details.append("-2.0 Terminal FAIL outcome")

    rb.total = rb.milestones + rb.penalties + rb.step_cost + rb.terminal
    return rb


class ProcessRewardModel:
    """
    Process Reward Model (PRM) wrapping dense_reward() with state tracking.

    Tracks state progression, action history, consecutive spam clicks,
    and returns comprehensive step reward breakdowns.
    """

    def __init__(self, deprecated_selectors: set[str] | list[str] | None = None) -> None:
        self.deprecated_selectors: set[str] = set(deprecated_selectors or [])
        self.last_action: dict[str, Any] | None = None
        self.last_dom_hash: str | None = None
        self.consecutive_spam_clicks: int = 0
        self.total_reward: float = 0.0
        self.step_count: int = 0

    def reset(self) -> None:
        """Reset PRM state for a new episode."""
        self.last_action = None
        self.last_dom_hash = None
        self.consecutive_spam_clicks = 0
        self.total_reward = 0.0
        self.step_count = 0

    def step_reward(
        self,
        state_before: Any,
        state_after: Any,
        action: dict[str, Any] | Any,
        rules: list[CheckResult] | None = None,
        is_terminal: bool = False,
        terminal_pass: bool = False,
        terminal_fail: bool = False,
    ) -> RewardBreakdown:
        """
        Evaluate reward for a single action step transition.
        """
        self.step_count += 1

        # Check for spam-click: same click selector executed with no DOM hash change
        action_dict = action if isinstance(action, dict) else {
            "type": getattr(action, "type", ""),
            "selector": getattr(action, "selector", ""),
            "text": getattr(action, "text", ""),
        }
        act_type = str(action_dict.get("type", "")).lower()
        sel = str(action_dict.get("selector", ""))
        curr_dom_hash = getattr(state_after, "dom_hash", None) or getattr(state_after, "url", "")
        prev_dom_hash = getattr(state_before, "dom_hash", None) or getattr(state_before, "url", "")

        is_same_click = (
            self.last_action is not None
            and act_type == "click"
            and self.last_action.get("type") == "click"
            and sel == self.last_action.get("selector")
        )
        is_dom_unchanged = (
            curr_dom_hash is not None
            and prev_dom_hash is not None
            and curr_dom_hash == prev_dom_hash
        )

        if is_same_click and is_dom_unchanged:
            self.consecutive_spam_clicks += 1
        else:
            self.consecutive_spam_clicks = 0

        self.last_action = dict(action_dict)
        self.last_dom_hash = curr_dom_hash

        rb = dense_reward(
            rules=rules,
            state_before=state_before,
            state_after=state_after,
            action=action,
            is_terminal=is_terminal,
            terminal_pass=terminal_pass,
            terminal_fail=terminal_fail,
            consecutive_spam_clicks=self.consecutive_spam_clicks,
            deprecated_selectors=self.deprecated_selectors,
        )

        self.total_reward += rb.total
        return rb
