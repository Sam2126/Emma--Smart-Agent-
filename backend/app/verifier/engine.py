"""
Verification engine — orchestrates the three-tier verification system.

Tier 1: Rule-based checks (fast, cheap, trustworthy). Site-specific rules where
        they exist (rules/amazon.py), generic rules everywhere else
        (rules/generic.py).
Tier 2: Judges. A text judge reads the page state; a vision judge reads a
        screenshot of the final page. When the vision judge is confident, it
        decides, because it sees what DOM text misses (overlays, rendered
        results, canvas content).
Tier 3: Human confirmation for irreversible actions (checkout, payment) — see
        requires_human_confirmation and app/utils/confirmation.py.

The engine combines the tiers into one VerificationResult.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from app.browser.page_state import PageState
from app.verifier.rule_checks import RuleCheck, CheckResult
from app.verifier.llm_judge import LLMJudge, LLMJudgment
from app.verifier.rules.amazon import get_step_rules, get_task_rules
from app.verifier.rules.generic import get_generic_task_rules

logger = structlog.get_logger(__name__)

# A vision verdict below this confidence is treated as "couldn't tell".
VISION_DECISIVE_CONFIDENCE = 0.6
# The vision judge only runs when it can change the outcome: the rules passed
# and the text judge did not already pass the task with this confidence. Found
# 2026-09-15: rules 2/2 and the text judge at 99% had already passed a Google
# search, and the result still waited ~100 s for an overloaded vision model.
TEXT_CONFIDENT_PASS = 0.85
VISION_JUDGE_TIMEOUT_SECONDS = 40


class VerificationTier(str, Enum):
    """Which verification tier produced this result."""
    RULE = "rule"
    LLM = "llm"
    VISION = "vision"
    HUMAN = "human"


@dataclass
class VerificationResult:
    """Combined result from all verification tiers."""
    passed: bool
    tier: VerificationTier
    confidence: float  # 0.0 to 1.0
    rule_results: list[CheckResult] = field(default_factory=list)
    llm_judgment: LLMJudgment | None = None
    vision_judgment: LLMJudgment | None = None
    human_confirmed: bool | None = None  # None = not asked, True/False = answered
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging and WebSocket transmission."""
        def _judgment(j: LLMJudgment | None) -> dict[str, Any] | None:
            if not j:
                return None
            return {"passed": j.passed, "confidence": j.confidence, "reasoning": j.reasoning}

        return {
            "passed": self.passed,
            "tier": self.tier.value,
            "confidence": self.confidence,
            "summary": self.summary,
            "rule_results": [
                {
                    "check": r.check_name,
                    "passed": r.passed,
                    "expected": r.expected,
                    "actual": r.actual,
                }
                for r in self.rule_results
            ],
            "llm_judgment": _judgment(self.llm_judgment),
            "vision_judgment": _judgment(self.vision_judgment),
            "human_confirmed": self.human_confirmed,
        }


# Actions that are considered irreversible and require human confirmation
IRREVERSIBLE_ACTIONS = {
    "place_order",
    "confirm_payment",
    "checkout",
    "buy_now",
    "submit_order",
}


class VerificationEngine:
    """
    Orchestrates multi-tier verification for agent actions and tasks.

    Usage:
        engine = VerificationEngine()
        result = await engine.verify_step("add_to_cart", before_state, after_state)
        if not result.passed:
            # handle failure...

        if engine.requires_human_confirmation("place_order"):
            # show confirmation dialog...
    """

    def __init__(self, llm_judge: LLMJudge | None = None) -> None:
        self._llm_judge = llm_judge or LLMJudge()

    async def verify_step(
        self,
        step_type: str,
        before: PageState | None,
        after: PageState,
        action_description: str = "",
        use_llm_judge: bool = False,
    ) -> VerificationResult:
        """
        Verify a single action step using tiered checks.

        Tier 1 (always): Run rule-based checks for this step type.
        Tier 2 (optional): If rules are ambiguous or use_llm_judge=True,
                          get LLM second opinion.
        """
        rules = get_step_rules(step_type)
        rule_results = await self._run_rules(rules, before, after, "rule_check_failed")

        if rule_results:
            rules_passed = all(r.passed for r in rule_results)
            rules_conclusive = True
        else:
            rules_passed = True  # No rules defined for this step — inconclusive
            rules_conclusive = False

        logger.info(
            "step_rules_checked",
            step_type=step_type,
            rules_count=len(rule_results),
            rules_passed=rules_passed,
            rules_conclusive=rules_conclusive,
        )

        llm_judgment: LLMJudgment | None = None
        if use_llm_judge or not rules_conclusive:
            try:
                llm_judgment = await self._llm_judge.verify_step(action_description or step_type, before, after)
                logger.info("step_llm_judged", passed=llm_judgment.passed, confidence=llm_judgment.confidence)
            except Exception as e:
                logger.error("llm_judge_step_failed", error=str(e))

        if rules_conclusive:
            final_passed = rules_passed
            final_confidence = 1.0 if rules_passed else 0.0
            final_tier = VerificationTier.RULE
        elif llm_judgment:
            final_passed = llm_judgment.passed
            final_confidence = llm_judgment.confidence
            final_tier = VerificationTier.LLM
        else:
            final_passed = True
            final_confidence = 0.5
            final_tier = VerificationTier.RULE

        failed_rules = [r for r in rule_results if not r.passed]
        summary_parts = []
        if rule_results:
            summary_parts.append(f"Rules: {len(rule_results) - len(failed_rules)}/{len(rule_results)} passed")
        if failed_rules:
            summary_parts.append(f"Failed: {', '.join(r.check_name for r in failed_rules)}")
        if llm_judgment:
            summary_parts.append(f"LLM: {'PASS' if llm_judgment.passed else 'FAIL'} ({llm_judgment.confidence:.1%})")

        return VerificationResult(
            passed=final_passed,
            tier=final_tier,
            confidence=final_confidence,
            rule_results=rule_results,
            llm_judgment=llm_judgment,
            summary=" | ".join(summary_parts),
        )

    async def verify_task(
        self,
        task_type: str,
        task_instruction: str,
        final_state: PageState,
        action_summary: str,
        initial_state: PageState | None = None,
        screenshot_b64: str | None = None,
    ) -> VerificationResult:
        """
        Verify that an entire task was completed successfully.

        Rules: site-specific rules for known task types, otherwise generic rules
        inferred from the instruction. Rules must pass.
        Judges: the text judge always runs. When a screenshot is supplied and
        vision verification is enabled, the vision judge runs too and decides
        whenever its confidence is at least VISION_DECISIVE_CONFIDENCE.
        """
        from app.config import get_settings

        rules = get_task_rules(task_type)
        rule_source = "site"
        if not rules:
            rules = get_generic_task_rules(task_instruction, initial_state, final_state)
            rule_source = "generic"
        rule_results = await self._run_rules(rules, initial_state, final_state, "task_rule_check_failed")
        rules_passed = all(r.passed for r in rule_results) if rule_results else True

        llm_judgment: LLMJudgment | None = None
        try:
            llm_judgment = await self._llm_judge.verify_task(task_instruction, final_state, action_summary)
        except Exception as e:
            logger.error("llm_judge_task_failed", error=str(e))

        vision_judgment: LLMJudgment | None = None
        vision_note = ""
        if screenshot_b64 and get_settings().vision_verification_enabled:
            text_confident = bool(
                llm_judgment and llm_judgment.passed and llm_judgment.confidence >= TEXT_CONFIDENT_PASS
            )
            if not rules_passed:
                vision_note = "skipped (a rule already failed)"
            elif text_confident:
                vision_note = "skipped (rules and text judge agree)"
            else:
                try:
                    vision_judgment = await asyncio.wait_for(
                        self._llm_judge.verify_task_visual(task_instruction, action_summary, screenshot_b64),
                        VISION_JUDGE_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    vision_note = f"timed out after {VISION_JUDGE_TIMEOUT_SECONDS}s"
                    logger.warning("vision_judge_timed_out", seconds=VISION_JUDGE_TIMEOUT_SECONDS)
                except Exception as e:
                    logger.error("vision_judge_task_failed", error=str(e))

        if vision_judgment and vision_judgment.confidence >= VISION_DECISIVE_CONFIDENCE:
            judge_passed, judge_confidence, tier = vision_judgment.passed, vision_judgment.confidence, VerificationTier.VISION
        elif llm_judgment:
            judge_passed, judge_confidence, tier = llm_judgment.passed, llm_judgment.confidence, VerificationTier.LLM
        else:
            judge_passed, judge_confidence, tier = None, None, VerificationTier.RULE

        if judge_passed is None:
            final_passed = rules_passed
            final_confidence = 0.8 if rules_passed else 0.0
        else:
            final_passed = rules_passed and judge_passed
            final_confidence = min(1.0 if rules_passed else 0.0, judge_confidence)

        failed_rules = [r for r in rule_results if not r.passed]
        summary_parts = [f"Task: {task_type}"]
        if rule_results:
            summary_parts.append(
                f"{rule_source.capitalize()} rules: {len(rule_results) - len(failed_rules)}/{len(rule_results)} passed"
            )
        if failed_rules:
            summary_parts.append("Failed: " + ", ".join(f"{r.check_name} ({r.actual})" for r in failed_rules))
        if llm_judgment:
            summary_parts.append(f"LLM: {'PASS' if llm_judgment.passed else 'FAIL'} ({llm_judgment.confidence:.1%})")
        if vision_judgment:
            summary_parts.append(
                f"Vision: {'PASS' if vision_judgment.passed else 'FAIL'} ({vision_judgment.confidence:.1%})"
            )
        elif vision_note:
            summary_parts.append(f"Vision: {vision_note}")

        return VerificationResult(
            passed=final_passed,
            tier=tier,
            confidence=final_confidence,
            rule_results=rule_results,
            llm_judgment=llm_judgment,
            vision_judgment=vision_judgment,
            summary=" | ".join(summary_parts),
        )

    @staticmethod
    async def _run_rules(
        rules: list[RuleCheck],
        before: PageState | None,
        after: PageState,
        error_event: str,
    ) -> list[CheckResult]:
        results: list[CheckResult] = []
        for rule in rules:
            try:
                results.append(await rule.check(before, after))
            except Exception as e:
                logger.error(error_event, check=rule.name, error=str(e))
                results.append(CheckResult(check_name=rule.name, passed=False, details=f"Check raised exception: {str(e)}"))
        return results

    @staticmethod
    def requires_human_confirmation(action_type: str) -> bool:
        """
        Check if an action type requires human confirmation before execution.

        This is the safety gate for irreversible actions (checkout, payment).
        """
        return requires_human_confirmation(action_type)


def requires_human_confirmation(
    action_type: str,
    target_description: str = "",
    current_url: str = "",
) -> bool:
    """
    Defense-in-depth safety gate for irreversible actions (checkout, payment, order placement).

    Layers:
    1. Action Type Gate: explicit irreversible action names
    2. URL / Gateway Gate: intercept any interaction on active checkout/payment gateway URLs
    3. Multi-platform Pattern Gate: regex matches across e-commerce & payment terms
    4. Auto-submit / Keyboard Gate: intercept Enter keys on payment pages
    """
    action_lower = action_type.lower()
    target_lower = target_description.lower()
    url_lower = current_url.lower()

    # Layer 1: Explicit irreversible action types
    if action_lower in IRREVERSIBLE_ACTIONS:
        return True

    # Layer 2: Active Checkout / Payment Gateway URL Gate
    sensitive_url_patterns = [
        "/checkout", "/payment", "/order/place", "/pay/", "/buy/",
        "/gateway", "/cashfree", "/razorpay", "/stripe", "/billdesk",
        "/review-order", "/place-order", "/confirm-order"
    ]
    for url_pat in sensitive_url_patterns:
        if url_pat in url_lower:
            # If on a payment page, any submit, click, or Enter press requires confirmation
            if action_lower in ("click", "press_key", "submit") or "enter" in target_lower:
                return True

    # Layer 3: Multi-Platform Action & Button Pattern Gate
    sensitive_button_patterns = [
        "place order", "buy now", "order now", "proceed to checkout",
        "confirm payment", "pay now", "proceed to pay", "continue to payment",
        "complete purchase", "confirm and pay", "confirm & pay", "submit order",
        "swipe to pay", "make payment", "submitorder", "checkout-button",
        "buy-now", "btn-checkout", "pay_button", "placeyourorder"
    ]
    for p in sensitive_button_patterns:
        if p in target_lower or p in action_lower:
            return True

    return False
