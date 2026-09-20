"""
Domain Memory — episodic knowledge store for domain-specific skills and lessons.

Enables true self-improvement:
1. Recalls known selectors, strategies, and quirk workarounds for any given website.
2. Updates and refines knowledge when tasks succeed or fail.
"""

from __future__ import annotations

import json
from typing import Any, Optional
from urllib.parse import urlparse
import structlog
from sqlalchemy import select

from app.state.database import get_session
from app.state.models import DomainMemoryRecord

logger = structlog.get_logger(__name__)

# Every local task currently writes to general_tips up to three times (a
# direct call from flow.py, plus brain._update_domain_memory and
# brain._persist_reflection both calling record_domain_learning again for
# the same task). Left unbounded, this field grows forever and gets stuffed
# into the planner's prompt on every future task — verified in production:
# after ~45 tasks it had accumulated the same few reflections duplicated
# 3-4x verbatim, directly inflating every subsequent prompt's token count
# and slowing every task down. Cap entry count so old noise gets evicted
# instead of accumulating without limit.
_MAX_GENERAL_TIPS_ENTRIES = 12


def _merge_general_tips(existing: str | None, new_tip: str) -> str:
    """Append a new tip, skip if it's a near-duplicate, and cap total entries."""
    new_tip = (new_tip or "").strip()
    if not new_tip:
        return existing or ""

    entries = [e.strip() for e in (existing or "").split("\n- ") if e.strip()]

    # Skip appending if this exact tip (or the previous most-recent one) is
    # already the same text — the common case when several call sites in
    # the same task's learn_from_task all describe the same outcome.
    if entries and entries[-1] == new_tip:
        return "\n- ".join(entries)

    entries.append(new_tip)
    # Keep only the most recent N — older, likely-superseded lessons are
    # dropped rather than accumulating forever.
    entries = entries[-_MAX_GENERAL_TIPS_ENTRIES:]
    return "\n- ".join(entries)


def extract_domain(url_or_domain: str) -> str:
    """Normalize a URL or domain string to a clean domain (e.g. 'amazon.in', 'flipkart.com')."""
    text = url_or_domain.strip().lower()
    if not text.startswith("http://") and not text.startswith("https://"):
        text = f"https://{text}"
    try:
        parsed = urlparse(text)
        netloc = parsed.netloc or parsed.path
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc.split(":")[0]
    except Exception:
        return url_or_domain.strip().lower()


class DomainMemoryStore:
    """Manages reading and writing learned domain knowledge."""

    async def get_domain_memory(self, domain_or_url: str) -> dict[str, Any]:
        """
        Fetch stored episodic knowledge for a target domain.
        Returns a dictionary with known patterns or an empty default template.
        """
        domain = extract_domain(domain_or_url)
        try:
            session = await get_session()
            async with session.begin():
                stmt = select(DomainMemoryRecord).where(DomainMemoryRecord.domain == domain)
                result = await session.execute(stmt)
                record = result.scalar_one_or_none()

                if record:
                    return {
                        "domain": record.domain,
                        "site_name": record.site_name,
                        "search_selectors": json.loads(record.search_selectors_json or "[]"),
                        "product_link_selectors": json.loads(record.product_link_selectors_json or "[]"),
                        "add_to_cart_selectors": json.loads(record.add_to_cart_selectors_json or "[]"),
                        "popup_dismiss_selectors": json.loads(record.popup_dismiss_selectors_json or "[]"),
                        "general_tips": record.general_tips or "",
                        "successful_runs": record.successful_runs,
                        "failed_runs": record.failed_runs,
                    }
        except Exception as e:
            logger.warning("get_domain_memory_failed", domain=domain, error=str(e))

        return {
            "domain": domain,
            "site_name": domain.split(".")[0].capitalize(),
            "search_selectors": [],
            "product_link_selectors": [],
            "add_to_cart_selectors": [],
            "popup_dismiss_selectors": [],
            "general_tips": "No prior experience recorded for this domain yet.",
            "successful_runs": 0,
            "failed_runs": 0,
        }

    async def record_domain_learning(
        self,
        domain_or_url: str,
        site_name: str = "",
        search_selectors: list[str] | None = None,
        product_link_selectors: list[str] | None = None,
        add_to_cart_selectors: list[str] | None = None,
        popup_dismiss_selectors: list[str] | None = None,
        general_tips: str = "",
        success: bool = True,
        count_run: bool = True,
    ) -> None:
        """
        Save or update learned patterns for a domain.
        Merges new selectors and appends lessons learned.

        count_run=False for a write that only adds knowledge about a run that
        was already counted (a reflection, a tip), so each run counts once.
        """
        domain = extract_domain(domain_or_url)
        try:
            session = await get_session()
            async with session.begin():
                stmt = select(DomainMemoryRecord).where(DomainMemoryRecord.domain == domain)
                result = await session.execute(stmt)
                record = result.scalar_one_or_none()

                if not record:
                    record = DomainMemoryRecord(
                        domain=domain,
                        site_name=site_name or domain.split(".")[0].capitalize(),
                        search_selectors_json=json.dumps(search_selectors or []),
                        product_link_selectors_json=json.dumps(product_link_selectors or []),
                        add_to_cart_selectors_json=json.dumps(add_to_cart_selectors or []),
                        popup_dismiss_selectors_json=json.dumps(popup_dismiss_selectors or []),
                        general_tips=general_tips,
                        successful_runs=int(count_run and success),
                        failed_runs=int(count_run and not success),
                    )
                    session.add(record)
                else:
                    # Merge selectors
                    def _merge_lists(old_json: Optional[str], new_list: Optional[list[str]]) -> str:
                        existing = set(json.loads(old_json or "[]"))
                        if new_list:
                            existing.update(new_list)
                        return json.dumps(list(existing))

                    record.search_selectors_json = _merge_lists(record.search_selectors_json, search_selectors)
                    record.product_link_selectors_json = _merge_lists(record.product_link_selectors_json, product_link_selectors)
                    record.add_to_cart_selectors_json = _merge_lists(record.add_to_cart_selectors_json, add_to_cart_selectors)
                    record.popup_dismiss_selectors_json = _merge_lists(record.popup_dismiss_selectors_json, popup_dismiss_selectors)

                    if general_tips:
                        record.general_tips = _merge_general_tips(record.general_tips, general_tips)

                    if count_run and success:
                        record.successful_runs += 1
                    elif count_run:
                        record.failed_runs += 1

            logger.info("domain_learning_recorded", domain=domain, success=success)
        except Exception as e:
            logger.error("record_domain_learning_failed", domain=domain, error=str(e))

    async def deprecate_failed_selector(
        self, domain_or_url: str, failed_selector: str, threshold: int = 3
    ) -> None:
        """
        Anti-churn memory decay: requires 3 consecutive failures before permanently
        pruning a selector from domain memory. Prevents temporary network blips or
        A/B tests from prematurely destroying good selectors.
        """
        if not failed_selector:
            return
        domain = extract_domain(domain_or_url)
        try:
            session = await get_session()
            async with session.begin():
                stmt = select(DomainMemoryRecord).where(DomainMemoryRecord.domain == domain)
                result = await session.execute(stmt)
                record = result.scalar_one_or_none()

                if record:
                    # Increment failed runs counter
                    record.failed_runs += 1

                    # Check failure streak in general_tips / metadata
                    prune_now = True if threshold <= 1 else (record.failed_runs >= threshold)

                    if prune_now:
                        def _remove_item(json_str: Optional[str]) -> str:
                            items = json.loads(json_str or "[]")
                            # Exact match only. Removing every selector that merely
                            # CONTAINED the failed one let a failed "button" wipe
                            # out every stored 'button:has-text(...)' selector.
                            filtered = [i for i in items if i != failed_selector]
                            return json.dumps(filtered)

                        record.search_selectors_json = _remove_item(record.search_selectors_json)
                        record.product_link_selectors_json = _remove_item(record.product_link_selectors_json)
                        record.add_to_cart_selectors_json = _remove_item(record.add_to_cart_selectors_json)
                        record.popup_dismiss_selectors_json = _remove_item(record.popup_dismiss_selectors_json)
                        logger.info("selector_pruned_after_threshold", domain=domain, selector=failed_selector, strikes=record.failed_runs)
                    else:
                        logger.info("selector_failure_strike_incremented", domain=domain, selector=failed_selector, strikes=record.failed_runs, threshold=threshold)

        except Exception as e:
            logger.warning("selector_deprecation_failed", domain=domain, error=str(e))

