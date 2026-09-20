"""
Tests for bounded, deduplicated general_tips growth in domain memory.

Every local task previously wrote to general_tips up to three times (direct
call in flow.py, plus brain._update_domain_memory and brain._persist_reflection
both calling record_domain_learning again for the same task), with no cap
and no deduplication. Verified in production: after ~45 tasks the field had
accumulated the same few reflections duplicated 3-4x verbatim, which was
being stuffed into the planner's prompt on every subsequent task — directly
inflating token counts and slowing every task down. These tests lock in the
fix: a hard cap on entry count, and skipping an immediate exact-duplicate.
"""

from __future__ import annotations

import uuid

import pytest

from app.state.domain_memory import (
    DomainMemoryStore,
    _merge_general_tips,
    _MAX_GENERAL_TIPS_ENTRIES,
)


def test_merge_general_tips_caps_entry_count():
    existing = ""
    for i in range(_MAX_GENERAL_TIPS_ENTRIES + 10):
        existing = _merge_general_tips(existing, f"tip number {i}")

    entries = [e for e in existing.split("\n- ") if e]
    assert len(entries) == _MAX_GENERAL_TIPS_ENTRIES
    # The oldest entries must have been evicted, newest kept.
    assert "tip number 0" not in existing
    assert f"tip number {_MAX_GENERAL_TIPS_ENTRIES + 9}" in existing


def test_merge_general_tips_skips_immediate_duplicate():
    existing = _merge_general_tips("", "same tip")
    existing = _merge_general_tips(existing, "same tip")
    entries = [e for e in existing.split("\n- ") if e]
    assert entries == ["same tip"]


def test_merge_general_tips_keeps_distinct_consecutive_entries():
    existing = _merge_general_tips("", "first tip")
    existing = _merge_general_tips(existing, "second tip")
    entries = [e for e in existing.split("\n- ") if e]
    assert entries == ["first tip", "second tip"]


@pytest.mark.asyncio
async def test_record_domain_learning_does_not_grow_unbounded():
    """End-to-end: simulate what actually happened in production — one
    'task' writing general_tips 3 times, repeated many times over — and
    confirm the stored field stays bounded instead of growing forever."""
    domain = f"tips-cap-test-{uuid.uuid4().hex[:8]}"
    store = DomainMemoryStore()

    for i in range(20):
        await store.record_domain_learning(domain, general_tips=f"TASK: do thing {i} -> DONE.")
        await store.record_domain_learning(domain, general_tips=f"report for thing {i}")
        await store.record_domain_learning(domain, general_tips=f"[REFLEXION] lesson {i}")

    mem = await store.get_domain_memory(domain)
    tips = mem["general_tips"]
    entries = [e for e in tips.split("\n- ") if e]
    assert len(entries) <= _MAX_GENERAL_TIPS_ENTRIES
    # Must retain the most recent content, not the oldest.
    assert "lesson 19" in tips
    assert "lesson 0" not in tips
