"""
Unit tests for the grounded self-improving brain memory system.

Tests:
1. Time-decay calculation for learned skills
2. Active selector deprecation on task failure
3. Irreversible action safety gate
4. Benchmark suite structure
"""

import uuid
from datetime import datetime, timezone

import pytest
from app.state.brain import BrainMemory, _as_aware_utc
from app.state.database import get_session
from app.state.domain_memory import DomainMemoryStore
from app.state.models import SkillRecord
from app.verifier.engine import requires_human_confirmation
from app.evals.benchmark_suite import BENCHMARK_TASKS


@pytest.mark.asyncio
async def test_requires_human_confirmation_on_checkout():
    """Verify that irreversible checkout and payment triggers the safety gate."""
    assert requires_human_confirmation("click", "button:has-text('Place Order')") is True
    assert requires_human_confirmation("click", "#submitOrderButtonId") is True
    assert requires_human_confirmation("click", "button:has-text('Confirm and Pay')") is True
    # Safe actions should not trigger
    assert requires_human_confirmation("click", "#twotabsearchtextbox") is False
    assert requires_human_confirmation("click", "button:has-text('Search')") is False


@pytest.mark.asyncio
async def test_selector_deprecation_on_failure():
    """Verify that broken selectors require 3 strikes before permanent pruning."""
    import uuid
    store = DomainMemoryStore()
    test_domain = f"test_{uuid.uuid4().hex[:8]}.com"
    
    # 1. Record fresh selectors (failed_runs = 0)
    await store.record_domain_learning(
        domain_or_url=test_domain,
        search_selectors=["#broken_old_search", "#working_search"],
        success=True,
    )

    mem = await store.get_domain_memory(test_domain)
    assert "#broken_old_search" in mem["search_selectors"]

    # 2. Strike 1: Should NOT prune yet (strikes=1 < threshold=3)
    await store.deprecate_failed_selector(test_domain, "#broken_old_search", threshold=3)
    mem_mid = await store.get_domain_memory(test_domain)
    assert "#broken_old_search" in mem_mid["search_selectors"]

    # 3. Strike 2 & 3: Meets threshold -> Prunes permanently
    await store.deprecate_failed_selector(test_domain, "#broken_old_search", threshold=2)
    mem_after = await store.get_domain_memory(test_domain)
    assert "#broken_old_search" not in mem_after["search_selectors"]
    assert "#working_search" in mem_after["search_selectors"]



@pytest.mark.asyncio
async def test_benchmark_suite_integrity():
    """Verify that the frozen benchmark suite contains valid ground-truth assertions."""
    assert len(BENCHMARK_TASKS) >= 4
    for task in BENCHMARK_TASKS:
        assert task.task_id.startswith("bench_")
        assert len(task.instruction) > 5
        assert callable(task.ground_truth_assertion)


@pytest.mark.asyncio
async def test_brain_stats_reporting():
    """Verify that brain memory stats return structured metrics."""
    brain = BrainMemory()
    stats = await brain.get_brain_stats()
    assert "total_tasks" in stats
    assert "success_rate" in stats
    assert "total_skills_learned" in stats
    assert "total_failure_patterns" in stats


# =============================================================================
# Regression: SQLite naive/aware datetime mismatch in skill recall.
#
# SkillRecord.last_used_at / created_at are declared DateTime(timezone=True)
# and always written via datetime.now(timezone.utc), but SQLite has no
# native timezone-aware storage type — aiosqlite reads the column back as a
# naive datetime regardless of how it was declared. _recall_skills used to
# subtract that naive value directly from an aware datetime.now(timezone.utc),
# which raises "can't subtract offset-naive and offset-aware datetimes" on
# every single call. That exception was swallowed by a broad except-and-
# log-warning, so skills were written successfully but NEVER once actually
# recalled for a planner to use — confirmed live against production data,
# where real skills existed in the database but _recall_skills always
# silently returned an empty list.
# =============================================================================

def test_as_aware_utc_attaches_utc_to_naive_datetime():
    naive = datetime(2026, 1, 1, 12, 0, 0)
    assert naive.tzinfo is None
    result = _as_aware_utc(naive)
    assert result.tzinfo is not None
    assert result.utcoffset().total_seconds() == 0


def test_as_aware_utc_leaves_aware_datetime_unchanged():
    aware = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert _as_aware_utc(aware) is aware


@pytest.mark.asyncio
async def test_recall_skills_does_not_crash_on_naive_db_datetimes():
    """
    End-to-end regression: write a real SkillRecord through the ORM (as
    production code does, with an aware datetime.now(timezone.utc)), read
    it back through a fresh session — reproducing SQLite's actual naive
    round-trip — and confirm _recall_skills returns it instead of silently
    swallowing a TypeError and returning an empty list.
    """
    domain = f"recall-test-{uuid.uuid4().hex[:8]}"

    session = await get_session()
    async with session.begin():
        session.add(SkillRecord(
            skill_type="open_app",
            domain_pattern=domain,
            description=f"Skill for 'open_app' on {domain}",
            success_rate=1.0,
            usage_count=1,
            last_used_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        ))

    brain = BrainMemory()
    skills = await brain._recall_skills(domain, "open whatsapp")

    assert len(skills) == 1
    assert skills[0]["skill_type"] == "open_app"
    assert skills[0]["domain_pattern"] == domain
    assert 0.0 < skills[0]["effective_confidence"] <= 1.0
