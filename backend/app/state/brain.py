"""
Brain Memory — the unified self-improving intelligence layer.

This is the heart of the self-improvement system. It ties together:
- DomainMemoryRecord (per-site learned selectors & quirks)
- SkillRecord (generalized reusable action patterns)
- FailurePatternRecord (error→solution mappings)
- BrainStateRecord (global evolution metrics)

Pre-flight: `recall_for_task()` assembles all relevant context for the planner.
Post-flight: `learn_from_task()` extracts and persists new knowledge.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Optional

from dataclasses import dataclass
import structlog
from sqlalchemy import select, func, or_

from app.config import get_settings
from app.state.database import get_session
from app.state.models import (
    SkillRecord,
    FailurePatternRecord,
    BrainStateRecord,
    DomainMemoryRecord,
    TaskRecord,
)
from app.state.domain_memory import DomainMemoryStore, extract_domain

logger = structlog.get_logger(__name__)


@dataclass
class Reflection:
    """Structured reflection on a task execution for procedural self-improvement."""
    what_succeeded: str
    what_failed: str
    root_cause: str
    what_to_try_next: str
    confidence: float = 0.9


def _as_aware_utc(dt: datetime) -> datetime:
    """
    Ensure a datetime is timezone-aware (UTC), assuming naive values are
    already UTC — which is what this app always intends to store.

    SQLAlchemy's DateTime(timezone=True) columns are declared aware, and
    the app always writes them via datetime.now(timezone.utc), but SQLite
    has no native timezone-aware storage type: it round-trips the value as
    plain text and aiosqlite reads it back naive regardless of how the
    column was declared. That mismatch broke _recall_skills in production —
    every call raised "can't subtract offset-naive and offset-aware
    datetimes", so structured skills were written successfully but never
    once actually recalled for a planner to use.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# Which structured skill a local/desktop tool represents, highest priority
# first. Priority matters: a run that used open_app AND send_keys was a
# message/interaction task, not an app launch. The old lookup iterated a set,
# so the chosen type for such a run was effectively random.
LOCAL_SKILL_PRIORITY: list[tuple[str, str]] = [
    # The defining action of the run when it happened (added 2026-09-17).
    ("install_app", "install_app"),
    ("add_to_cart", "add_to_cart"),
    ("download_file", "file_download"),
    ("send_email", "email_send"),
    ("create_email_draft", "email_draft"),
    ("read_email", "email_read"),
    ("search_emails", "email_read"),
    ("find_email_address", "email_read"),
    ("send_keys", "app_interaction"),
    ("click_window", "app_interaction"),
    ("write_file", "file_write"),
    ("run_command", "run_command"),
    ("open_file_or_folder", "open_file"),
    ("read_file", "file_read"),
    ("open_app", "open_app"),
    ("find_files", "file_search"),
    ("list_recent_files", "file_search"),
    ("list_folder", "file_search"),
]


def local_skill_type_for(tools) -> str | None:
    """Highest-priority skill type among the tools a run used, or None."""
    used = set(tools or [])
    return next((skill for tool, skill in LOCAL_SKILL_PRIORITY if tool in used), None)


def _skill_type_from_instruction(instruction: str) -> str:
    """Skill type guessed from the wording, for runs whose tools name none."""
    text = (instruction or "").lower()
    if "code" in text or "program" in text or "run" in text or "compile" in text:
        return "execute_code"
    if "search" in text:
        return "search_product"
    if "add to cart" in text or "add_to_cart" in text:
        return "add_to_cart"
    if "open" in text or "view" in text:
        return "view_content"
    if "submit" in text or "fill" in text:
        return "form_submission"
    return "general_browse"


def skill_type_for_run(tools, instruction: str) -> str:
    """The skill a run exercised: from its tools first, else from its wording.

    A success and a failure of the same kind of run must name the same skill.
    Found 2026-09-16: a success was filed under its tools ("file_write") while a
    failure was guessed from the wording ("search" -> "search_product"), so a
    failure penalised a skill the run never used and the real one kept its score.
    """
    return local_skill_type_for(tools) or _skill_type_from_instruction(instruction)


# Steps whose "selector" is a real page selector. A local step's is a path, an
# app name or a command.
_PAGE_SELECTOR_TOOLS = {"click_element", "type_into_element", "select_dropdown_option"}
# Current tool names (plus the old engine's, still present in older history).
_TYPE_TOOLS = {"type_into_element", "type", "type_search_query"}
_CLICK_TOOLS = {"click_element", "click", "add_to_cart"}
# How many unstored runs are remembered for late 👍/👎.
_UNSTORED_RUNS_KEPT = 50


def _normalize_error(error: str) -> str:
    """Normalize an error string into a stable signature for pattern matching."""
    # Strip dynamic values: IDs, numbers, timestamps, URLs
    sig = re.sub(r"\b\d{4,}\b", "<NUM>", error)
    sig = re.sub(r"https?://[^\s]+", "<URL>", sig)
    sig = re.sub(r"['\"][^'\"]{40,}['\"]", "<LONG_STR>", sig)
    sig = re.sub(r"\s+", " ", sig).strip()
    return sig[:300]


class BrainMemory:
    """
    Unified brain interface for self-improving agent memory.

    Usage:
        brain = BrainMemory()

        # Before task execution — assemble context for the planner
        context = await brain.recall_for_task("search noise smartwatch on flipkart")

        # After task execution — extract and persist learnings
        await brain.learn_from_task(
            task_id="abc-123",
            instruction="search noise smartwatch on flipkart",
            domain="flipkart.com",
            trajectory=[...],  # list of action records
            success=True,
            error=None,
            duration_seconds=45.2,
            raw_result="Successfully searched..."
        )
    """

    def __init__(self, semantic=None, reflection=None) -> None:
        self._domain_store = DomainMemoryStore()
        # Level 3 components. Injected in tests; otherwise the process-wide
        # singletons are created lazily on first use.
        self._semantic = semantic
        self._reflection = reflection
        # Finished runs that were deliberately not stored (no tool ran), kept
        # for a while so the user's 👍 / 👎 on them can still be learned from.
        self._unstored_runs: dict[str, dict[str, Any]] = {}

    # =========================================================================
    # PRE-FLIGHT: Recall context for task planning
    # =========================================================================

    async def recall_for_task(
        self, instruction: str, domain: str | None = None
    ) -> str:
        """
        Assemble all relevant brain context for a task.

        Returns a formatted string that should be injected into the planner's
        task description so the agent benefits from prior experience.
        """
        # Extract domain from instruction if not provided
        if not domain:
            domain = self._extract_domain_from_instruction(instruction)

        parts: list[str] = []

        # 1. Domain memory (site-specific selectors & quirks)
        # IMPORTANT: Skip if domain is 'general' — the shared 'general' bucket
        # mixes memories from many different sites and causes irrelevant suggestions.
        if domain and domain not in ("general", "", "unknown"):
            domain_mem = await self._domain_store.get_domain_memory(domain)
            if domain_mem.get("successful_runs", 0) > 0 or domain_mem.get("failed_runs", 0) > 0:
                parts.append("=== DOMAIN MEMORY (learned from past runs) ===")
                parts.append(f"Domain: {domain_mem['domain']}")
                parts.append(f"Experience: {domain_mem['successful_runs']} successes, {domain_mem['failed_runs']} failures")
                if domain_mem.get("search_selectors"):
                    parts.append(f"Known search selectors: {domain_mem['search_selectors']}")
                if domain_mem.get("popup_dismiss_selectors"):
                    parts.append(f"Known popup dismiss: {domain_mem['popup_dismiss_selectors']}")
                if domain_mem.get("add_to_cart_selectors"):
                    parts.append(f"Known add-to-cart: {domain_mem['add_to_cart_selectors']}")
                if domain_mem.get("general_tips"):
                    tips = domain_mem["general_tips"]
                    parts.append(f"Tips: {tips[:800]}")
        elif domain in ("", "unknown", "general"):
            parts.append("No specific domain detected — using general strategy without domain memory.")

        # 2. Relevant skills
        skills = await self._recall_skills(domain, instruction)
        if skills:
            parts.append("\n=== LEARNED SKILLS (reusable patterns) ===")
            for s in skills[:5]:  # Top 5 most relevant
                parts.append(
                    f"- [{s['skill_type']}] {s['description'][:100]} "
                    f"(success_rate={s['success_rate']:.0%}, used={s['usage_count']}x)"
                )
                if s.get("selector_chain"):
                    parts.append(f"  Selectors: {s['selector_chain'][:200]}")

        # 3. Known failure patterns for this domain
        failures = await self._recall_failure_patterns(domain)
        if failures:
            parts.append("\n=== KNOWN FAILURE PATTERNS (avoid these) ===")
            for f in failures[:3]:
                parts.append(
                    f"- ⚠ {f['root_cause'][:100]} → Solution: {f['solution_strategy'][:150]}"
                )

        # 3.5. Reflexion (lessons from previous runs) — Rung 1
        settings = get_settings()
        if getattr(settings, "reflexion_enabled", True) and domain and domain not in ("general", "", "unknown"):
            reflections = await self._recall_reflections(domain)
            if reflections:
                parts.append("\n=== REFLEXION (lessons from previous runs) ===")
                for r in reflections[:4]:
                    parts.append(f"- {r}")

        # 4. Brain stats summary
        stats = await self.get_brain_stats()
        if stats.get("total_tasks", 0) > 0:
            rate = stats.get("success_rate", 0)
            parts.append(
                f"\n=== BRAIN STATUS: {stats['total_tasks']} tasks completed, "
                f"{rate:.0%} success rate, {stats['total_skills_learned']} skills learned ==="
            )

        if not parts:
            return "No prior experience recorded yet. This is the brain's first run on this domain."

        return "\n".join(parts)

    async def _recall_skills(
        self, domain: str, instruction: str
    ) -> list[dict[str, Any]]:
        """Recall relevant skills for a domain with time-decay confidence filtering."""
        try:
            session = await get_session()
            async with session.begin():
                stmt = (
                    select(SkillRecord)
                    .where(
                        or_(
                            SkillRecord.domain_pattern == domain,
                            SkillRecord.domain_pattern == "*",
                        )
                    )
                    .order_by(
                        SkillRecord.success_rate.desc(),
                        SkillRecord.usage_count.desc(),
                    )
                    .limit(10)
                )
                result = await session.execute(stmt)
                records = result.scalars().all()

                valid_skills = []
                now = datetime.now(timezone.utc)

                for r in records:
                    # Time decay: 2% confidence decay per 7 days of inactivity
                    last_active = _as_aware_utc(r.last_used_at or r.created_at)
                    days_old = (now - last_active).total_seconds() / 86400.0
                    decay_factor = 0.98 ** (days_old / 7.0)
                    effective_confidence = r.success_rate * decay_factor

                    # Discard stale or deprecated skills
                    if effective_confidence < 0.5:
                        continue

                    valid_skills.append({
                        "skill_type": r.skill_type,
                        "domain_pattern": r.domain_pattern,
                        "description": r.description,
                        "selector_chain": r.selector_chain_json,
                        "preconditions": r.preconditions,
                        "success_rate": r.success_rate,
                        "effective_confidence": effective_confidence,
                        "usage_count": r.usage_count,
                    })

                return valid_skills
        except Exception as e:
            logger.warning("recall_skills_failed", error=str(e))
            return []

    async def _recall_failure_patterns(
        self, domain: str
    ) -> list[dict[str, Any]]:
        """Recall known failure patterns for a domain."""
        try:
            session = await get_session()
            async with session.begin():
                stmt = (
                    select(FailurePatternRecord)
                    .where(
                        or_(
                            FailurePatternRecord.domain == domain,
                            FailurePatternRecord.domain == "*",
                        )
                    )
                    .order_by(FailurePatternRecord.occurrence_count.desc())
                    .limit(5)
                )
                result = await session.execute(stmt)
                records = result.scalars().all()

                return [
                    {
                        "error_signature": r.error_signature,
                        "root_cause": r.root_cause,
                        "solution_strategy": r.solution_strategy,
                        "occurrence_count": r.occurrence_count,
                    }
                    for r in records
                ]
        except Exception as e:
            logger.warning("recall_failure_patterns_failed", error=str(e))
            return []

    # =========================================================================
    # POST-FLIGHT: Learn from completed tasks
    # =========================================================================

    async def learn_from_task(
        self,
        task_id: str,
        instruction: str,
        domain: str,
        trajectory: list[dict[str, Any]],
        success: bool,
        error: str | None = None,
        duration_seconds: float = 0.0,
        raw_result: str = "",
        persist_template_reflection: bool = True,
    ) -> None:
        """
        Extract and persist learnings from a completed task.

        The agent engine passes persist_template_reflection=False: it runs
        learn_semantic right after, which writes an LLM-generated reflection
        (falling back to the template only if the LLM is unavailable), so the
        template lesson would otherwise be stored twice.

        Called after every task (success or failure) to continuously improve.
        """
        logger.info(
            "brain_learning_start",
            task_id=task_id,
            domain=domain,
            success=success,
        )

        try:
            # 1. Update domain memory with any new selectors found
            await self._update_domain_memory(domain, trajectory, success, raw_result)

            # 2. If successful, extract reusable skills
            if success and trajectory:
                await self._extract_skills(domain, instruction, trajectory)
            elif not success:
                # Active decay: if task failed, deprecate failed selectors and penalize skill confidence
                await self._degrade_skills_and_selectors(domain, instruction, trajectory, error)

            # 3. If failed, record the failure pattern
            if not success and error:
                await self._record_failure(domain, error, raw_result)

            # 3.5. Reflexion (Rung 1): Generate structured reflection and persist
            settings = get_settings()
            if persist_template_reflection and getattr(settings, "reflexion_enabled", True) and (trajectory or error or raw_result):
                reflection = self._generate_reflection(
                    task=instruction,
                    trajectory=trajectory,
                    success=success,
                    error=error,
                    raw_result=raw_result,
                )
                await self._persist_reflection(domain, reflection, success)

            # 4. Update global brain stats
            await self._update_brain_stats(success, duration_seconds)

            logger.info(
                "brain_learning_complete",
                task_id=task_id,
                domain=domain,
                success=success,
            )

        except Exception as e:
            logger.error("brain_learning_failed", task_id=task_id, error=str(e))

    async def _update_domain_memory(
        self,
        domain: str,
        trajectory: list[dict[str, Any]],
        success: bool,
        raw_result: str,
    ) -> None:
        """Extract selectors from trajectory and update domain memory.

        This is the one write that counts the run for the domain.
        """
        if domain == "local":
            # A local run has no page selectors, and the learner already wrote a
            # fuller tip for it (with its report); here the run is only counted.
            await self._domain_store.record_domain_learning(domain_or_url=domain, success=success)
            return

        search_selectors: list[str] = []
        cart_selectors: list[str] = []
        popup_selectors: list[str] = []

        for action in trajectory:
            if not isinstance(action, dict):
                continue
            if not action.get("success", False):
                continue

            selector = action.get("selector_used") or ""
            action_type = action.get("action_type", "")
            lowered = selector.lower()

            if not selector:
                continue

            # The current engine's tool names. Found 2026-09-16: these checks
            # still expected the old engine's "type" / "add_to_cart" steps, so no
            # page selector had been learned from a run since the engine changed.
            if action_type in _TYPE_TOOLS and ("search" in lowered or (action.get("args") or {}).get("press_enter")):
                search_selectors.append(selector)
            elif action_type in _CLICK_TOOLS and ("cart" in lowered or "buy" in lowered):
                cart_selectors.append(selector)
            elif action_type in _CLICK_TOOLS and ("close" in lowered or "dismiss" in lowered or "✕" in selector):
                popup_selectors.append(selector)

        # Also extract from raw_result text (CrewAI outputs selector info)
        for pattern in [r"input\[name=\"q\"\]", r"input\[type=\"search\"\]", r"#twotabsearchtextbox",
                        r"input\[placeholder\*=\"Search\"", r"input\[name=\"field-keywords\"\]"]:
            if pattern.replace("\\", "") in raw_result:
                search_selectors.append(pattern.replace("\\", ""))

        # Extract popup dismiss selectors from result text
        if "✕" in raw_result or "close" in raw_result.lower():
            for match in re.findall(r'button:has-text\("[^"]+"\)', raw_result):
                popup_selectors.append(match)

        await self._domain_store.record_domain_learning(
            domain_or_url=domain,
            search_selectors=list(set(search_selectors)) if search_selectors else None,
            add_to_cart_selectors=list(set(cart_selectors)) if cart_selectors else None,
            popup_dismiss_selectors=list(set(popup_selectors)) if popup_selectors else None,
            general_tips=raw_result[:300] if raw_result else "",
            success=success,
        )

    async def _extract_skills(
        self,
        domain: str,
        instruction: str,
        trajectory: list[dict[str, Any]],
    ) -> None:
        """Extract reusable skills from a successful trajectory."""
        # Group actions into logical skill sequences
        successful_actions = [
            a for a in trajectory
            if isinstance(a, dict) and a.get("success", False)
        ]

        if not successful_actions:
            return

        # Determine skill type. Prefer the actual tool names used — ground
        # truth from the trajectory — over guessing from the instruction's
        # wording. This is what makes local-device skills (open_app,
        # run_command, file operations) classify correctly: those tools
        # never existed in the instruction-keyword heuristic below, which
        # was written only with browser/e-commerce phrasing in mind.
        action_types_used = {a.get("action_type", "") for a in successful_actions}
        local_skill_types = {
            "open_app": "open_app",
            "run_command": "run_command",
            "write_file": "file_write",
            "read_file": "file_read",
            "find_files": "file_search",
            "list_recent_files": "file_search",
            "list_folder": "file_search",
            "open_file_or_folder": "open_file",
        }
        # Browser tasks, whose tools have no 1:1 skill type, fall back to the
        # instruction's wording.
        skill_type = skill_type_for_run(action_types_used, instruction)

        # Build selector chain from the trajectory (a CSS selector for
        # browser tools, or the resolved path/app/command/query for local
        # tools — see _summarize_tool_args in crew/flow.py).
        selector_chain = [
            {
                "action": a.get("action_type", ""),
                "selector": a.get("selector_used", ""),
                "input": str(a.get("input_value", ""))[:50],
            }
            for a in successful_actions
            if a.get("selector_used")
        ]

        description = f"Skill for '{skill_type}' on {domain}: {instruction[:80]}"
        if domain == "local":
            preconditions = "Local Computer Operator has file/app/command access"
            postconditions = f"Successfully completed '{skill_type}' on the local device"
        else:
            preconditions = f"On {domain} homepage or search page"
            postconditions = f"Successfully completed {skill_type}"

        try:
            session = await get_session()
            async with session.begin():
                # Check if a similar skill already exists
                stmt = select(SkillRecord).where(
                    SkillRecord.skill_type == skill_type,
                    SkillRecord.domain_pattern == domain,
                )
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()

                if existing:
                    # Update existing skill
                    existing.usage_count += 1
                    existing.success_rate = (
                        (existing.success_rate * (existing.usage_count - 1) + 1.0)
                        / existing.usage_count
                    )
                    existing.selector_chain_json = json.dumps(selector_chain)
                    existing.last_used_at = datetime.now(timezone.utc)
                    existing.description = description
                else:
                    # Create new skill
                    skill = SkillRecord(
                        skill_type=skill_type,
                        domain_pattern=domain,
                        description=description,
                        selector_chain_json=json.dumps(selector_chain),
                        preconditions=preconditions,
                        postconditions=postconditions,
                        success_rate=1.0,
                        usage_count=1,
                    )
                    session.add(skill)

            logger.info("skill_extracted", skill_type=skill_type, domain=domain)
        except Exception as e:
            logger.warning("skill_extraction_failed", error=str(e))

    async def _degrade_skills_and_selectors(
        self,
        domain: str,
        instruction: str,
        trajectory: list[dict[str, Any]],
        error: str | None = None,
    ) -> None:
        """Active memory decay: penalize skills and deprecate failed selectors on task failure."""
        try:
            # 1. Deprecate specific failed page selectors from the trajectory. A
            #    local step's "selector" is a path or app name, and each one used
            #    to count as one more failed run of the domain.
            for action in trajectory:
                if (
                    isinstance(action, dict)
                    and not action.get("success", True)
                    and action.get("action_type") in _PAGE_SELECTOR_TOOLS
                ):
                    failed_sel = action.get("selector_used", "")
                    if failed_sel:
                        await self._domain_store.deprecate_failed_selector(domain, failed_sel)

            # 2. Penalize the skill this run exercised, named as a success would name it
            tools_tried = {a.get("action_type", "") for a in trajectory if isinstance(a, dict)}
            skill_type = skill_type_for_run(tools_tried, instruction)

            session = await get_session()
            async with session.begin():
                stmt = select(SkillRecord).where(
                    SkillRecord.skill_type == skill_type,
                    SkillRecord.domain_pattern == domain,
                )
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()

                if existing:
                    existing.usage_count += 1
                    # Penalize success rate: count failure
                    existing.success_rate = (
                        (existing.success_rate * (existing.usage_count - 1))
                        / existing.usage_count
                    )
                    existing.last_used_at = datetime.now(timezone.utc)
                    logger.info(
                        "skill_penalized_on_failure",
                        skill_type=skill_type,
                        domain=domain,
                        new_success_rate=f"{existing.success_rate:.1%}",
                    )

        except Exception as e:
            logger.warning("degrade_skills_failed", error=str(e))

    async def _record_failure(
        self, domain: str, error: str, raw_result: str
    ) -> None:
        """Record a failure pattern for future avoidance."""
        error_sig = _normalize_error(error)

        # Extract root cause heuristically
        root_cause = error[:200]
        if "429" in error or "rate_limit" in error:
            root_cause = "LLM API rate limit exceeded"
            solution = "Add retry with exponential backoff; reduce prompt token count"
        elif "timeout" in error.lower():
            root_cause = "Browser action timed out — element not found or page too slow"
            solution = "Try alternative selectors; scroll page first; increase timeout"
        elif "captcha" in error.lower():
            root_cause = "CAPTCHA challenge triggered by automation"
            solution = "Add longer delays between actions; use less automation-detectable patterns"
        elif "closed" in error.lower():
            root_cause = "Browser page or context was unexpectedly closed"
            solution = "Add auto-reconnect; check page health before each action"
        else:
            solution = f"Investigate error: {error[:150]}"

        try:
            session = await get_session()
            async with session.begin():
                stmt = select(FailurePatternRecord).where(
                    FailurePatternRecord.error_signature == error_sig,
                    FailurePatternRecord.domain == domain,
                )
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()

                if existing:
                    existing.occurrence_count += 1
                    existing.last_seen_at = datetime.now(timezone.utc)
                    if len(solution) > len(existing.solution_strategy):
                        existing.solution_strategy = solution
                else:
                    pattern = FailurePatternRecord(
                        error_signature=error_sig,
                        domain=domain,
                        root_cause=root_cause,
                        solution_strategy=solution,
                        occurrence_count=1,
                    )
                    session.add(pattern)

            logger.info("failure_pattern_recorded", domain=domain, sig=error_sig[:80])
        except Exception as e:
            logger.warning("failure_recording_failed", error=str(e))

    async def _update_brain_stats(
        self, success: bool, duration_seconds: float
    ) -> None:
        """Update the global brain state singleton."""
        try:
            session = await get_session()
            async with session.begin():
                stmt = select(BrainStateRecord).where(BrainStateRecord.id == 1)
                result = await session.execute(stmt)
                brain = result.scalar_one_or_none()

                if not brain:
                    brain = BrainStateRecord(
                        id=1,
                        total_tasks=1,
                        successful_tasks=1 if success else 0,
                        failed_tasks=0 if success else 1,
                        avg_duration_seconds=duration_seconds,
                        last_learning_at=datetime.now(timezone.utc),
                    )
                    session.add(brain)
                else:
                    brain.total_tasks += 1
                    if success:
                        brain.successful_tasks += 1
                    else:
                        brain.failed_tasks += 1
                    # Running average
                    if duration_seconds > 0:
                        brain.avg_duration_seconds = (
                            (brain.avg_duration_seconds * (brain.total_tasks - 1) + duration_seconds)
                            / brain.total_tasks
                        )
                    brain.last_learning_at = datetime.now(timezone.utc)

                # Count skills and failure patterns
                skills_count = await session.execute(
                    select(func.count()).select_from(SkillRecord)
                )
                brain.total_skills_learned = skills_count.scalar() or 0

                fp_count = await session.execute(
                    select(func.count()).select_from(FailurePatternRecord)
                )
                brain.total_failure_patterns = fp_count.scalar() or 0

                # Determine mastered domains (>= 3 successes, >= 75% success rate)
                domain_stmt = select(
                    DomainMemoryRecord.domain,
                    DomainMemoryRecord.successful_runs,
                    DomainMemoryRecord.failed_runs,
                ).where(DomainMemoryRecord.successful_runs >= 3)
                domain_result = await session.execute(domain_stmt)
                mastered = []
                for row in domain_result:
                    total = row.successful_runs + row.failed_runs
                    if total > 0 and (row.successful_runs / total) >= 0.75:
                        mastered.append(row.domain)
                brain.domains_mastered_json = json.dumps(mastered)

        except Exception as e:
            logger.warning("brain_stats_update_failed", error=str(e))

    # =========================================================================
    # LEVEL 3: semantic experience memory, LLM reflection, user feedback
    # =========================================================================

    @property
    def semantic(self):
        if self._semantic is None:
            from app.state.semantic_memory import get_semantic_memory

            self._semantic = get_semantic_memory()
        return self._semantic

    @property
    def reflection(self):
        if self._reflection is None:
            from app.state.reflection import get_reflection_engine

            self._reflection = get_reflection_engine()
        return self._reflection

    async def recall_semantic(
        self, instruction: str, exclude_task_id: str | None = None, k: int | None = None
    ) -> list:
        """Past experiences similar in meaning to this instruction, best first."""
        if not get_settings().semantic_memory_enabled:
            return []
        try:
            return await self.semantic.recall(instruction, exclude_task_id=exclude_task_id, k=k)
        except Exception as e:
            logger.warning("semantic_recall_failed", error=str(e)[:200])
            return []

    async def learn_semantic(
        self,
        *,
        task_id: str,
        instruction: str,
        domain: str,
        scope: str,
        trajectory: list[dict[str, Any]],
        success: bool,
        error: str | None,
        raw_result: str,
        reflection=None,
        similar: list | None = None,
        previous_attempt: dict[str, Any] | None = None,
    ):
        """Reflect on a finished run with the LLM and store it as an experience.

        Safe to run in the background: it never raises. Order matters — similar
        past experiences are recalled BEFORE this run is stored, so the
        reflection can compare against earlier attempts and not against itself.
        """
        from app.state.reflection import OutcomeReflection

        settings = get_settings()
        try:
            if similar is None:
                similar = await self.recall_semantic(instruction, exclude_task_id=task_id)
            if reflection is None:
                reflection = await self.reflection.reflect_on_outcome(
                    instruction=instruction,
                    trajectory=trajectory,
                    success=success,
                    error=error,
                    raw_result=raw_result,
                    similar=similar,
                    previous_attempt=previous_attempt,
                )
            if reflection is None:
                template = self._generate_reflection(
                    task=instruction, trajectory=trajectory, success=success,
                    error=error, raw_result=raw_result,
                )
                reflection = OutcomeReflection(
                    what_succeeded=template.what_succeeded,
                    what_failed=template.what_failed,
                    root_cause=template.root_cause,
                    what_to_try_next=template.what_to_try_next,
                    from_llm=False,
                    confidence=template.confidence,
                )

            if settings.reflexion_enabled:
                await self._persist_reflection(domain, reflection, success)
            if settings.semantic_memory_enabled:
                await self.semantic.upsert(
                    task_id=task_id,
                    instruction=instruction,
                    domain=domain,
                    scope=scope,
                    success=success,
                    trajectory=trajectory,
                    lesson=reflection.as_lesson(),
                    error=error,
                )
            logger.info(
                "semantic_learning_complete",
                task_id=task_id,
                success=success,
                llm_reflection=getattr(reflection, "from_llm", False),
            )
            return reflection
        except Exception as e:
            logger.warning("semantic_learning_failed", task_id=task_id, error=str(e)[:300])
            return None

    async def record_feedback(self, task_id: str, rating: int, comment: str = "") -> dict[str, Any]:
        """Apply the user's 👍 (+1) or 👎 (-1) to a finished task.

        The rating is stored on the task's experience, where it weights future
        recall. A 👎 on a run the agent believed succeeded also marks that run
        as disputed and penalises the skills it taught, so a false success
        stops being reused.
        """
        rating = 1 if rating > 0 else -1
        result: dict[str, Any] = {"task_id": task_id, "rating": rating, "applied": False, "queued": False, "skills_penalized": 0}
        if not get_settings().semantic_memory_enabled:
            result["message"] = "Semantic memory is disabled, so feedback cannot be stored."
            return result

        unstored = self._unstored_runs.pop(task_id, None)
        if unstored is not None:
            # Stored now, before the rating, so the rating and the note land on it.
            await self._store_rated_run(task_id, unstored)
        exp = await self.semantic.record_feedback(task_id, rating, comment)
        if exp is None:
            result["queued"] = self.semantic.available
            result["message"] = (
                "Thanks. The task is still being saved to memory, and your rating will be applied as soon as it is."
                if result["queued"] else "Memory is unavailable right now, so the rating could not be stored."
            )
            return result

        result["applied"] = True
        note = f' I noted: "{comment[:150]}".' if comment else " Tell me what went wrong and I will avoid exactly that."
        if rating < 0 and exp.success:
            result["skills_penalized"] = await self._penalize_skills_for_tools(exp.domain, exp.tools)
            result["message"] = (
                "Got it. This run will no longer be reused as a working example, and the approach it "
                "taught was down-weighted." + note
            )
        elif rating < 0:
            result["message"] = "Got it. This failure is weighted more strongly so the approach is avoided." + note
        elif not exp.success:
            result["message"] = (
                "Thanks. I had marked this run as failed; it now counts as a success and its approach will be reused."
            )
        else:
            result["message"] = "Thanks. This approach will be preferred for similar tasks."
        return result

    def note_unstored_run(
        self,
        *,
        task_id: str,
        instruction: str,
        domain: str,
        scope: str,
        success: bool,
        error: str,
        report: str,
    ) -> None:
        """Remember a finished run that was not stored, in case the user rates it.

        A run in which no tool ran is not stored as an experience: nothing was
        done, so there is no approach to learn from. Found 2026-09-16: a 👎 with
        a note on such a run was queued for an experience that would never be
        written, so the user's explanation of what went wrong was lost.
        """
        self._unstored_runs[task_id] = {
            "instruction": instruction,
            "domain": domain,
            "scope": scope,
            "success": bool(success),
            "error": error or "",
            "report": report or "",
        }
        while len(self._unstored_runs) > _UNSTORED_RUNS_KEPT:
            self._unstored_runs.pop(next(iter(self._unstored_runs)))

    async def _store_rated_run(self, task_id: str, run: dict[str, Any]) -> None:
        """Store a no-tool run the user has rated, so the rating can teach."""
        what = (run.get("error") or run.get("report") or "").strip()[:300]
        if run.get("success"):
            lesson = f"Worked: answered without needing any tool. {what}".strip()
        else:
            lesson = f"Failed: no tool was used, so nothing was actually done. {what}".strip()
        await self.semantic.upsert(
            task_id=task_id,
            instruction=run.get("instruction", ""),
            domain=run.get("domain", ""),
            scope=run.get("scope", ""),
            success=bool(run.get("success")),
            trajectory=[],
            lesson=lesson,
            error=run.get("error") or None,
        )

    async def _penalize_skills_for_tools(self, domain: str, tools: list[str]) -> int:
        """Down-weight the skill a disputed run taught. Returns rows changed."""
        skill_type = local_skill_type_for(tools)
        if not skill_type:
            return 0
        try:
            session = await get_session()
            async with session.begin():
                stmt = select(SkillRecord).where(
                    SkillRecord.skill_type == skill_type,
                    SkillRecord.domain_pattern == domain,
                )
                existing = (await session.execute(stmt)).scalar_one_or_none()
                if not existing:
                    return 0
                existing.usage_count += 1
                existing.success_rate = (existing.success_rate * (existing.usage_count - 1)) / existing.usage_count
                existing.last_used_at = datetime.now(timezone.utc)
                logger.info(
                    "skill_penalized_by_user_feedback",
                    skill_type=skill_type,
                    domain=domain,
                    new_success_rate=f"{existing.success_rate:.1%}",
                )
                return 1
        except Exception as e:
            logger.warning("feedback_skill_penalty_failed", error=str(e)[:200])
            return 0

    # =========================================================================
    # DASHBOARD: Brain stats for the extension
    # =========================================================================

    async def get_brain_stats(self) -> dict[str, Any]:
        """Get global brain metrics for dashboard display."""
        try:
            session = await get_session()
            async with session.begin():
                stmt = select(BrainStateRecord).where(BrainStateRecord.id == 1)
                result = await session.execute(stmt)
                brain = result.scalar_one_or_none()

                if not brain:
                    return {
                        "total_tasks": 0,
                        "successful_tasks": 0,
                        "failed_tasks": 0,
                        "success_rate": 0.0,
                        "avg_duration_seconds": 0.0,
                        "total_skills_learned": 0,
                        "total_failure_patterns": 0,
                        "domains_mastered": [],
                        "last_learning_at": None,
                    }

                total = brain.total_tasks or 1
                return {
                    "total_tasks": brain.total_tasks,
                    "successful_tasks": brain.successful_tasks,
                    "failed_tasks": brain.failed_tasks,
                    "success_rate": brain.successful_tasks / total,
                    "avg_duration_seconds": brain.avg_duration_seconds,
                    "total_skills_learned": brain.total_skills_learned,
                    "total_failure_patterns": brain.total_failure_patterns,
                    "domains_mastered": json.loads(brain.domains_mastered_json or "[]"),
                    "last_learning_at": (
                        brain.last_learning_at.isoformat()
                        if brain.last_learning_at else None
                    ),
                }
        except Exception as e:
            logger.warning("get_brain_stats_failed", error=str(e))
            return {"total_tasks": 0, "success_rate": 0.0, "error": str(e)}

    async def get_all_skills(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get all learned skills for dashboard display."""
        try:
            session = await get_session()
            async with session.begin():
                stmt = (
                    select(SkillRecord)
                    .order_by(SkillRecord.usage_count.desc())
                    .limit(limit)
                )
                result = await session.execute(stmt)
                records = result.scalars().all()
                return [
                    {
                        "id": r.id,
                        "skill_type": r.skill_type,
                        "domain_pattern": r.domain_pattern,
                        "description": r.description,
                        "success_rate": r.success_rate,
                        "usage_count": r.usage_count,
                        "last_used_at": r.last_used_at.isoformat() if r.last_used_at else None,
                    }
                    for r in records
                ]
        except Exception as e:
            logger.warning("get_all_skills_failed", error=str(e))
            return []

    async def get_all_failure_patterns(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get all known failure patterns for dashboard display."""
        try:
            session = await get_session()
            async with session.begin():
                stmt = (
                    select(FailurePatternRecord)
                    .order_by(FailurePatternRecord.occurrence_count.desc())
                    .limit(limit)
                )
                result = await session.execute(stmt)
                records = result.scalars().all()
                return [
                    {
                        "id": r.id,
                        "error_signature": r.error_signature,
                        "domain": r.domain,
                        "root_cause": r.root_cause,
                        "solution_strategy": r.solution_strategy,
                        "occurrence_count": r.occurrence_count,
                        "last_seen_at": r.last_seen_at.isoformat() if r.last_seen_at else None,
                    }
                    for r in records
                ]
        except Exception as e:
            logger.warning("get_all_failure_patterns_failed", error=str(e))
            return []

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def _extract_domain_from_instruction(instruction: str) -> str:
        """Extract a domain name from a natural language instruction."""
        known_sites = {
            "amazon": "amazon.in",
            "flipkart": "flipkart.com",
            "onlinegdb": "onlinegdb.com",
            "c++ compiler": "onlinegdb.com",
            "online compiler": "onlinegdb.com",
            "c++": "onlinegdb.com",
            "leetcode": "leetcode.com",
            "codepen": "codepen.io",
            "replit": "replit.com",
            "github": "github.com",
            "jsfiddle": "jsfiddle.net",
            "hackerrank": "hackerrank.com",
            "programiz": "programiz.com",
            "w3schools": "w3schools.com",
            "ebay": "ebay.com",
            "walmart": "walmart.com",
            "apple": "apple.com",
            "nike": "nike.com",
            "myntra": "myntra.com",
            "ajio": "ajio.com",
            "snapdeal": "snapdeal.com",
            "meesho": "meesho.com",
        }

        instruction_lower = instruction.lower()
        for name, domain in known_sites.items():
            if name in instruction_lower:
                return domain

        # Try to find a URL in the instruction
        url_match = re.search(r"https?://[^\s]+", instruction)
        if url_match:
            return extract_domain(url_match.group())

        # Try to find domain-like patterns
        domain_match = re.search(r"\b(\w+\.\w{2,})\b", instruction)
        if domain_match:
            return domain_match.group(1)

        return "general"

    # =========================================================================
    # REFLEXION ENGINE (Rung 1: Procedural Self-Improvement)
    # =========================================================================

    def _generate_reflection(
        self,
        task: str,
        trajectory: list[dict[str, Any]],
        success: bool,
        error: str | None = None,
        raw_result: str = "",
    ) -> Reflection:
        """
        Generate a structured reflection from a completed task trajectory.

        Structured fields:
          - what_succeeded: summary of actions that worked
          - what_failed: failing action or error description
          - root_cause: underlying cause of failure or success enabling factor
          - what_to_try_next: actionable strategy/heuristic for subsequent attempts
          - confidence: confidence rating of this reflection
        """
        successful_actions = [
            a for a in trajectory
            if isinstance(a, dict) and a.get("success", False)
        ]
        failed_actions = [
            a for a in trajectory
            if isinstance(a, dict) and not a.get("success", True)
        ]

        if success:
            if successful_actions:
                act_types = [a.get("action_type", "") for a in successful_actions[:3]]
                what_succeeded = f"Executed {len(successful_actions)} action(s) successfully ({', '.join(act_types)})"
            else:
                what_succeeded = f"Task completed successfully: {task[:80]}"
            what_failed = "None"
            root_cause = "Effective action sequence and valid selectors matched page structure"
            what_to_try_next = "Continue using established selector hierarchy and action timing"
            confidence = 0.95
        else:
            err_text = error or (failed_actions[-1].get("error", "") if failed_actions else "Task failed")
            if successful_actions:
                what_succeeded = f"Progressed {len(successful_actions)} step(s) before stalling"
            else:
                what_succeeded = "None"

            what_failed = err_text[:150]
            err_lower = err_text.lower()
            if "timeout" in err_lower or "timed out" in err_lower:
                root_cause = "Action timed out waiting for DOM element or transition"
                what_to_try_next = "Scroll element into view first, verify selector stability, or lengthen wait timeout"
            elif "429" in err_lower or "rate_limit" in err_lower:
                root_cause = "LLM API rate limit exceeded during step reasoning"
                what_to_try_next = "Apply exponential backoff and trim prompt token context"
            elif "captcha" in err_lower or "bot" in err_lower:
                root_cause = "Anti-automation bot detection or CAPTCHA triggered"
                what_to_try_next = "Insert human-like delays and avoid repetitive rapid actions"
            elif "closed" in err_lower:
                root_cause = "Browser target page or session disconnected"
                what_to_try_next = "Ensure session persistence and reconnect cleanly before issuing commands"
            elif "selector" in err_lower or "not found" in err_lower:
                root_cause = "Target element selector did not match any visible interactive nodes"
                what_to_try_next = "Use semantic text matching or fallback accessible selector chain"
            else:
                root_cause = err_text[:120] if err_text else "Execution verification failed"
                what_to_try_next = "Inspect page snapshot carefully and verify interactive element availability"
            confidence = 0.85

        return Reflection(
            what_succeeded=what_succeeded,
            what_failed=what_failed,
            root_cause=root_cause,
            what_to_try_next=what_to_try_next,
            confidence=confidence,
        )

    async def _persist_reflection(
        self, domain: str, reflection: Reflection, success: bool
    ) -> None:
        """
        Persist structured reflection into domain memory and failure patterns.
        No schema migration required; reuses existing fields with typed tags.
        """
        try:
            # count_run=False: the run itself was already counted. Found
            # 2026-09-16: every task was counted two or three times, because each
            # tip and reflection written for it counted as another run.
            if success:
                tip = f"[REFLEXION] Known-good pattern: {reflection.what_succeeded}"
                await self._domain_store.record_domain_learning(
                    domain_or_url=domain,
                    general_tips=tip,
                    success=True,
                    count_run=False,
                )
            else:
                tip = f"[REFLEXION] Previous attempt failed: {reflection.root_cause} → this time: {reflection.what_to_try_next}"
                await self._domain_store.record_domain_learning(
                    domain_or_url=domain,
                    general_tips=tip,
                    success=False,
                    count_run=False,
                )
                session = await get_session()
                async with session.begin():
                    sig = _normalize_error(reflection.root_cause)
                    stmt = select(FailurePatternRecord).where(
                        FailurePatternRecord.error_signature == sig,
                        FailurePatternRecord.domain == domain,
                    )
                    result = await session.execute(stmt)
                    existing = result.scalar_one_or_none()
                    sol = f"this time: {reflection.what_to_try_next}"
                    if existing:
                        existing.occurrence_count += 1
                        existing.last_seen_at = datetime.now(timezone.utc)
                        existing.solution_strategy = sol
                    else:
                        rec = FailurePatternRecord(
                            error_signature=sig,
                            domain=domain,
                            root_cause=reflection.root_cause,
                            solution_strategy=sol,
                            occurrence_count=1,
                        )
                        session.add(rec)
        except Exception as e:
            logger.warning("persist_reflection_failed", domain=domain, error=str(e))

    async def _recall_reflections(self, domain: str) -> list[str]:
        """Recall reflections for a domain from failure patterns and domain memory."""
        reflections: list[str] = []
        try:
            # 1. From FailurePatternRecord for this domain
            session = await get_session()
            async with session.begin():
                stmt = select(FailurePatternRecord).where(
                    or_(
                        FailurePatternRecord.domain == domain,
                        FailurePatternRecord.domain == "*",
                    )
                ).order_by(FailurePatternRecord.occurrence_count.desc()).limit(3)
                result = await session.execute(stmt)
                records = result.scalars().all()
                for r in records:
                    strat = r.solution_strategy
                    if strat.startswith("this time: "):
                        strat = strat[len("this time: "):]
                    entry = f"Previous attempt failed: {r.root_cause} → this time: {strat}"
                    if entry not in reflections:
                        reflections.append(entry)

            # 2. From domain memory tips with [REFLEXION] tag
            domain_mem = await self._domain_store.get_domain_memory(domain)
            tips = domain_mem.get("general_tips", "")
            for line in tips.splitlines():
                clean_line = line.strip().lstrip("- ").strip()
                if clean_line.startswith("[REFLEXION]"):
                    content = clean_line[len("[REFLEXION]"):].strip()
                    if content and content not in reflections:
                        reflections.append(content)

        except Exception as e:
            logger.warning("recall_reflections_failed", domain=domain, error=str(e))

        return reflections
