"""
Self-Evolving Task Curriculum Engine for Online RL.

Implements §8 of the RL Implementation Plan (v2):
- Discovers error signatures and recurring failure modes from FailurePatternRecords
- Generates tiered tasks:
    - Tier 1: Basic navigation and single-target search
    - Tier 2: Search + attribute filtering / facet selection
    - Tier 3: Multi-step traversal, overlays, sorting, and edge cases
- Mutates tasks via parameter variation (brands, queries, price constraints, ratings)
- CurriculumScheduler: biased sampling toward current weak accuracy bands,
  strictly tags tasks with `source: curriculum` (preventing benchmark eval contamination).
- Closed-loop reflection: failed rollouts directly feed new curriculum shards.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import structlog
from sqlalchemy import select, func

from app.state.database import get_session
from app.state.models import FailurePatternRecord, DomainMemoryRecord

logger = structlog.get_logger(__name__)

# Base templates for curriculum task generation
TIER_TEMPLATES = {
    1: [
        "Go to {domain} and search for {product}",
        "Search for {product} on {domain}",
        "Look up {product} on {domain} homepage",
    ],
    2: [
        "Search for {product} on {domain} and filter by brand {brand}",
        "Find {product} on {domain} with customer rating 4 stars & up",
        "Search {product} on {domain} and select price under {price_limit}",
        "Filter for {brand} {product} on {domain} and sort by price low to high",
    ],
    3: [
        "Search {product} on {domain}, apply brand {brand} filter, dismiss any popups, and open first item",
        "On {domain}, search for {product}, filter by 4+ stars, and add the lowest price item to cart",
        "Find {brand} {product} under {price_limit} on {domain}, verify in-stock, and proceed to cart",
    ],
}

SEED_PRODUCTS = [
    ("headphones", "Sony", "₹5,000"),
    ("smartwatch", "Noise", "₹3,000"),
    ("laptop", "Dell", "₹60,000"),
    ("running shoes", "Nike", "₹4,500"),
    ("wireless mouse", "Logitech", "₹1,500"),
    ("mechanical keyboard", "Keychron", "₹7,000"),
]


@dataclass
class CurriculumTask:
    """A generated curriculum task for RL training."""
    task_id: str
    instruction: str
    domain: str
    tier: int
    target_failure_sig: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "curriculum"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "domain": self.domain,
            "tier": self.tier,
            "target_failure_sig": self.target_failure_sig,
            "metadata": self.metadata,
            "source": self.source,
        }


class TaskCurriculumGenerator:
    """
    Generates and evolves tiered tasks targeting known agent weaknesses.
    """

    def __init__(self) -> None:
        self.rng = random.Random(42)

    async def discover_failure_signatures(self, domain: str | None = None) -> list[dict[str, Any]]:
        """
        Aggregate failure patterns by error signature and domain.
        """
        try:
            session = await get_session()
            async with session.begin():
                stmt = select(FailurePatternRecord).order_by(
                    FailurePatternRecord.occurrence_count.desc()
                )
                if domain:
                    stmt = stmt.where(FailurePatternRecord.domain == domain)
                result = await session.execute(stmt)
                records = result.scalars().all()

                signatures = []
                for r in records:
                    signatures.append({
                        "error_signature": r.error_signature,
                        "domain": r.domain,
                        "root_cause": r.root_cause,
                        "solution_strategy": r.solution_strategy,
                        "occurrence_count": r.occurrence_count,
                    })
                return signatures
        except Exception as e:
            logger.warning("discover_failure_signatures_failed", error=str(e))
            return []

    def generate_tiered_tasks(
        self,
        domain: str = "sandbox://ecommerce",
        n_per_tier: int = 2,
        failure_signatures: list[dict[str, Any]] | None = None,
    ) -> list[CurriculumTask]:
        """
        Generate synthetic tasks across Tiers 1, 2, and 3.
        """
        tasks: list[CurriculumTask] = []
        task_counter = 1

        for tier in (1, 2, 3):
            templates = TIER_TEMPLATES[tier]
            for _ in range(n_per_tier):
                prod, brand, price = self.rng.choice(SEED_PRODUCTS)
                template = self.rng.choice(templates)

                instruction = template.format(
                    domain=domain,
                    product=prod,
                    brand=brand,
                    price_limit=price,
                )

                # Pair with relevant failure signature if available (rotate by index)
                sig = ""
                if failure_signatures:
                    sig = failure_signatures[task_counter % len(failure_signatures)]["error_signature"]

                task_id = f"curriculum_t{tier}_{task_counter:04d}"
                tasks.append(
                    CurriculumTask(
                        task_id=task_id,
                        instruction=instruction,
                        domain=domain,
                        tier=tier,
                        target_failure_sig=sig,
                        metadata={
                            "product": prod,
                            "brand": brand,
                            "price_limit": price,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                )
                task_counter += 1

        return tasks

    def mutate_task(self, task: CurriculumTask) -> CurriculumTask:
        """
        Mutate task parameters to create novel variations (WebRL mutation trick).
        """
        new_prod, new_brand, new_price = self.rng.choice(SEED_PRODUCTS)
        # Substitute words in instruction
        text = task.instruction
        for p, b, pr in SEED_PRODUCTS:
            text = text.replace(p, new_prod).replace(b, new_brand).replace(pr, new_price)

        if text == task.instruction:
            # Parameter-level fallback per spec §8: swap the numeric budget, or
            # append a constraint variation when no known parameter is present.
            text = re.sub(r"₹[\d,]+", f"₹{new_price}", text, count=1)
        if text == task.instruction:
            text = f"{text} (variation: pick a different brand if the first is unavailable)"

        mutated_id = f"{task.task_id}_mut_{self.rng.randint(100, 999)}"
        new_meta = dict(task.metadata)
        new_meta["mutated_from"] = task.task_id
        new_meta["product"] = new_prod
        new_meta["brand"] = new_brand

        return CurriculumTask(
            task_id=mutated_id,
            instruction=text,
            domain=task.domain,
            tier=task.tier,
            target_failure_sig=task.target_failure_sig,
            metadata=new_meta,
        )


class CurriculumScheduler:
    """
    Schedules tasks for online RL rollouts biased toward agent's active weak band.
    """

    def __init__(self, tasks: list[CurriculumTask] | None = None) -> None:
        self.tasks: list[CurriculumTask] = tasks or []
        self.history: list[dict[str, Any]] = []
        self.accuracy_by_tier: dict[int, float] = {1: 0.8, 2: 0.4, 3: 0.2}

    def add_tasks(self, tasks: list[CurriculumTask]) -> None:
        self.tasks.extend(tasks)

    def sample_task(self) -> CurriculumTask:
        """
        Sample task with probability inversely proportional to tier accuracy
        (more tasks sampled from weak tiers).
        """
        if not self.tasks:
            gen = TaskCurriculumGenerator()
            self.tasks = gen.generate_tiered_tasks()

        # Weight tiers: lower accuracy -> higher weight
        weights = []
        for t in self.tasks:
            acc = self.accuracy_by_tier.get(t.tier, 0.5)
            weight = max(0.1, 1.0 - acc)
            weights.append(weight)

        total_w = sum(weights)
        norm_weights = [w / total_w for w in weights]

        chosen = random.choices(self.tasks, weights=norm_weights, k=1)[0]
        return chosen

    def update_performance(self, tier: int, success: bool) -> None:
        """Update running accuracy estimation for a tier."""
        curr = self.accuracy_by_tier.get(tier, 0.5)
        new_val = 0.9 * curr + 0.1 * (1.0 if success else 0.0)
        self.accuracy_by_tier[tier] = round(new_val, 3)

    def reflect_into_curriculum(self, failed_task: CurriculumTask, error: str) -> CurriculumTask:
        """Closed-loop feedback: create a remedial task variant addressing failure."""
        gen = TaskCurriculumGenerator()
        mutated = gen.mutate_task(failed_task)
        mutated.target_failure_sig = error[:80]
        mutated.metadata["remedial"] = True
        self.tasks.append(mutated)
        return mutated
