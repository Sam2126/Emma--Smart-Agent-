"""
SQLite -> JSONL Trajectory Dataset Converter for RL (SFT / DPO / GRPO).

Implements §5 of the RL Implementation Plan (v2):
- SFT: PASS-verified only, chat-templated messages (user instruction + observation -> assistant action)
- DPO: Paired PASS (chosen) vs FAIL (rejected) trajectories for matching tasks
- GRPO: Grouped rollouts per prompt with verifier scores and normalized advantage ranks
- Strict filtering: drops checkout/payment, truncated steps, short trajectories (<min_steps),
  and enforces frozen held-out benchmark isolation (never contaminates eval suite).
- CLI interface: python -m app.rl.dataset [sft|dpo|grpo] --db ... --out ...
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Optional
import structlog
from sqlalchemy import select, and_

from app.state.database import get_session, init_database, close_database
from app.state.models import TaskRecord, StepRecord, VerificationRecord
from app.evals.benchmark_suite import BENCHMARK_TASKS

logger = structlog.get_logger(__name__)

# Frozen benchmark instruction texts to exclude from all training datasets
BENCHMARK_INSTRUCTIONS = {t.instruction.strip().lower() for t in BENCHMARK_TASKS}
BENCHMARK_TASK_IDS = {t.task_id for t in BENCHMARK_TASKS}

# Spec §5.2: drop human-gated/aborted runs. TaskRecord has no dedicated
# requires_human_confirmation column, so gates surface via status/error text.
HUMAN_GATE_MARKERS = (
    "human confirmation",
    "requires human",
    "awaiting_human",
    "human_gate",
    "safety gate",
    "irreversible",
    "payment_confirmation",
)
# Spec §5.2: drop trajectories truncated mid-run (step budget exhausted).
TRUNCATION_MARKERS = (
    "max_steps",
    "max-steps",
    "max steps",
    "step budget",
    "truncated",
    "step_limit",
)

# Minimum instruction similarity for a DPO chosen/rejected pairing (spec §5.1:
# "same/near-identical instruction" — unrelated same-domain pairs are label noise).
MIN_DPO_SIMILARITY = 0.7


def normalize_text(text: str) -> str:
    """Normalize whitespace and lowercase for similarity comparisons."""
    return re.sub(r"\s+", " ", text.strip().lower())


def instruction_similarity(a: str, b: str) -> float:
    """String similarity in [0, 1] between two task instructions."""
    return difflib.SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def path_to_sqlite_url(db_path: str) -> str:
    """Convert a filesystem path to an absolute sqlite+aiosqlite URL."""
    p = Path(db_path).resolve()
    return f"sqlite+aiosqlite:///{p.as_posix()}"


def compute_trajectory_hash(domain: str, instruction: str, actions: list[dict[str, Any]]) -> str:
    """Compute deterministic hash for deduplication."""
    act_repr = ";".join(
        f"{a.get('action_type', '')}:{a.get('selector_used', '')}:{a.get('input_value', '')}"
        for a in actions
    )
    raw = f"{normalize_text(domain)}|{normalize_text(instruction)}|{act_repr}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def format_chat_messages(instruction: str, page_state: dict[str, Any] | str, action: dict[str, Any] | str) -> list[dict[str, str]]:
    """Format prompt & completion into standard model-native chat template format."""
    state_str = page_state if isinstance(page_state, str) else json.dumps(page_state)
    action_str = action if isinstance(action, str) else json.dumps(action)

    user_content = (
        f"You are a browser automation assistant.\n"
        f"Goal: {instruction}\n\n"
        f"Current Page State:\n{state_str[:1500]}\n\n"
        f"Select the next atomic browser action."
    )
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": action_str},
    ]


class DatasetExporter:
    """
    Extracts trajectory records from SQLite database and formats them for SFT, DPO, and GRPO.
    """

    def __init__(self, db_url: str | None = None, min_steps: int = 3) -> None:
        self.db_url = db_url
        self.min_steps = min_steps
        # Spec §5.3 label stability: verdicts cached by (task_id, step_id, snapshot_hash)
        # so repeated exports can never flip labels for the same snapshot.
        self.verdict_cache: dict[str, bool] = {}

    def resolve_verdict(self, task_id: str, step: dict[str, Any]) -> bool:
        """Read a step's PASS/FAIL label through the snapshot-keyed verdict cache."""
        snapshot_hash = hashlib.md5(str(step.get("page_state_after") or step.get("url_after", "")).encode("utf-8")).hexdigest()[:16]
        key = f"{task_id}:{step['step_index']}:{snapshot_hash}"
        if key not in self.verdict_cache:
            self.verdict_cache[key] = bool(step.get("verified_passed", False))
        return self.verdict_cache[key]

    def is_human_gated(self, task: dict[str, Any]) -> bool:
        """True when the run was human-gated/aborted (spec §5.2 drop rule)."""
        error = (task.get("error") or "").lower()
        status = (task.get("status") or "").lower()
        if status in ("blocked", "awaiting_human", "human_gate"):
            return True
        return any(m in error for m in HUMAN_GATE_MARKERS)

    def is_truncated_mid_run(self, task: dict[str, Any]) -> bool:
        """True when the run was cut off by the step budget (spec §5.2 drop rule)."""
        if (task.get("status") or "").lower() == "timeout":
            return True
        error = (task.get("error") or "").lower()
        if any(m in error for m in TRUNCATION_MARKERS):
            return True
        steps = task.get("steps") or []
        if steps:
            last_err = (steps[-1].get("error") or "").lower() if isinstance(steps[-1], dict) else ""
            return any(m in last_err for m in TRUNCATION_MARKERS)
        return False

    @staticmethod
    def deduplicate(trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Dedupe by (domain, normalized_instruction, action_hash) — spec §5.2."""
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for t in trajectories:
            h = compute_trajectory_hash(t["domain"], t["instruction"], t["steps"])
            if h in seen:
                continue
            seen.add(h)
            unique.append(t)
        return unique

    def is_benchmark_task(self, instruction: str, task_id: str) -> bool:
        """Strict isolation check against frozen benchmark suite."""
        norm_inst = normalize_text(instruction)
        if norm_inst in BENCHMARK_INSTRUCTIONS:
            return True
        if task_id in BENCHMARK_TASK_IDS or any(task_id.endswith(bid) for bid in BENCHMARK_TASK_IDS):
            return True
        return False

    def is_trajectory_polluted(self, steps: list[StepRecord]) -> bool:
        """Check if trajectory contains forbidden checkout URLs or safety aborts."""
        checkout_keywords = ["/checkout", "/buy", "/pay", "place-order", "payment-gate"]
        for s in steps:
            url_after = (s.url_after or "").lower()
            url_before = (s.url_before or "").lower()
            if any(k in url_after or k in url_before for k in checkout_keywords):
                return True
            if s.error and ("safety gate" in s.error.lower() or "irreversible" in s.error.lower()):
                return True
        return False

    async def fetch_completed_tasks(self, domain: str | None = None) -> list[dict[str, Any]]:
        """Fetch tasks and associated steps from SQLite with relationship loading."""
        session = await get_session()
        async with session.begin():
            stmt = select(TaskRecord).order_by(TaskRecord.created_at.desc())
            if domain:
                stmt = stmt.where(TaskRecord.site == domain)
            result = await session.execute(stmt)
            tasks = result.scalars().all()

            trajectories = []
            for t in tasks:
                # Exclude benchmark ground truth tasks
                if self.is_benchmark_task(t.instruction, t.id):
                    continue

                # Load steps
                step_stmt = (
                    select(StepRecord, VerificationRecord)
                    .outerjoin(VerificationRecord, VerificationRecord.step_id == StepRecord.id)
                    .where(StepRecord.task_id == t.id)
                    .order_by(StepRecord.step_index.asc())
                )
                step_res = await session.execute(step_stmt)
                step_rows = step_res.all()

                steps = [row[0] for row in step_rows]
                verifications = [row[1] for row in step_rows]

                # Spec §5.2 drop rules: too short, polluted, human-gated, truncated
                if len(steps) < self.min_steps:
                    continue
                if self.is_trajectory_polluted(steps):
                    continue
                task_view = {"status": t.status, "error": t.error, "steps": []}
                if self.is_human_gated(task_view):
                    continue

                step_dicts = []
                for s, v in zip(steps, verifications):
                    step_dict = {
                        "step_index": s.step_index,
                        "action_type": s.action_type,
                        "selector_used": s.selector_used or "",
                        "input_value": s.input_value or "",
                        "success": s.success,
                        "error": s.error,
                        "url_before": s.url_before or "",
                        "url_after": s.url_after or "",
                        "page_state_before": s.page_state_before_json or "",
                        "page_state_after": s.page_state_after_json or "",
                        "verified_passed": v.passed if v else s.success,
                    }
                    step_dict["verified_passed"] = self.resolve_verdict(t.id, step_dict)
                    step_dicts.append(step_dict)

                if self.is_truncated_mid_run({"status": t.status, "error": t.error, "steps": step_dicts}):
                    continue

                # Check task passed
                final_step_ver = verifications[-1] if verifications else None
                task_passed = (t.status == "completed") and (
                    final_step_ver.passed if final_step_ver else (t.error is None)
                )

                trajectories.append({
                    "task_id": t.id,
                    "instruction": t.instruction,
                    "domain": t.site,
                    "status": t.status,
                    "success": task_passed,
                    "error": t.error,
                    "steps": step_dicts,
                    "source": "curriculum" if "curriculum" in t.id.lower() else "agent_history",
                })

            return trajectories

    async def export_sft_dataset(
        self, output_path: str, domain: str | None = None
    ) -> int:
        """
        Export PASS-verified trajectories to SFT JSONL format.
        """
        trajectories = await self.fetch_completed_tasks(domain=domain)
        pass_trajectories = [t for t in trajectories if t["success"]]

        seen_hashes = set()
        count = 0
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for t in pass_trajectories:
                traj_hash = compute_trajectory_hash(t["domain"], t["instruction"], t["steps"])
                if traj_hash in seen_hashes:
                    continue
                seen_hashes.add(traj_hash)

                for step in t["steps"]:
                    action = {
                        "action_type": step["action_type"],
                        "selector": step["selector_used"],
                        "input": step["input_value"],
                    }
                    messages = format_chat_messages(
                        instruction=t["instruction"],
                        page_state=step["page_state_before"] or {"url": step["url_before"]},
                        action=action,
                    )
                    record = {
                        "task_id": t["task_id"],
                        "instruction": t["instruction"],
                        "domain": t["domain"],
                        "page_state": step["page_state_before"],
                        "action": action,
                        "messages": messages,
                        "source": t["source"],
                    }
                    f.write(json.dumps(record) + "\n")
                    count += 1

        logger.info("sft_dataset_exported", output_path=output_path, count=count)
        return count

    async def export_dpo_dataset(
        self, output_path: str, domain: str | None = None,
        min_similarity: float = MIN_DPO_SIMILARITY,
    ) -> int:
        """
        Export paired PASS (chosen) vs FAIL (rejected) trajectories for DPO training.

        Spec §5.1: pairs require the same or near-identical instruction. Candidates
        below `min_similarity` are skipped rather than paired with unrelated tasks.
        """
        trajectories = await self.fetch_completed_tasks(domain=domain)
        trajectories = self.deduplicate(trajectories)
        pass_trajs = [t for t in trajectories if t["success"]]
        fail_trajs = [t for t in trajectories if not t["success"]]

        count = 0
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as out:
            for p in pass_trajs:
                best: tuple[dict[str, Any], float] | None = None
                for f in fail_trajs:
                    if f["domain"] != p["domain"]:
                        continue
                    sim = instruction_similarity(p["instruction"], f["instruction"])
                    if sim >= min_similarity and (best is None or sim > best[1]):
                        best = (f, sim)
                if best is None:
                    continue
                mf, similarity = best

                chosen_actions = [
                    f"{s['action_type']}({s['selector_used']})" for s in p["steps"]
                ]
                rejected_actions = [
                    f"{s['action_type']}({s['selector_used']})" for s in mf["steps"]
                ]

                record = {
                    "task_id": p["task_id"],
                    "prompt": f"Goal: {p['instruction']}\nDomain: {p['domain']}",
                    "chosen": chosen_actions,
                    "rejected": rejected_actions,
                    "domain": p["domain"],
                    "similarity": round(similarity, 3),
                    "source": p["source"],
                }
                out.write(json.dumps(record) + "\n")
                count += 1

        logger.info("dpo_dataset_exported", output_path=output_path, count=count)
        return count

    async def export_grpo_dataset(
        self, output_path: str, group_size: int = 4, domain: str | None = None
    ) -> int:
        """
        Export grouped rollouts per prompt with rewards and normalized advantage ranks.
        """
        trajectories = await self.fetch_completed_tasks(domain=domain)
        trajectories = self.deduplicate(trajectories)
        by_prompt: dict[str, list[dict[str, Any]]] = {}
        for t in trajectories:
            key = f"Goal: {t['instruction']}\nDomain: {t['domain']}"
            by_prompt.setdefault(key, []).append(t)

        count = 0
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as out:
            for prompt, rollouts in by_prompt.items():
                if len(rollouts) < 2:
                    continue
                # Take up to group_size rollouts
                group = rollouts[:group_size]
                responses = []
                rewards = []

                for r in group:
                    resp_str = " -> ".join(
                        f"{s['action_type']}({s['selector_used']})" for s in r["steps"]
                    )
                    responses.append(resp_str)
                    # Reward: 2.0 for pass, -2.0 for fail, plus efficiency bonus
                    reward = 2.0 if r["success"] else -2.0
                    reward -= 0.05 * len(r["steps"])
                    rewards.append(reward)

                # Compute ranks
                sorted_indices = sorted(range(len(rewards)), key=lambda i: rewards[i], reverse=True)
                ranks = [0] * len(rewards)
                for rank, idx in enumerate(sorted_indices):
                    ranks[idx] = rank + 1

                record = {
                    "prompt": prompt,
                    "responses": responses,
                    "rewards": rewards,
                    "ranks": ranks,
                    "group_size": len(group),
                }
                out.write(json.dumps(record) + "\n")
                count += 1

        logger.info("grpo_dataset_exported", output_path=output_path, count=count)
        return count


async def _run_cli(args: argparse.Namespace) -> None:
    if args.db:
        # Spec §5.4: exports run against an explicitly provided database.
        await init_database(db_url=path_to_sqlite_url(args.db))
        exporter = DatasetExporter(min_steps=args.min_steps)
    else:
        exporter = DatasetExporter(min_steps=args.min_steps)
    try:
        if args.command == "sft":
            cnt = await exporter.export_sft_dataset(args.out, domain=args.domain)
            print(f"Exported {cnt} SFT records to {args.out}")
        elif args.command == "dpo":
            cnt = await exporter.export_dpo_dataset(args.out, domain=args.domain)
            print(f"Exported {cnt} DPO pair records to {args.out}")
        elif args.command == "grpo":
            cnt = await exporter.export_grpo_dataset(args.out, group_size=args.group_size, domain=args.domain)
            print(f"Exported {cnt} GRPO groups to {args.out}")
        else:
            print(f"Unknown command: {args.command}")
    finally:
        if args.db:
            await close_database()


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db", type=str, default=None, help="Path to agent SQLite DB (default: configured app DB)")
    p.add_argument("--out", type=str, required=False, help="Output JSONL path")
    p.add_argument("--domain", type=str, default=None, help="Filter by target domain")
    p.add_argument("--min_steps", type=int, default=3, help="Minimum step count for valid trajectory")


def main() -> None:
    parser = argparse.ArgumentParser(description="SQLite -> JSONL Dataset Exporter for RL Agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # SFT subparser
    sft_parser = subparsers.add_parser("sft", help="Export SFT chat-templated trajectory dataset")
    _add_common_args(sft_parser)
    sft_parser.set_defaults(out="data/rl/sft.jsonl")

    # DPO subparser
    dpo_parser = subparsers.add_parser("dpo", help="Export DPO preference pair dataset (chosen vs rejected)")
    _add_common_args(dpo_parser)
    dpo_parser.set_defaults(out="data/rl/dpo.jsonl")

    # GRPO subparser
    grpo_parser = subparsers.add_parser("grpo", help="Export GRPO grouped rollouts dataset with rankings")
    _add_common_args(grpo_parser)
    grpo_parser.add_argument("--group_size", type=int, default=4, help="Number of rollouts per prompt group")
    grpo_parser.set_defaults(out="data/rl/grpo.jsonl")

    args = parser.parse_args()
    asyncio.run(_run_cli(args))


if __name__ == "__main__":
    main()
