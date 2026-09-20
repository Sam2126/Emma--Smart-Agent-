"""
Adapter Registry and Evaluation Rollback Gate.

Implements §12.1 of the RL Implementation Plan (v2):
- Records all trained adapters (SFT / DPO / GRPO / PPO)
- Mandatory Post-Training Gate:
    Evaluates new adapters against frozen baseline.
    If success rate regresses > 2.0 points vs previous best -> AUTO-REJECT,
    preserving the prior production adapter.
- Provides a clean inference hook for the agent engine (default: no adapter -> 100% unchanged).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import structlog

logger = structlog.get_logger(__name__)

DEFAULT_REGISTRY_PATH = "data/rl/adapter_registry.json"
_BACKEND_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class AdapterRecord:
    """Record of a trained policy adapter."""
    run_id: str
    rung: int  # 2=SFT, 3=DPO, 4=GRPO, 5=PPO
    base_model: str
    adapter_path: str
    benchmark_accuracy: float = 0.0  # 0.0 to 100.0
    status: str = "pending"  # "pending", "accepted", "rejected"
    rejection_reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class AdapterRegistry:
    """
    Manages adapter versioning, promotion gates, and automatic rollbacks.
    """

    def __init__(self, registry_path: str = DEFAULT_REGISTRY_PATH) -> None:
        # Relative to the backend folder, like the other data paths, so the
        # model router finds a promoted adapter wherever the backend was started.
        path = Path(registry_path)
        self.registry_path = path if path.is_absolute() else _BACKEND_ROOT / path
        self.adapters: dict[str, AdapterRecord] = {}
        self.active_adapter_id: str | None = None
        self.best_accuracy: float = 0.0
        self.load()

    def load(self) -> None:
        """Load registry from JSON storage."""
        if not self.registry_path.exists():
            return
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.active_adapter_id = data.get("active_adapter_id")
            self.best_accuracy = data.get("best_accuracy", 0.0)
            self.adapters = {
                k: AdapterRecord(**v) for k, v in data.get("adapters", {}).items()
            }
        except Exception as e:
            logger.warning("failed_to_load_adapter_registry", error=str(e))

    def save(self) -> None:
        """Save registry to JSON storage."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "active_adapter_id": self.active_adapter_id,
            "best_accuracy": self.best_accuracy,
            "adapters": {k: asdict(v) for k, v in self.adapters.items()},
        }
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def register_adapter(
        self,
        run_id: str,
        rung: int,
        base_model: str,
        adapter_path: str,
        metrics: dict[str, Any] | None = None,
    ) -> AdapterRecord:
        """Register a newly trained adapter in pending state."""
        rec = AdapterRecord(
            run_id=run_id,
            rung=rung,
            base_model=base_model,
            adapter_path=adapter_path,
            metrics=metrics or {},
        )
        self.adapters[run_id] = rec
        self.save()
        logger.info("adapter_registered", run_id=run_id, rung=rung, path=adapter_path)
        return rec

    def evaluate_and_gate(
        self,
        run_id: str,
        eval_accuracy: float,
        regression_tolerance: float = 2.0,
    ) -> tuple[bool, str]:
        """
        Evaluate adapter accuracy against the frozen benchmark.

        Rollback gate:
          If eval_accuracy < (best_accuracy - regression_tolerance) -> AUTO-REJECT.
          Otherwise -> ACCEPT and promote if it improves performance.
        """
        rec = self.adapters.get(run_id)
        if not rec:
            raise KeyError(f"Adapter '{run_id}' not found in registry.")

        rec.benchmark_accuracy = eval_accuracy

        # Check regression threshold
        allowed_floor = self.best_accuracy - regression_tolerance
        if eval_accuracy < allowed_floor and self.best_accuracy > 0.0:
            rec.status = "rejected"
            rec.rejection_reason = (
                f"Regression > {regression_tolerance}% vs previous best "
                f"({eval_accuracy:.1f}% vs {self.best_accuracy:.1f}%)"
            )
            self.save()
            logger.warning("adapter_rejected_by_gate", run_id=run_id, reason=rec.rejection_reason)
            return False, rec.rejection_reason

        # Accepted
        rec.status = "accepted"
        if eval_accuracy >= self.best_accuracy:
            self.best_accuracy = eval_accuracy
            self.active_adapter_id = run_id
            logger.info("adapter_promoted_to_active", run_id=run_id, accuracy=eval_accuracy)

        self.save()
        return True, "Adapter passed frozen evaluation gate"

    def get_active_adapter(self) -> AdapterRecord | None:
        """Get the currently active production adapter, if any."""
        if self.active_adapter_id and self.active_adapter_id in self.adapters:
            return self.adapters[self.active_adapter_id]
        return None


def get_current_production_adapter(registry_path: str = DEFAULT_REGISTRY_PATH) -> str | None:
    """
    Inference hook for agent nodes.
    Returns path to active adapter if registered and accepted; None otherwise.
    """
    registry = AdapterRegistry(registry_path=registry_path)
    active = registry.get_active_adapter()
    if active and active.status == "accepted":
        return active.adapter_path
    return None
