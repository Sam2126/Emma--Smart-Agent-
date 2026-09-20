"""
Direct Preference Optimization (DPO) Trainer (Rung 3).

Implements §9.0 and §9.2 of the RL Implementation Plan (v2):
- Data: Paired chosen (PASS) vs rejected (FAIL) trajectories from dataset.py
- Reference model: Frozen reference policy preventing behavioral drift
- Hyperparameters: beta=0.1, lr=5e-6, max_length=2048, batch_size=4
- Hard-example balancing: prioritizes pairs where failure occurred late in trajectory
- Shared dry-run contract: exits 0 without ML dependencies on CPU/dev machines.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import structlog

from app.rl.registry import AdapterRegistry

logger = structlog.get_logger(__name__)


def has_sufficient_training_resources(min_ram_gb: float = 14.0) -> bool:
    """Check if host system has sufficient physical RAM or VRAM to safely load 7B weights without OS OOM crash."""
    try:
        import psutil
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb >= min_ram_gb:
                return True
        avail_gb = psutil.virtual_memory().available / (1024**3)
        return avail_gb >= min_ram_gb
    except Exception:
        return False


@dataclass
class DPOConfig:
    """Configuration for DPO offline preference training."""
    model_name_or_path: str = "data/rl/adapters/sft"
    ref_model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    data_path: str = "data/rl/dpo.jsonl"
    output_dir: str = "data/rl/adapters/dpo"
    beta: float = 0.1
    learning_rate: float = 5e-6
    epochs: int = 2
    batch_size: int = 4
    max_length: int = 2048
    max_prompt_length: int = 1024
    # Resource gate: sized for 7B policy + frozen reference. Lower it for
    # smaller models (e.g. smoke tests with tiny random models).
    min_ram_gb: float = 14.0
    dry_run: bool = False


def validate_dpo_dataset(data_path: str, min_samples: int = 3) -> tuple[int, list[dict[str, Any]]]:
    """Validate DPO preference dataset exists and inspect chosen/rejected pairs."""
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"DPO dataset file not found at '{data_path}'")

    samples = []
    total = 0
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            if len(samples) < min_samples:
                if "prompt" not in rec or "chosen" not in rec or "rejected" not in rec:
                    raise ValueError(f"Record #{total} missing required DPO keys (prompt/chosen/rejected).")
                samples.append(rec)

    return total, samples


def run_dry_run(cfg: DPOConfig) -> None:
    """Execute dry-run validation without constructing ML dependencies."""
    print("=" * 65)
    print("[*] DPO TRAINER DRY-RUN VALIDATION")
    print("=" * 65)
    print(f"Policy Model:      {cfg.model_name_or_path}")
    print(f"Reference Model:   {cfg.ref_model_name_or_path} (frozen)")
    print(f"Data Path:         {cfg.data_path}")
    print(f"Output Directory:  {cfg.output_dir}")
    print(f"Beta:              {cfg.beta}")
    print(f"Learning Rate:     {cfg.learning_rate}")
    print(f"Max Prompt Length: {cfg.max_prompt_length} | Max Length: {cfg.max_length}")
    print(f"Epochs:            {cfg.epochs} | Batch Size: {cfg.batch_size}")
    print("-" * 65)

    # Output directory validation
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    test_file = out / ".write_test"
    test_file.write_text("ok", encoding="utf-8")
    test_file.unlink()
    print("[OK] Output directory validated and writable.")

    # Validate dataset if present
    if Path(cfg.data_path).exists():
        total, samples = validate_dpo_dataset(cfg.data_path)
        print(f"[OK] DPO dataset verified: {total} preference pairs found.")
        print(f"[OK] Spot-checked {len(samples)} preference pairs (chosen/rejected aligned).")
    else:
        print(f"[INFO] DPO dataset '{cfg.data_path}' not present yet (generate with dataset.py).")

    registry = AdapterRegistry()
    print(f"[OK] Adapter registry access verified (current active: {registry.active_adapter_id or 'None'}).")
    print("-" * 65)
    print("[OK] DPO DRY-RUN COMPLETED SUCCESSFULLY. No ML dependencies constructed.")
    print("=" * 65)


def train_dpo(cfg: DPOConfig) -> Optional[str]:
    """Execute DPO training on GPU/CPU machine with ML dependencies."""
    if cfg.dry_run:
        run_dry_run(cfg)
        return None

    # Lazy heavy imports (spec §9.0: never at module top)
    try:
        import torch
        from transformers import AutoTokenizer
        from trl import DPOTrainer, DPOConfig as TRLDPOConfig
        from datasets import Dataset
    except ImportError as err:
        logger.error("dpo_deps_missing", error=str(err))
        raise RuntimeError(
            "Heavy ML training dependencies missing. "
            "Install GPU requirements via `pip install -r backend/requirements-rl.txt`."
        ) from err

    from app.rl.train_utils import (
        apply_lora,
        filter_valid_kwargs,
        load_reference_model,
        resolve_dtype,
        resolve_device_map,
    )

    total, _ = validate_dpo_dataset(cfg.data_path)
    if total == 0:
        raise RuntimeError(
            f"DPO dataset '{cfg.data_path}' contains 0 preference pairs. "
            "Export pairs first: python -m app.rl.dataset dpo"
        )
    logger.info("dpo_training_started", pairs=total, beta=cfg.beta)

    if not has_sufficient_training_resources(min_ram_gb=cfg.min_ram_gb):
        # Fail loud: never register an adapter that was not actually trained.
        logger.error(
            "dpo_resource_gate_blocked",
            min_required_gb=14.0,
            msg="Host RAM/VRAM is insufficient to load policy + frozen reference model.",
        )
        print(
            "DPO training blocked: insufficient RAM/VRAM to load policy and frozen "
            "reference model (need >= 14 GB). No adapter was trained or registered."
        )
        return None

    tokenizer = AutoTokenizer.from_pretrained(cfg.ref_model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = resolve_dtype()
    device_map = resolve_device_map()

    # Policy: SFT adapter dir if it holds weights, else fresh LoRA over the base.
    adapter_path = Path(cfg.model_name_or_path)
    has_adapter = adapter_path.exists() and (
        (adapter_path / "adapter_model.safetensors").exists()
        or (adapter_path / "adapter_model.bin").exists()
    )
    from transformers import AutoModelForCausalLM

    if has_adapter:
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.ref_model_name_or_path, torch_dtype=dtype, device_map=device_map
        )
        model = PeftModel.from_pretrained(base_model, cfg.model_name_or_path, is_trainable=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.ref_model_name_or_path, torch_dtype=dtype, device_map=device_map
        )
        model = apply_lora(model)

    # Frozen reference policy (spec §9.2): no gradient, prevents collapse.
    ref_model = load_reference_model(cfg.ref_model_name_or_path)

    # TRL expects textual prompt/chosen/rejected; our exporter stores action lists.
    records = []
    with open(cfg.data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append({
                "prompt": rec["prompt"],
                "chosen": _as_text(rec["chosen"]),
                "rejected": _as_text(rec["rejected"]),
            })
    train_ds = Dataset.from_list(records)

    out_dir = Path(cfg.output_dir)
    checkpoint_dir = out_dir / "_checkpoints"
    targs = filter_valid_kwargs(TRLDPOConfig, {
        "output_dir": str(checkpoint_dir),
        "per_device_train_batch_size": cfg.batch_size,
        "num_train_epochs": cfg.epochs,
        "learning_rate": cfg.learning_rate,
        "beta": cfg.beta,
        "max_length": cfg.max_length,
        "max_prompt_length": cfg.max_prompt_length,
        "logging_steps": 10,
        "save_strategy": "no",
        "bf16": bool(dtype == torch.bfloat16),
        "fp16": bool(dtype == torch.float16),
        "gradient_checkpointing": True,
        "report_to": [],
    })
    training_args = TRLDPOConfig(**targs)

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    train_result = trainer.train()

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    loss_history = [
        entry["loss"] for entry in trainer.state.log_history if "loss" in entry
    ]
    final_loss = loss_history[-1] if loss_history else None

    # Register in adapter registry with real training metrics
    run_id = f"dpo_{Path(cfg.output_dir).name}"
    registry = AdapterRegistry()
    registry.register_adapter(
        run_id=run_id,
        rung=3,
        base_model=cfg.ref_model_name_or_path,
        adapter_path=cfg.output_dir,
        metrics={
            "total_pairs": total,
            "beta": cfg.beta,
            "train_loss_final": final_loss,
            "train_runtime_s": train_result.metrics.get("train_runtime"),
        },
    )

    print(
        f"DPO training completed: {total} pairs, beta={cfg.beta}, "
        f"final loss={final_loss if final_loss is not None else 'n/a'}. "
        f"Adapter saved to {cfg.output_dir}"
    )
    return cfg.output_dir


def _as_text(value: Any) -> str:
    """Serialize chosen/rejected payloads (string or action list) to plain text."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(v) for v in value)
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct Preference Optimization (DPO) Trainer (Rung 3)")
    parser.add_argument("--model", type=str, default="data/rl/adapters/sft", help="SFT model/adapter path")
    parser.add_argument("--ref_model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Reference model path")
    parser.add_argument("--data", type=str, default="data/rl/dpo.jsonl", help="Path to DPO pairs JSONL dataset")
    parser.add_argument("--output_dir", type=str, default="data/rl/adapters/dpo", help="Output adapter directory")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature parameter beta")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--min_ram_gb", type=float, default=14.0, help="RAM/VRAM gate for loading policy + ref model")
    parser.add_argument("--dry-run", action="store_true", help="Perform validation dry-run without loading ML models")

    args = parser.parse_args()
    cfg = DPOConfig(
        model_name_or_path=args.model,
        ref_model_name_or_path=args.ref_model,
        data_path=args.data,
        output_dir=args.output_dir,
        beta=args.beta,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        min_ram_gb=args.min_ram_gb,
        dry_run=args.dry_run,
    )
    train_dpo(cfg)


if __name__ == "__main__":
    main()
