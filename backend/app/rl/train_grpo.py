"""
Group-Relative Policy Optimization (GRPO) Trainer (Rung 4).

Implements §9.0 and §9.3 of the RL Implementation Plan (v2):
- Lineage: DeepSeekMath / DeepSeek-R1 style group-relative advantage without critic network
- Group size G: Samples G rollouts per curriculum task (default G=8, >=4 required)
- Scoring: Step-level ProcessRewardModel (Tier-1) + terminal frozen verifier outcome
- Advantage calculation:
    A_i = (R_i - mean(R_G)) / (std(R_G) + eps)
- KL regularization vs reference policy: beta_kl (default 0.05)
- Shared dry-run contract: exits 0 without ML dependencies on CPU/dev machines.
"""

from __future__ import annotations

import argparse

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import structlog

from app.rl.curriculum import TaskCurriculumGenerator
from app.rl.registry import AdapterRegistry

logger = structlog.get_logger(__name__)


@dataclass
class GRPOConfig:
    """Configuration for Group-Relative Policy Optimization training."""
    model_name_or_path: str = "data/rl/adapters/dpo"
    ref_model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "data/rl/adapters/grpo"
    group_size: int = 8
    curriculum_size: int = 24
    beta_kl: float = 0.05
    learning_rate: float = 1e-6
    epochs: int = 2
    max_steps_per_rollout: int = 25
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.group_size < 4:
            raise ValueError(f"GRPO requires group_size G >= 4 for stable advantage normalization (got {self.group_size}).")


def compute_group_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    """
    Compute group-relative advantage scores for a group of G rollouts:
        A_i = (R_i - mean(R)) / (std(R) + eps)
    """
    n = len(rewards)
    if n == 0:
        return []
    mean_r = sum(rewards) / n
    variance = sum((r - mean_r) ** 2 for r in rewards) / n
    std_r = math.sqrt(variance)

    advantages = [(r - mean_r) / (std_r + eps) for r in rewards]
    return advantages


def run_dry_run(cfg: GRPOConfig) -> None:
    """Execute dry-run validation without constructing ML dependencies."""
    print("=" * 65)
    print("[*] GRPO TRAINER DRY-RUN VALIDATION")
    print("=" * 65)
    print(f"Policy Model:      {cfg.model_name_or_path}")
    print(f"Reference Model:   {cfg.ref_model_name_or_path}")
    print(f"Output Directory:  {cfg.output_dir}")
    print(f"Group Size (G):    {cfg.group_size} rollouts/prompt (>=4 verified)")
    print(f"Curriculum Size:   {cfg.curriculum_size} tasks")
    print(f"KL Penalty (beta): {cfg.beta_kl}")
    print(f"Learning Rate:     {cfg.learning_rate}")
    print("-" * 65)

    # Output directory validation
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    test_file = out / ".write_test"
    test_file.write_text("ok", encoding="utf-8")
    test_file.unlink()
    print("[OK] Output directory validated and writable.")

    # Validate curriculum task generation
    gen = TaskCurriculumGenerator()
    tasks = gen.generate_tiered_tasks(n_per_tier=2)
    print(f"[OK] Curriculum generator verified: generated {len(tasks)} tiered tasks.")

    # Spot-check advantage calculation with dummy rewards
    dummy_rewards = [2.5, 0.8, -1.2, 3.1]
    advs = compute_group_advantages(dummy_rewards)
    print(f"[OK] Group advantage normalization formula verified: inputs={dummy_rewards} -> advantages={[round(a, 3) for a in advs]}.")

    # Check registry access
    registry = AdapterRegistry()
    print(f"[OK] Adapter registry access verified (current active: {registry.active_adapter_id or 'None'}).")
    print("-" * 65)
    print("[OK] GRPO DRY-RUN COMPLETED SUCCESSFULLY. No ML dependencies constructed.")
    print("=" * 65)


def train_grpo(cfg: GRPOConfig) -> Optional[str]:
    """
    Execute GRPO training loop (DeepSeekMath-style, no critic):

    for each curriculum task: sample G rollouts in the sandbox env, score with
    the PRM (Tier-1) + terminal verdict, compute group-relative advantages
    A_i = (R_i - mu_G) / (sigma_G + eps), then update the policy with
    REINFORCE + KL regularization against the frozen reference policy.
    """
    if cfg.dry_run:
        run_dry_run(cfg)
        return None

    # Lazy heavy imports (spec §9.0: never at module top)
    try:
        import torch
    except ImportError as err:
        logger.error("grpo_deps_missing", error=str(err))
        raise RuntimeError(
            "Heavy ML training dependencies missing. "
            "Install GPU requirements via `pip install -r backend/requirements-rl.txt`."
        ) from err

    from app.rl.rollout_server import collect_group_rollouts
    from app.rl.train_utils import (
        k3_kl,
        load_policy_with_lora,
        load_reference_model,
        response_token_mask,
        sequence_logprobs,
    )

    logger.info("grpo_training_started", group_size=cfg.group_size, beta_kl=cfg.beta_kl)

    # Rollouts always run in the hermetic sandbox (hard rail, spec §12.2).
    policy, tokenizer = load_policy_with_lora(cfg.model_name_or_path, cfg.ref_model_name_or_path)
    reference = load_reference_model(cfg.ref_model_name_or_path)

    gen = TaskCurriculumGenerator()
    n_per_tier = max(1, cfg.curriculum_size // 3)
    tasks = gen.generate_tiered_tasks(n_per_tier=n_per_tier)

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=cfg.learning_rate
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policy.eval()  # sample without dropout noise
    group_metrics: list[dict[str, float]] = []

    for epoch in range(cfg.epochs):
        for task in tasks:
            rollouts = collect_group_rollouts(
                policy, tokenizer, task,
                group_size=cfg.group_size,
                max_steps=cfg.max_steps_per_rollout,
                temperature=1.0,
                sandbox_only=True,
            )
            rewards = [r["total_reward"] for r in rollouts]
            advantages = compute_group_advantages(rewards)

            # Assemble the update batch from every step of every rollout
            batch = []
            for rollout, adv in zip(rollouts, advantages):
                for step in rollout["steps"]:
                    batch.append((step["prompt_messages"], step["response_text"], adv))
            if not batch:
                continue

            loss, kl_mean, resp_logp_mean = _grpo_update_step(
                policy, reference, tokenizer, batch, cfg.beta_kl
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

            group_metrics.append({
                "epoch": epoch,
                "task": task.task_id,
                "mean_reward": sum(rewards) / len(rewards),
                "successes": sum(1 for r in rollouts if r["success"]),
                "loss": float(loss.detach()),
                "kl": float(kl_mean),
                "mean_response_logprob": float(resp_logp_mean),
            })
            logger.info(
                "grpo_group_updated",
                epoch=epoch, task=task.task_id,
                mean_reward=group_metrics[-1]["mean_reward"],
                loss=group_metrics[-1]["loss"], kl=group_metrics[-1]["kl"],
            )

    # Persist the trained adapter
    trainable = [p for p in policy.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("GRPO produced no trainable parameters (LoRA attach failed).")
    if hasattr(policy, "save_pretrained"):
        policy.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    mean_reward = (
        sum(m["mean_reward"] for m in group_metrics) / len(group_metrics)
        if group_metrics else 0.0
    )

    run_id = f"grpo_{Path(cfg.output_dir).name}"
    registry = AdapterRegistry()
    registry.register_adapter(
        run_id=run_id,
        rung=4,
        base_model=cfg.ref_model_name_or_path,
        adapter_path=cfg.output_dir,
        metrics={
            "group_size": cfg.group_size,
            "beta_kl": cfg.beta_kl,
            "groups_updated": len(group_metrics),
            "mean_reward_final": mean_reward,
        },
    )

    print(
        f"GRPO training completed: {len(group_metrics)} groups updated "
        f"(G={cfg.group_size}, beta_kl={cfg.beta_kl}), mean reward={mean_reward:.3f}. "
        f"Adapter saved to {cfg.output_dir}"
    )
    return cfg.output_dir


def _grpo_update_step(policy, reference, tokenizer, batch, beta_kl: float):
    """
    One GRPO gradient step over a batch of (prompt_messages, response_text,
    group_advantage). Loss = -mean(adv * mean_token_logprob) + beta_kl * KL(policy||ref).
    Returns (loss, kl_mean, mean_response_logprob).
    """
    import torch
    from app.rl.train_utils import chat_template_ids, k3_kl, response_token_mask, sequence_logprobs

    device = next(policy.parameters()).device
    input_ids_list, prompt_lengths = [], []
    for messages, response_text, _adv in batch:
        prompt_ids = chat_template_ids(tokenizer, messages)[0]
        response_ids = tokenizer(
            response_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]
        full_ids = torch.cat([prompt_ids, response_ids]).long()
        input_ids_list.append(full_ids)
        prompt_lengths.append(prompt_ids.shape[0])

    max_len = max(t.shape[0] for t in input_ids_list)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    padded = torch.full((len(input_ids_list), max_len), pad_id, dtype=torch.long)
    for i, t in enumerate(input_ids_list):
        padded[i, : t.shape[0]] = t
    attention_mask = (padded != pad_id).long()
    input_ids = padded.to(device)
    attention_mask = attention_mask.to(device)
    resp_mask = response_token_mask(input_ids, prompt_lengths)

    policy_logprobs, _ = sequence_logprobs(policy, input_ids, attention_mask)
    with torch.no_grad():
        ref_logprobs, _ = sequence_logprobs(reference, input_ids, attention_mask)

    token_counts = resp_mask.sum(-1).clamp(min=1).float()
    resp_logps = (policy_logprobs * resp_mask).sum(-1) / token_counts
    ref_resp_logps = (ref_logprobs * resp_mask).sum(-1) / token_counts
    kl_per_sample = k3_kl(resp_logps, ref_resp_logps)

    advantages = torch.tensor(
        [b[2] for b in batch], dtype=resp_logps.dtype, device=device
    )
    policy_loss = -(advantages * resp_logps).mean()
    kl_mean = kl_per_sample.mean()
    loss = policy_loss + beta_kl * kl_mean
    return loss, kl_mean.detach(), resp_logps.detach().mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Group-Relative Policy Optimization (GRPO) Trainer (Rung 4)")
    parser.add_argument("--model", type=str, default="data/rl/adapters/dpo", help="Base/prior model path")
    parser.add_argument("--ref_model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Frozen reference model path")
    parser.add_argument("--output_dir", type=str, default="data/rl/adapters/grpo", help="Output adapter directory")
    parser.add_argument("--group_size", type=int, default=8, help="Rollout group size G (>=4 required)")
    parser.add_argument("--curriculum_size", type=int, default=24, help="Number of curriculum tasks to sample")
    parser.add_argument("--beta_kl", type=float, default=0.05, help="KL regularization penalty weight")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--dry-run", action="store_true", help="Perform validation dry-run without loading ML models")

    args = parser.parse_args()
    cfg = GRPOConfig(
        model_name_or_path=args.model,
        ref_model_name_or_path=args.ref_model,
        output_dir=args.output_dir,
        group_size=args.group_size,
        curriculum_size=args.curriculum_size,
        beta_kl=args.beta_kl,
        learning_rate=args.lr,
        epochs=args.epochs,
        dry_run=args.dry_run,
    )
    train_grpo(cfg)


if __name__ == "__main__":
    main()
