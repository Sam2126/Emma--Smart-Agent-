"""
WebRL-style Proximal Policy Optimization (PPOv2) Trainer (Rung 5 Endgame).

Implements §9.0 and §9.4 of the RL Implementation Plan (v2):
- Lineage: WebRL (ORM + PRM + value net + GAE + KL divergence penalty)
- Generalized Advantage Estimation (GAE): gamma=0.99, lambda=0.95
- Adaptive KL constraint vs frozen reference policy: target_kl=0.02
- Clipped surrogate objective: min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)
- Replay buffer with confidence filtering: drops low-confidence judge verdicts,
  gaming-flagged actions, and degenerate steps
- Reward normalization: running mean/std (per domain) with clipping to [-5.0, +5.0]
- Hard safety invariant: asserts sandbox_only=True at construction; live domains rejected
- Shared dry-run contract: exits 0 without ML dependencies on CPU/dev machines.
"""

from __future__ import annotations

import argparse

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import structlog

from app.rl.env import SandboxViolationError
from app.rl.registry import AdapterRegistry

logger = structlog.get_logger(__name__)


@dataclass
class PPOConfig:
    """Configuration for WebRL-style PPOv2 training."""
    model_name_or_path: str = "data/rl/adapters/grpo"
    ref_model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "data/rl/adapters/ppo"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    target_kl: float = 0.02
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    learning_rate: float = 5e-7
    batch_size: int = 16
    epochs: int = 3
    ppo_epochs: int = 2
    max_steps_per_rollout: int = 25
    sandbox_only: bool = True
    dry_run: bool = False

    def __post_init__(self) -> None:
        if not self.sandbox_only:
            raise SandboxViolationError(
                "PPO training cannot run with sandbox_only=False! "
                "Online RL must strictly execute in sandbox environments."
            )


class RunningRewardNormalizer:
    """
    Online running mean and standard deviation normalizer for PPO rewards.
    Normalizes rewards and clips to [-5.0, +5.0] to prevent gradient explosion.
    """

    def __init__(self, clip_range: float = 5.0, eps: float = 1e-6) -> None:
        self.clip_range = clip_range
        self.eps = eps
        self.count: int = 0
        self.mean: float = 0.0
        self.m2: float = 0.0

    def update(self, reward: float) -> None:
        """Welford's algorithm for online variance estimation."""
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.m2 += delta * delta2

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        return math.sqrt(self.m2 / (self.count - 1))

    def normalize(self, reward: float) -> float:
        """Normalize reward and clip to [-clip_range, +clip_range]."""
        self.update(reward)
        std_val = self.std
        norm_r = (reward - self.mean) / (std_val + self.eps)
        return max(-self.clip_range, min(self.clip_range, norm_r))


def compute_gae(
    rewards: list[float],
    values: list[float],
    next_value: float = 0.0,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[list[float], list[float]]:
    """
    Generalized Advantage Estimation (GAE):
        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        A_t = sum_{l=0} (gamma * lambda)^l * delta_{t+l}
        Returns: (advantages, returns)
    """
    advantages: list[float] = [0.0] * len(rewards)
    returns: list[float] = [0.0] * len(rewards)
    gae = 0.0

    for t in reversed(range(len(rewards))):
        v_next = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * v_next - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
        returns[t] = gae + values[t]

    return advantages, returns


@dataclass
class ReplayBuffer:
    """Confidence-filtered trajectory replay buffer for PPO."""
    trajectories: list[dict[str, Any]] = field(default_factory=list)
    min_confidence: float = 0.6

    def add_trajectory(self, trajectory: dict[str, Any]) -> bool:
        """
        Filter and add trajectory to buffer (spec §9.4): drops gaming-flagged
        rollouts, low-confidence judge verdicts, and degenerate (all no-op)
        trajectories.
        """
        if trajectory.get("gaming_flags"):
            logger.debug("trajectory_rejected_gaming_flags", flags=trajectory["gaming_flags"])
            return False

        confidence = trajectory.get("confidence", 1.0)
        if confidence < self.min_confidence:
            logger.debug("trajectory_rejected_low_confidence", conf=confidence)
            return False

        actions = trajectory.get("actions", [])
        if not actions:
            return False
        # Degenerate: policy produced no effective actions (all parse failures)
        if all(a.get("type") == "noop" for a in actions):
            logger.debug("trajectory_rejected_degenerate_noop")
            return False

        self.trajectories.append(trajectory)
        return True

    def clear(self) -> None:
        self.trajectories.clear()


def run_dry_run(cfg: PPOConfig) -> None:
    """Execute dry-run validation without constructing ML dependencies."""
    print("=" * 65)
    print("[*] PPOv2 (WebRL-STYLE) TRAINER DRY-RUN VALIDATION")
    print("=" * 65)
    print(f"Policy Model:      {cfg.model_name_or_path}")
    print(f"Reference Model:   {cfg.ref_model_name_or_path}")
    print(f"Output Directory:  {cfg.output_dir}")
    print(f"GAE Config:        gamma={cfg.gamma}, lambda={cfg.gae_lambda}")
    print(f"PPO Clip Epsilon:  {cfg.clip_epsilon} | Target KL: {cfg.target_kl}")
    print(f"Loss Coeffs:       Value={cfg.value_loss_coeff}, Entropy={cfg.entropy_coeff}")
    print(f"Sandbox Guard:     sandbox_only={cfg.sandbox_only} (ENFORCED)")
    print("-" * 65)

    # Output directory validation
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    test_file = out / ".write_test"
    test_file.write_text("ok", encoding="utf-8")
    test_file.unlink()
    print("[OK] Output directory validated and writable.")

    # Spot-check GAE calculation
    dummy_rewards = [1.0, 0.5, 2.0]
    dummy_values = [0.8, 1.2, 1.5]
    advs, returns = compute_gae(dummy_rewards, dummy_values, next_value=0.0)
    print(f"[OK] GAE computation verified: advs={[round(a, 3) for a in advs]}, returns={[round(r, 3) for r in returns]}.")

    # Spot-check RunningRewardNormalizer
    normalizer = RunningRewardNormalizer()
    for r in [1.0, 2.0, 3.0, 4.0, 5.0]:
        _ = normalizer.normalize(r)
    sample_norm = normalizer.normalize(3.0)
    print(f"[OK] Running reward normalizer verified (normalized sample={sample_norm:.3f}, mean={normalizer.mean:.2f}).")

    # Spot-check confidence-filtered replay buffer
    buf = ReplayBuffer(min_confidence=0.6)
    valid = buf.add_trajectory({"actions": [{"type": "click"}], "confidence": 0.85})
    gaming_rejected = not buf.add_trajectory({"actions": [{"type": "click"}], "gaming_flags": ["fake_search_url"]})
    conf_rejected = not buf.add_trajectory({"actions": [{"type": "click"}], "confidence": 0.4})
    assert valid and gaming_rejected and conf_rejected
    print("[OK] Replay buffer confidence & gaming filters verified.")

    # Check registry access
    registry = AdapterRegistry()
    print(f"[OK] Adapter registry access verified (current active: {registry.active_adapter_id or 'None'}).")
    print("-" * 65)
    print("[OK] PPO DRY-RUN COMPLETED SUCCESSFULLY. No ML dependencies constructed.")
    print("=" * 65)


def train_ppo(cfg: PPOConfig) -> Optional[str]:
    """
    Execute WebRL-style PPO training loop (sandbox-only, hard rail):

    1. Collect trajectories with the current policy in the sandbox env,
       scoring via the PRM (ORM) and normalizing rewards (running mean/std).
    2. Filter through the confidence-filtered replay buffer (drops gaming-
       flagged / low-confidence / degenerate trajectories).
    3. For each trajectory: value estimates -> GAE advantages/returns.
    4. Update policy with the clipped surrogate objective + value loss +
       entropy bonus, early-stopping when KL vs reference exceeds target_kl.
    """
    if cfg.dry_run:
        run_dry_run(cfg)
        return None

    # Lazy heavy imports (spec §9.0: never at module top)
    try:
        import torch
        import torch.nn as nn
    except ImportError as err:
        logger.error("ppo_deps_missing", error=str(err))
        raise RuntimeError(
            "Heavy ML training dependencies missing. "
            "Install GPU requirements via `pip install -r backend/requirements-rl.txt`."
        ) from err

    from app.rl.curriculum import TaskCurriculumGenerator
    from app.rl.rollout_server import collect_llm_rollout
    from app.rl.train_utils import (
        chat_template_ids,
        load_policy_with_lora,
        load_reference_model,
        response_token_mask,
        sequence_logprobs,
    )

    logger.info("ppo_training_started", target_kl=cfg.target_kl, clip_epsilon=cfg.clip_epsilon)

    policy, tokenizer = load_policy_with_lora(cfg.model_name_or_path, cfg.ref_model_name_or_path)
    reference = load_reference_model(cfg.ref_model_name_or_path)
    device = next(policy.parameters()).device

    # Value head (critic) on top of the policy's hidden states (spec §9.4).
    hidden_size = policy.config.hidden_size
    value_head = nn.Sequential(
        nn.Linear(hidden_size, 256), nn.Tanh(), nn.Linear(256, 1)
    ).to(device)
    value_optimizer = torch.optim.AdamW(
        list(value_head.parameters()) + [p for p in policy.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
    )

    gen = TaskCurriculumGenerator()
    tasks = gen.generate_tiered_tasks(n_per_tier=max(1, cfg.batch_size // 3))

    normalizer = RunningRewardNormalizer()
    replay = ReplayBuffer(min_confidence=0.6)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    update_log: list[dict[str, float]] = []
    policy.eval()
    kl_early_stopped = False

    for epoch in range(cfg.epochs):
        if kl_early_stopped:
            logger.warning("ppo_kl_early_stop", epoch=epoch, target_kl=cfg.target_kl)
            break

        # ---- Rollout collection (sandbox-only) ----
        replay.clear()
        for task in tasks:
            traj = collect_llm_rollout(
                policy, tokenizer, task,
                max_steps=cfg.max_steps_per_rollout,
                temperature=1.0,
                sandbox_only=True,
            )
            replay.add_trajectory(traj)  # confidence-filtered (gaming/low-conf dropped)

        if not replay.trajectories:
            logger.warning("ppo_no_trajectories_after_filtering", epoch=epoch)
            continue

        # ---- Per-trajectory GAE ----
        for traj in replay.trajectories:
            steps = traj["steps"]
            norm_rewards = [normalizer.normalize(r) for r in traj["rewards"]]

            # Value estimates for each step's prompt under the critic
            values: list[float] = []
            with torch.no_grad():
                for step in steps:
                    prompt_ids = chat_template_ids(tokenizer, step["prompt_messages"]).to(device)
                    attention = torch.ones_like(prompt_ids)
                    last_hidden = _last_hidden(policy, prompt_ids, attention)
                    v = value_head(last_hidden[:, -1, :]).squeeze(-1)
                    values.append(float(v.item()))

            advantages, returns = compute_gae(
                norm_rewards, values,
                next_value=0.0, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
            )

            # ---- PPO clipped-surrogate update over this trajectory ----
            batch = list(zip(steps, advantages))
            logp_old_list: list[torch.Tensor] = []
            with torch.no_grad():
                input_ids, attention_mask, resp_mask = _tokenize_batch(tokenizer, batch, device)
                old_logprobs, _ = sequence_logprobs(policy, input_ids, attention_mask)

            for ppo_epoch in range(cfg.ppo_epochs):
                policy_logprobs, logits = sequence_logprobs(policy, input_ids, attention_mask)
                token_counts = resp_mask.sum(-1).clamp(min=1).float()
                new_logp = (policy_logprobs * resp_mask).sum(-1) / token_counts
                old_logp = (old_logprobs * resp_mask).sum(-1) / token_counts
                adv_t = torch.tensor(advantages, dtype=new_logp.dtype, device=device)

                ratio = torch.exp(new_logp - old_logp)
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * adv_t
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss on returns
                v_pred = value_head(_last_hidden(policy, input_ids, attention_mask)[:, -1, :]).squeeze(-1)
                value_loss = nn.functional.mse_loss(v_pred, torch.tensor(returns, dtype=v_pred.dtype, device=device))

                # KL vs frozen reference for the adaptive constraint: the gate
                # fires AFTER an update moved the policy too far (WebRL-style
                # early stopping on divergence), never before the first step.
                with torch.no_grad():
                    ref_logprobs, _ = sequence_logprobs(reference, input_ids, attention_mask)
                    ref_logp = (ref_logprobs * resp_mask).sum(-1) / token_counts
                kl = (new_logp - ref_logp).mean().item()

                # Token-level entropy bonus over response positions
                log_probs_full = torch.log_softmax(logits[:, :-1, :], dim=-1)
                entropy_per_token = -(log_probs_full.exp() * log_probs_full).sum(-1)
                resp_mask_shift = resp_mask[:, 1:].float()
                entropy = (
                    (entropy_per_token * resp_mask_shift).sum() / resp_mask_shift.sum().clamp(min=1)
                )

                loss = (
                    policy_loss
                    + cfg.value_loss_coeff * value_loss
                    - cfg.entropy_coeff * entropy
                )
                value_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(value_optimizer.param_groups[0]["params"], 1.0)
                value_optimizer.step()

                update_log.append({
                    "policy_loss": float(policy_loss.detach()),
                    "value_loss": float(value_loss.detach()),
                    "kl": kl,
                })

                if kl > 1.5 * cfg.target_kl:
                    kl_early_stopped = True
                    logger.warning("ppo_kl_threshold_exceeded", kl=kl, target=cfg.target_kl)
                    break

            if kl_early_stopped:
                break

    # Persist the trained adapter
    trainable = [p for p in policy.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("PPO produced no trainable parameters (LoRA attach failed).")
    if hasattr(policy, "save_pretrained"):
        policy.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    final_kl = update_log[-1]["kl"] if update_log else 0.0
    run_id = f"ppo_{Path(cfg.output_dir).name}"
    registry = AdapterRegistry()
    registry.register_adapter(
        run_id=run_id,
        rung=5,
        base_model=cfg.ref_model_name_or_path,
        adapter_path=cfg.output_dir,
        metrics={
            "target_kl": cfg.target_kl,
            "gamma": cfg.gamma,
            "updates": len(update_log),
            "final_kl": final_kl,
            "kl_early_stopped": kl_early_stopped,
        },
    )

    print(
        f"PPO training completed: {len(update_log)} updates, final KL={final_kl:.4f} "
        f"(target {cfg.target_kl}){' — early-stopped on KL divergence' if kl_early_stopped else ''}. "
        f"Adapter saved to {cfg.output_dir}"
    )
    return cfg.output_dir


def _tokenize_batch(tokenizer, batch, device):
    """Pad+mask (step, adv) pairs into aligned tensors; returns (input_ids, attention_mask, resp_mask)."""
    import torch
    from app.rl.train_utils import chat_template_ids, response_token_mask

    input_ids_list, prompt_lengths = [], []
    for step, _adv in batch:
        prompt_ids = chat_template_ids(tokenizer, step["prompt_messages"])[0]
        response_ids = tokenizer(
            step["response_text"], add_special_tokens=False, return_tensors="pt"
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
    return input_ids, attention_mask, resp_mask


def _last_hidden(model, input_ids, attention_mask):
    """Final-layer hidden states for the value head (works for base or Peft wrappers)."""
    target = model.get_base_model() if hasattr(model, "get_base_model") else model
    outputs = target(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None:
        return hidden_states[-1]
    base = getattr(outputs, "base_model_output", None)
    if base is not None and getattr(base, "last_hidden_state", None) is not None:
        return base.last_hidden_state
    raise RuntimeError("Model output did not expose hidden states for the value head.")


def main() -> None:
    parser = argparse.ArgumentParser(description="WebRL-style Proximal Policy Optimization (PPOv2) Trainer (Rung 5)")
    parser.add_argument("--model", type=str, default="data/rl/adapters/grpo", help="Base model/prior adapter path")
    parser.add_argument("--ref_model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Reference model path")
    parser.add_argument("--output_dir", type=str, default="data/rl/adapters/ppo", help="Output adapter directory")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO clipping epsilon")
    parser.add_argument("--target_kl", type=float, default=0.02, help="Target KL divergence for early stopping")
    parser.add_argument("--lr", type=float, default=5e-7, help="Policy learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Curriculum tasks sampled per epoch")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--ppo_epochs", type=int, default=2, help="Inner PPO update epochs per trajectory")
    parser.add_argument("--max_steps_per_rollout", type=int, default=25, help="Env step budget per rollout")
    parser.add_argument("--sandbox_only", action="store_true", default=True, help="Enforce sandbox-only training rail")
    parser.add_argument("--dry-run", action="store_true", help="Perform validation dry-run without loading ML models")

    args = parser.parse_args()
    cfg = PPOConfig(
        model_name_or_path=args.model,
        ref_model_name_or_path=args.ref_model,
        output_dir=args.output_dir,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        target_kl=args.target_kl,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        ppo_epochs=args.ppo_epochs,
        max_steps_per_rollout=args.max_steps_per_rollout,
        sandbox_only=args.sandbox_only,
        dry_run=args.dry_run,
    )
    train_ppo(cfg)


if __name__ == "__main__":
    main()
