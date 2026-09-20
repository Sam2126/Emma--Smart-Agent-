"""
Async Rollout Server and Sampler for Online RL (GRPO / PPO).

Batches rollouts across tasks in hermetic sandbox environments,
computes step rewards via ProcessRewardModel, and yields rollout trajectories.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import structlog

from app.rl.env import BrowserGymEnv, AgentAction, Observation
from app.rl.curriculum import CurriculumTask

logger = structlog.get_logger(__name__)


@dataclass
class RolloutTrajectory:
    """A single completed rollout trajectory."""
    task_id: str
    instruction: str
    domain: str
    actions: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    rewards: list[float]
    total_reward: float
    success: bool
    terminated: bool
    truncated: bool
    gaming_flags: list[str] = field(default_factory=list)


class RolloutSampler:
    """
    Rollout generator executing tasks in parallel or sequential sandbox environments.
    """

    def __init__(
        self,
        sandbox_only: bool = True,
        max_steps: int = 25,
        policy_fn: Callable[[Observation, str], dict[str, Any]] | None = None,
    ) -> None:
        self.sandbox_only = sandbox_only
        self.max_steps = max_steps
        self.policy_fn = policy_fn or self._default_heuristic_policy

    @staticmethod
    def _default_heuristic_policy(obs: Observation, task: str) -> dict[str, Any]:
        """
        Baseline heuristic policy for sandbox environment rollout simulation.
        Translates task intent into sandbox actions.
        """
        task_lower = task.lower()
        url = obs.url.lower()

        if "sandbox://ecommerce/" == url:
            # On home page -> search
            query = "headphones"
            for p in ["headphones", "smartwatch", "laptop", "shoes"]:
                if p in task_lower:
                    query = p
                    break
            return {
                "type": "type",
                "selector": "#twotabsearchtextbox",
                "text": query,
            }

        elif "/s?k=" in url:
            if ("filter" in task_lower or "facet" in task_lower or "sony" in task_lower) and "filter=applied" not in url:
                return {
                    "type": "click",
                    "selector": "button:has-text('Sony')",
                    "text": "Sony",
                }
            else:
                return {
                    "type": "click",
                    "selector": "a.product-link:has-text('Sony WH-1000XM5')",
                    "text": "Sony WH-1000XM5",
                }

        elif "/dp/" in url:
            return {
                "type": "click",
                "selector": "#add-to-cart-button",
                "text": "Add to Cart",
            }

        return {"type": "click", "selector": "#nav-cart", "text": "Cart"}

    async def sample_rollout(self, task: CurriculumTask | str) -> RolloutTrajectory:
        """Sample a single rollout trajectory for a task."""
        env = BrowserGymEnv(sandbox_only=self.sandbox_only, max_steps=self.max_steps)
        task_desc = task.instruction if isinstance(task, CurriculumTask) else str(task)
        task_id = task.task_id if isinstance(task, CurriculumTask) else "rollout_task"
        domain = task.domain if isinstance(task, CurriculumTask) else "sandbox://ecommerce"

        obs, info = env.reset({"instruction": task_desc, "domain": domain})
        actions: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = [obs.to_dict()]
        rewards: list[float] = []
        gaming_flags: list[str] = []

        terminated = False
        truncated = False

        while not (terminated or truncated):
            action_dict = self.policy_fn(obs, task_desc)
            actions.append(action_dict)

            obs, reward, terminated, truncated, step_info = env.step(action_dict)
            observations.append(obs.to_dict())
            rewards.append(reward)

            if step_info.get("gaming_flags"):
                gaming_flags.extend(step_info["gaming_flags"])

        return RolloutTrajectory(
            task_id=task_id,
            instruction=task_desc,
            domain=domain,
            actions=actions,
            observations=observations,
            rewards=rewards,
            total_reward=sum(rewards),
            success=env.task_success,
            terminated=terminated,
            truncated=truncated,
            gaming_flags=gaming_flags,
        )

    async def sample_group_rollouts(
        self, task: CurriculumTask | str, group_size: int = 4
    ) -> list[RolloutTrajectory]:
        """Sample G parallel rollouts for a single task."""
        tasks = [self.sample_rollout(task) for _ in range(group_size)]
        return await asyncio.gather(*tasks)


# =============================================================================
# LLM-policy rollout engine (shared by train_grpo / train_ppo real loops)
# =============================================================================

# Per-step record consumed by the GRPO/PPO update passes.
# prompt_messages: chat messages shown to the policy; response_text: sampled action;
# reward: dense step reward from the env's PRM.
StepRecordRL = dict[str, Any]


def collect_llm_rollout(
    model: Any,
    tokenizer: Any,
    task: CurriculumTask | str,
    max_steps: int = 25,
    temperature: float = 0.7,
    max_new_tokens: int = 48,
    sandbox_only: bool = True,
    do_sample: bool = True,
) -> dict[str, Any]:
    """
    Run one rollout of an LLM policy in the sandbox BrowserGymEnv.

    The policy sees the observation and emits one JSON action per step; the env
    scores each step via the ProcessRewardModel (Tier-1 rules only, spec §6.3).
    Returns a trajectory dict with per-step (prompt, response, reward) records
    ready for policy-gradient updates.
    """
    import torch
    from app.rl.env import BrowserGymEnv
    from app.rl.train_utils import build_action_messages, chat_template_ids, parse_action_text

    env = BrowserGymEnv(sandbox_only=sandbox_only, max_steps=max_steps)
    task_desc = task.instruction if isinstance(task, CurriculumTask) else str(task)
    task_id = task.task_id if isinstance(task, CurriculumTask) else "rollout_task"
    domain = task.domain if isinstance(task, CurriculumTask) else "sandbox://ecommerce"

    obs, _info = env.reset({"instruction": task_desc, "domain": domain})

    steps: list[StepRecordRL] = []
    gaming_flags: list[str] = []
    parseable_count = 0
    was_terminated = False
    was_truncated = False

    while not (env.terminated or env.truncated):
        messages = build_action_messages(task_desc, obs.to_dict())
        prompt_ids = chat_template_ids(tokenizer, messages).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                prompt_ids,
                do_sample=do_sample and temperature > 0,
                temperature=temperature if do_sample else None,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        response_ids = output_ids[0, prompt_ids.shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        action_dict = parse_action_text(response_text)
        if action_dict.get("type") != "noop":
            parseable_count += 1

        obs, reward, was_terminated, was_truncated, step_info = env.step(action_dict)
        if step_info.get("gaming_flags"):
            gaming_flags.extend(step_info["gaming_flags"])

        steps.append({
            "prompt_messages": messages,
            "response_text": response_text,
            "action": action_dict,
            "reward": float(reward),
        })

    confidence = 1.0  # sandbox rollouts use Tier-1 rules only (no LLM judge in the loop)
    return {
        "task_id": task_id,
        "instruction": task_desc,
        "domain": domain,
        "steps": steps,
        "actions": [s["action"] for s in steps],
        "rewards": [s["reward"] for s in steps],
        "total_reward": sum(s["reward"] for s in steps),
        "success": env.task_success,
        "terminated": was_terminated,
        "truncated": was_truncated,
        "gaming_flags": gaming_flags,
        "confidence": confidence,
        # Diagnostic: share of steps where the policy emitted parseable JSON
        "parseable_ratio": (parseable_count / len(steps)) if steps else 0.0,
    }


def collect_group_rollouts(
    model: Any,
    tokenizer: Any,
    task: CurriculumTask | str,
    group_size: int,
    max_steps: int = 25,
    temperature: float = 0.7,
    sandbox_only: bool = True,
) -> list[dict[str, Any]]:
    """Sample G rollouts for one task (sync; env is in-process and fast)."""
    return [
        collect_llm_rollout(
            model, tokenizer, task,
            max_steps=max_steps, temperature=temperature, sandbox_only=sandbox_only,
        )
        for _ in range(group_size)
    ]
