"""
Reinforcement Learning (RL) Package for Self-Improving Browser Agent.

Exposes clean, lightweight interface for:
- Gymnasium Environment: BrowserGymEnv, SandboxBrowserBackend, SandboxViolationError, AgentAction, Observation
- Process Reward Model (PRM): dense_reward, ProcessRewardModel, RewardBreakdown
- Dataset Exporters: DatasetExporter, format_chat_messages
- Curriculum Engine: TaskCurriculumGenerator, CurriculumScheduler, CurriculumTask
- Rollout Server: RolloutSampler, RolloutTrajectory
- Adapter Registry: AdapterRegistry, AdapterRecord, get_current_production_adapter
- Training Entrypoints: train_sft, train_dpo, train_grpo, train_ppo (lazy loaded)
"""

from __future__ import annotations

from typing import Any

from app.rl.reward import (
    dense_reward,
    ProcessRewardModel,
    RewardBreakdown,
    is_fake_search_url,
    is_checkout_action,
)
from app.rl.env import (
    BrowserGymEnv,
    SandboxBrowserBackend,
    SandboxViolationError,
    LiveBrowserBackend,
    AgentAction,
    Observation,
)
from app.rl.dataset import (
    DatasetExporter,
    format_chat_messages,
)
from app.rl.curriculum import (
    TaskCurriculumGenerator,
    CurriculumScheduler,
    CurriculumTask,
)
from app.rl.rollout_server import (
    RolloutSampler,
    RolloutTrajectory,
)
from app.rl.registry import (
    AdapterRegistry,
    AdapterRecord,
    get_current_production_adapter,
)

__all__ = [
    "dense_reward",
    "ProcessRewardModel",
    "RewardBreakdown",
    "is_fake_search_url",
    "is_checkout_action",
    "BrowserGymEnv",
    "SandboxBrowserBackend",
    "LiveBrowserBackend",
    "SandboxViolationError",
    "AgentAction",
    "Observation",
    "DatasetExporter",
    "format_chat_messages",
    "TaskCurriculumGenerator",
    "CurriculumScheduler",
    "CurriculumTask",
    "RolloutSampler",
    "RolloutTrajectory",
    "AdapterRegistry",
    "AdapterRecord",
    "get_current_production_adapter",
    "train_sft",
    "SFTConfig",
    "train_dpo",
    "DPOConfig",
    "train_grpo",
    "GRPOConfig",
    "train_ppo",
    "PPOConfig",
]


def __getattr__(name: str) -> Any:
    """Lazy-load training modules on demand to prevent runpy module collisions."""
    if name == "train_sft":
        from app.rl.train_sft import train_sft
        return train_sft
    elif name == "SFTConfig":
        from app.rl.train_sft import SFTConfig
        return SFTConfig
    elif name == "train_dpo":
        from app.rl.train_dpo import train_dpo
        return train_dpo
    elif name == "DPOConfig":
        from app.rl.train_dpo import DPOConfig
        return DPOConfig
    elif name == "train_grpo":
        from app.rl.train_grpo import train_grpo
        return train_grpo
    elif name == "GRPOConfig":
        from app.rl.train_grpo import GRPOConfig
        return GRPOConfig
    elif name == "train_ppo":
        from app.rl.train_ppo import train_ppo
        return train_ppo
    elif name == "PPOConfig":
        from app.rl.train_ppo import PPOConfig
        return PPOConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
