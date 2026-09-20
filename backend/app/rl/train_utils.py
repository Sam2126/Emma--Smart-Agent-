"""
Shared training helpers for the RL trainers (Rungs 2-5).

All torch/transformers/peft imports are LAZY (inside functions) so that
`import app.rl.train_utils` stays safe on machines without the ML stack —
the dry-run contract (spec §9.0) depends on it.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Default chat template applied when a base model ships without one. Keeps
# apply_chat_template working for arbitrary small/local base models.
FALLBACK_CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{{ '<|' + m['role'] + '|>\\n' + m['content'] + '<|end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"
)

# Instructs the policy to emit exactly one machine-parseable browser action.
ACTION_SYSTEM_PROMPT = (
    "You are a browser automation policy. Given the goal and the current page "
    "observation, output exactly ONE next action as compact JSON on a single line:\n"
    '{"type": "click|type|select|press|scroll|navigate", "selector": "<css>", "text": "<input>", "url": "<target>"}\n'
    "Output the JSON object only — no prose, no markdown."
)

DEFAULT_LORA_KWARGS = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
}


def resolve_dtype():
    """Pick a training dtype that is valid on the available hardware."""
    import torch

    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32  # float16 on CPU is numerically unsafe for training


def resolve_device_map() -> str | None:
    try:
        import torch

        return "auto" if torch.cuda.is_available() else None
    except ImportError:  # pragma: no cover - torch missing is handled by callers
        return None


def filter_valid_kwargs(config_cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs that are fields of `config_cls` (version-tolerant TRL config)."""
    if dataclasses.is_dataclass(config_cls):
        valid = {f.name for f in dataclasses.fields(config_cls)}
    else:
        valid = set(getattr(config_cls, "__dataclass_fields__", {}).keys())
    dropped = sorted(set(kwargs) - valid)
    if dropped:
        logger.debug("config_kwargs_dropped", config=config_cls.__name__, dropped=dropped)
    return {k: v for k, v in kwargs.items() if k in valid}


def ensure_chat_template(tokenizer: Any) -> None:
    """Guarantee the tokenizer can render conversational `messages` records."""
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = FALLBACK_CHAT_TEMPLATE


def chat_template_ids(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool = True):
    """
    Tokenized chat prompt as a [1, seq_len] LongTensor. Handles both the old
    transformers API (returns tensor) and 5.x (returns BatchEncoding).
    """
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt, return_tensors="pt"
    )
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    return encoded


def build_action_messages(instruction: str, observation: dict[str, Any]) -> list[dict[str, str]]:
    """Compose the chat messages used to elicit one browser action from the policy."""
    obs_summary = {
        "url": observation.get("url", ""),
        "title": observation.get("title", ""),
        "elements": observation.get("elements", [])[:15],
        "milestones": observation.get("milestones", []),
    }
    user_content = (
        f"Goal: {instruction}\n\n"
        f"Page observation:\n{json.dumps(obs_summary, ensure_ascii=False)[:1200]}\n\n"
        "Next action (JSON only):"
    )
    return [
        {"role": "system", "content": ACTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_action_text(text: str) -> dict[str, Any]:
    """Extract the first JSON object from generated text as an action dict."""
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict) and obj.get("type"):
                return obj
        except json.JSONDecodeError:
            pass
    # Unparseable output: degenerate no-op step (per-step cost still applies,
    # so GRPO/PPO learn to emit parseable actions).
    return {"type": "noop", "selector": "", "text": text.strip()[:80]}


def apply_lora(model, lora_kwargs: dict[str, Any] | None = None):
    """Wrap a base causal LM in a new trainable LoRA adapter."""
    from peft import LoraConfig, get_peft_model, TaskType

    merged = dict(DEFAULT_LORA_KWARGS)
    merged.update(lora_kwargs or {})
    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, bias="none", **merged)
    return get_peft_model(model, lora_config)


def load_policy_with_lora(model_path: str, base_model: str, lora_kwargs: dict[str, Any] | None = None):
    """
    Load the trainable policy: full model if `model_path` is a model dir,
    PeftModel if it holds a LoRA adapter, otherwise the base model freshly
    wrapped in a new LoRA adapter. Returns (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    dtype = resolve_dtype()
    device_map = resolve_device_map()
    path = Path(model_path) if model_path else None

    tokenizer_source = model_path if (path and path.exists()) else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ensure_chat_template(tokenizer)

    has_adapter = bool(path and path.exists() and (
        (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()
    ))
    has_full_model = bool(path and path.exists() and (path / "config.json").exists())

    if has_adapter:
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, device_map=device_map)
        model = PeftModel.from_pretrained(base, model_path, is_trainable=True)
    elif has_full_model:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device_map)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, device_map=device_map)
        model = apply_lora(model, lora_kwargs)

    model.train()
    return model, tokenizer


def load_reference_model(model_name_or_path: str):
    """Load the frozen reference policy (no grad, eval mode)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=resolve_dtype(),
        device_map=resolve_device_map(),
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def response_token_mask(input_ids, prompt_lengths):
    """Boolean mask marking response tokens (everything after the prompt)."""
    import torch

    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for i, pl in enumerate(prompt_lengths):
        mask[i, pl:] = True
    return mask


def sequence_logprobs(model, input_ids, attention_mask):
    """
    Per-token log-probabilities of `input_ids` under `model`.
    Returns (token_logprobs, logits) where token_logprobs[i, t] is the log-prob
    of input_ids[i, t] given the prefix; position 0 has no prediction (0.0).
    """
    import torch

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    logprobs = torch.log_softmax(logits, dim=-1)
    token_logprobs = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    zero_col = torch.zeros(
        token_logprobs.size(0), 1, device=token_logprobs.device, dtype=token_logprobs.dtype
    )
    return torch.cat([zero_col, token_logprobs], dim=1), outputs.logits


def k3_kl(logp_policy, logp_reference):
    """
    Low-variance k3 KL estimator (Schulman): exp(lr) - lr - 1 with
    lr = logp_ref - logp_policy, computed over unmasked token positions.
    """
    import torch

    lr = logp_reference - logp_policy
    kl = torch.exp(lr) - lr - 1.0
    return kl
