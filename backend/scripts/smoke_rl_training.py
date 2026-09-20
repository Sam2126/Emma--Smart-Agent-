"""
Tiny-model smoke test for the REAL training paths (spec §14.4 GPU-smoke analog).

Builds a tiny random Llama + WordLevel tokenizer fully offline, then runs:
  1. SFT  (TRL SFTTrainer, LoRA, warm-start gate satisfied with 55 records)
  2. DPO  (TRL DPOTrainer, frozen reference model, pairs from SFT adapter)
  3. GRPO (sandbox rollouts -> group advantages -> policy-gradient + KL update)
  4. PPO  (sandbox rollouts -> reward normalization -> replay filter -> GAE
           -> clipped surrogate + value head + KL early stop)

Run from a temp CWD so adapter/registry writes stay isolated:
    cd <tempdir> && python <this script>
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

WORK = Path(tempfile.mkdtemp(prefix="rl_smoke_"))
MODEL_DIR = WORK / "tiny_model"
ADAPTERS = WORK / "adapters"
RESULTS = {}


def build_tiny_model() -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        PreTrainedTokenizerFast,
    )

    vocab_words = [
        "<unk>", "<pad>", "<s>", "</s>",
        "{", "}", "\"", ":", ",", "type", "selector", "text", "url",
        "click", "type_action", "navigate", "scroll", "press", "select",
        "noop", "headphones", "sony", "cart", "search", "filter", "goal",
        "page", "observation", "next", "action", "json", "only", "you",
        "are", "a", "browser", "automation", "policy", "assistant", "user",
        "system", "sandbox", "amazon", "in", "the", "add", "to",
    ]
    vocab = {w: i for i, w in enumerate(vocab_words)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.WordPiece(prefix="")
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>", pad_token="<pad>",
        bos_token="<s>", eos_token="</s>",
    )
    fast.chat_template = (
        "{% for m in messages %}{{ '<|' + m['role'] + '|>\\n' + m['content'] + '<|end|>\\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"
    )

    config = LlamaConfig(
        vocab_size=len(vocab_words),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = LlamaForCausalLM(config)
    model.save_pretrained(str(MODEL_DIR))
    fast.save_pretrained(str(MODEL_DIR))
    print(f"[setup] tiny model at {MODEL_DIR}")


def make_sft_dataset() -> Path:
    p = WORK / "sft.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for i in range(55):
            record = {
                "task_id": f"smoke_t{i}",
                "messages": [
                    {"role": "user", "content": "Goal: search for headphones\nNext action (JSON only):"},
                    {"role": "assistant", "content": '{ "type": "click", "selector": "#search", "text": "", "url": "" }'},
                ],
            }
            f.write(json.dumps(record) + "\n")
    return p


def make_dpo_dataset() -> Path:
    p = WORK / "dpo.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for i in range(6):
            record = {
                "task_id": f"smoke_d{i}",
                "prompt": f"Goal: search for headphones on amazon.in\nDomain: amazon.in",
                "chosen": 'click(#add-to-cart-button)',
                "rejected": 'click(#wrong-selector)',
            }
            f.write(json.dumps(record) + "\n")
    return p


def run(name: str, cmd: list[str]) -> None:
    """cmd is the full argument list after sys.executable."""
    import os
    import subprocess

    backend_root = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = backend_root + os.pathsep + env.get("PYTHONPATH", "")

    print(f"\n=== SMOKE {name} ===")
    proc = subprocess.run(
        [sys.executable] + cmd,
        cwd=str(WORK),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-6:])
    ok = proc.returncode == 0
    RESULTS[name] = ok
    print(("PASS" if ok else "FAIL") + f" (exit {proc.returncode})\n{tail}")


def main() -> int:
    build_tiny_model()
    sft_data = make_sft_dataset()
    dpo_data = make_dpo_dataset()

    sft_out = ADAPTERS / "sft"
    run("sft", ["-m", "app.rl.train_sft", "--model", str(MODEL_DIR), "--data", str(sft_data),
                "--output_dir", str(sft_out), "--epochs", "1", "--batch_size", "2",
                "--min_trajectories", "50"])

    dpo_out = ADAPTERS / "dpo"
    run("dpo", ["-m", "app.rl.train_dpo", "--model", str(sft_out), "--ref_model", str(MODEL_DIR),
                "--data", str(dpo_data), "--output_dir", str(dpo_out),
                "--epochs", "1", "--batch_size", "1", "--min_ram_gb", "1"])

    grpo_out = ADAPTERS / "grpo"
    run("grpo", ["-m", "app.rl.train_grpo", "--model", str(dpo_out), "--ref_model", str(MODEL_DIR),
                 "--output_dir", str(grpo_out), "--group_size", "4",
                 "--curriculum_size", "6", "--epochs", "1"])

    # PPO: a random tiny model emits unparseable output, which the spec §9.4
    # replay filter (correctly) drops as degenerate. To exercise the update
    # math deterministically, patch unparseable outputs into a scripted
    # policy that actually progresses through the fixture site.
    ppo_out = ADAPTERS / "ppo"
    ppo_boot = WORK / "_ppo_boot.py"
    ppo_boot.write_text(f"""
import sys
sys.argv = ['train_ppo',
    '--model', r'{grpo_out}', '--ref_model', r'{MODEL_DIR}',
    '--output_dir', r'{ppo_out}',
    '--epochs', '1', '--ppo_epochs', '1', '--batch_size', '3']

import app.rl.train_utils as tu

_orig = tu.parse_action_text
_seq = [
    {{"type": "type", "selector": "#twotabsearchtextbox", "text": "headphones"}},
    {{"type": "click", "selector": "a.product-link", "text": ""}},
    {{"type": "click", "selector": "#add-to-cart-button", "text": ""}},
    {{"type": "click", "selector": "#nav-cart", "text": ""}},
]
_n = [0]

def _patched(text):
    a = _orig(text)
    if a.get("type") == "noop":
        a = dict(_seq[_n[0] % 4])
        _n[0] += 1
    return a

tu.parse_action_text = _patched

from app.rl.train_ppo import main
main()
""", encoding="utf-8")
    run("ppo", [str(ppo_boot)])

    # Verify adapters actually persisted + registry recorded real runs
    print("\n=== ARTIFACTS ===")
    for name in ("sft", "dpo", "grpo", "ppo"):
        d = ADAPTERS / name
        has_adapter = (d / "adapter_model.safetensors").exists() or (d / "adapter_model.bin").exists()
        print(f"{name}: adapter_weights={'yes' if has_adapter else 'NO'}")
        RESULTS[f"{name}_artifact"] = has_adapter

    reg = WORK / "data" / "rl" / "adapter_registry.json"
    if reg.exists():
        data = json.loads(reg.read_text(encoding="utf-8"))
        entries = data.get("adapters", {})  # {run_id: AdapterRecord}
        print(f"registry: {len(entries)} runs, active={data.get('active_adapter_id')}")
        for run_id, rec in entries.items():
            print(f"  {run_id}: metrics={rec.get('metrics', {})}")
        RESULTS["registry"] = len(entries) >= 4
    else:
        RESULTS["registry"] = False
        print("registry: MISSING")

    failed = [k for k, v in RESULTS.items() if not v]
    print(f"\nSMOKE RESULT: {'ALL PASS' if not failed else 'FAILURES: ' + ', '.join(failed)}")
    print(f"workdir: {WORK}")
    if not failed:
        shutil.rmtree(WORK, ignore_errors=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
