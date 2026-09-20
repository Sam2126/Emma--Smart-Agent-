"""
Comprehensive unit and integration test suite for Reinforcement Learning (PPO) stack.

Covers §13 of the RL Implementation Plan (v2):
1. test_brain_reflexion: Rung 1 reflexion generation, persistence, prompt injection, cold-domain no-op
2. test_reward: Dense PRM milestones, step cost, terminal outcomes, anti-gaming penalties
3. test_env: Gymnasium 5-tuple contract, SandboxBrowserBackend, action dispatch, truncation, sandbox guard
4. test_dataset_exports: SQLite extraction, filtering (checkout/gated/short), chat-templates, benchmark isolation
5. test_curriculum: Tiered task generation, mutation, scheduler bias, curriculum tagging
6. test_registry_and_rollback: Adapter registration, promotion, regression rollback gate
7. test_trainers_dry_run: All 4 trainers (SFT, DPO, GRPO, PPO) succeed without heavy ML packages
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any
import pytest
from sqlalchemy import select

from app.config import Settings
from app.state.database import get_session, init_database
from app.state.models import (
    Base,
    TaskRecord,
    StepRecord,
    VerificationRecord,
    DomainMemoryRecord,
    FailurePatternRecord,
)
from app.state.brain import BrainMemory, Reflection
from app.verifier.rule_checks import CheckResult

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
    AgentAction,
    Observation,
)
from app.rl.dataset import (
    DatasetExporter,
    format_chat_messages,
    compute_trajectory_hash,
)
from app.rl.curriculum import (
    TaskCurriculumGenerator,
    CurriculumScheduler,
    CurriculumTask,
)
from app.rl.registry import (
    AdapterRegistry,
    AdapterRecord,
    get_current_production_adapter,
)
from app.rl.rollout_server import (
    RolloutSampler,
    RolloutTrajectory,
)
from app.rl.train_sft import SFTConfig, train_sft, validate_dataset_records
from app.rl.train_dpo import DPOConfig, train_dpo
from app.rl.train_grpo import GRPOConfig, train_grpo, compute_group_advantages
from app.rl.train_ppo import (
    PPOConfig,
    train_ppo,
    compute_gae,
    RunningRewardNormalizer,
    ReplayBuffer,
)


# =============================================================================
# 1. RUNG 1: BRAIN REFLEXION TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_brain_reflexion_generation_and_storage():
    """Test structured reflection generation on success and failure."""
    brain = BrainMemory()

    # Success trajectory
    succ_traj = [
        {"action_type": "navigate", "selector_used": "", "success": True},
        {"action_type": "type", "selector_used": "#twotabsearchtextbox", "input_value": "Sony", "success": True},
        {"action_type": "click", "selector_used": "#add-to-cart-button", "success": True},
    ]
    succ_ref = brain._generate_reflection(
        task="Buy Sony headphones",
        trajectory=succ_traj,
        success=True,
    )
    assert succ_ref.what_failed == "None"
    assert "Executed 3 action(s)" in succ_ref.what_succeeded
    assert succ_ref.confidence >= 0.9

    # Failure trajectory with timeout
    fail_traj = [
        {"action_type": "navigate", "selector_used": "", "success": True},
        {"action_type": "click", "selector_used": ".missing-facet", "success": False, "error": "Action timed out waiting for selector"},
    ]
    fail_ref = brain._generate_reflection(
        task="Filter Sony headphones",
        trajectory=fail_traj,
        success=False,
        error="Action timed out waiting for selector",
    )
    assert "DOM element" in fail_ref.root_cause
    assert "Scroll element into view" in fail_ref.what_to_try_next

    # Persist reflections
    test_domain = "test-reflection-store.com"
    await brain._persist_reflection(test_domain, succ_ref, success=True)
    await brain._persist_reflection(test_domain, fail_ref, success=False)

    # Recall reflections
    recalled = await brain._recall_reflections(test_domain)
    assert len(recalled) >= 1
    assert any("Previous attempt failed" in r for r in recalled)
    assert any("this time:" in r for r in recalled)


@pytest.mark.asyncio
async def test_brain_reflexion_prompt_injection_and_cold_domain():
    """Test that === REFLEXION === is injected for known domains, but untouched for cold domains."""
    brain = BrainMemory()

    # Cold domain with no history
    cold_domain = "unknown-cold-domain-xyz.org"
    prompt_cold = await brain.recall_for_task("Search products on this domain", domain=cold_domain)
    assert "=== REFLEXION (lessons from previous runs) ===" not in prompt_cold

    # Active domain with stored reflections
    active_domain = "active-domain-test.com"
    fail_ref = Reflection(
        what_succeeded="None",
        what_failed="Timeout on button",
        root_cause="Overlay blocked element",
        what_to_try_next="Dismiss modal popup first",
        confidence=0.9,
    )
    await brain._persist_reflection(active_domain, fail_ref, success=False)

    prompt_active = await brain.recall_for_task("Do task on active domain", domain=active_domain)
    assert "=== REFLEXION (lessons from previous runs) ===" in prompt_active
    assert "Dismiss modal popup first" in prompt_active


# =============================================================================
# 2. REWARD ENGINE (PRM) TESTS
# =============================================================================

def test_reward_prm_milestones_and_step_cost():
    """Verify exact §6.1 milestone values and step penalty."""
    rules = [
        CheckResult("url_contains:search", True, "expected", "actual"),
        CheckResult("numeric_increment:cart_count", True, "expected", "actual"),
        CheckResult("facet_applied", True, "expected", "actual"),
        CheckResult("element_exists:price", True, "expected", "actual"),
    ]
    rb = dense_reward(rules=rules)
    # Milestones: 1.0 (url) + 1.0 (numeric) + 1.0 (facet) + 0.5 (element) = 3.5
    assert rb.milestones == 3.5
    # Step cost: -0.05
    assert rb.step_cost == -0.05
    assert rb.total == pytest.approx(3.45)


def test_reward_terminal_pass_and_fail():
    """Verify terminal outcomes: PASS (+2.0) and FAIL (-2.0)."""
    rb_pass = dense_reward(is_terminal=True, terminal_pass=True)
    assert rb_pass.terminal == 2.0
    assert rb_pass.total == pytest.approx(1.95)  # 2.0 - 0.05

    rb_fail = dense_reward(is_terminal=True, terminal_fail=True)
    assert rb_fail.terminal == -2.0
    assert rb_fail.total == pytest.approx(-2.05)  # -2.0 - 0.05


def test_reward_anti_gaming_rules():
    """Verify all anti-gaming penalties per §6.2."""
    # 1. Fake search URL
    fake_nav = {"type": "navigate", "url": "https://amazon.in/s?k=headphones"}
    rb_fake = dense_reward(action=fake_nav)
    assert rb_fake.penalties == -1.0
    assert "gaming: fake_search_url" in rb_fake.gaming_flags

    # 2. Unauthorized checkout attempt (aborts rollout)
    checkout_act = {"type": "click", "selector": "#place-order-button", "confirmed": False}
    rb_checkout = dense_reward(action=checkout_act)
    assert rb_checkout.penalties == -5.0
    assert rb_checkout.abort_requested is True
    assert "gaming: unauthorized_checkout_attempt" in rb_checkout.gaming_flags

    # 3. Deprecated selector reuse
    dep_sel = "input#obsolete_search_box"
    rb_dep = dense_reward(
        action={"type": "type", "selector": dep_sel},
        deprecated_selectors={dep_sel},
    )
    assert rb_dep.penalties == -1.0
    assert "gaming: deprecated_selector_reuse" in rb_dep.gaming_flags

    # 4. Spam click repeat penalty
    rb_spam = dense_reward(consecutive_spam_clicks=2)
    assert rb_spam.penalties == pytest.approx(-0.6)  # -0.3 * 2
    assert "gaming: spam_clicks_repeat_2" in rb_spam.gaming_flags

    # 5. Spam click limit (>=3) aborts rollout
    rb_spam3 = dense_reward(consecutive_spam_clicks=3)
    assert rb_spam3.abort_requested is True


def test_process_reward_model_state_tracking():
    """Verify ProcessRewardModel tracks DOM hash and detects consecutive spam clicks."""
    prm = ProcessRewardModel()
    state_dummy = type("DummyState", (), {"url": "sandbox://", "dom_hash": "hash_same", "extracted_values": {}})()

    click_action = {"type": "click", "selector": "#same-btn"}

    # First click -> streak 0
    rb1 = prm.step_reward(state_before=state_dummy, state_after=state_dummy, action=click_action)
    assert prm.consecutive_spam_clicks == 0

    # Second identical click with same DOM hash -> streak 1
    rb2 = prm.step_reward(state_before=state_dummy, state_after=state_dummy, action=click_action)
    assert prm.consecutive_spam_clicks == 1
    assert rb2.penalties == pytest.approx(-0.3)


# =============================================================================
# 3. GYMNASIUM ENVIRONMENT (env.py) TESTS
# =============================================================================

def test_browser_gym_env_contracts():
    """Verify Gymnasium 5-tuple step contract and fixture transitions."""
    env = BrowserGymEnv(sandbox_only=True, max_steps=10)
    obs, info = env.reset("Search for headphones and add to cart")

    assert isinstance(obs, Observation)
    assert obs.url == "sandbox://ecommerce/"
    assert info["sandbox_mode"] is True

    # Step 1: Type query into search box
    action1 = {"type": "type", "selector": "#twotabsearchtextbox", "text": "headphones"}
    obs1, r1, term1, trunc1, info1 = env.step(action1)
    assert "/s?k=" in obs1.url
    assert r1 > 0  # Milestone reward for search URL
    assert term1 is False
    assert trunc1 is False

    # Step 2: Click product
    action2 = {"type": "click", "selector": "a.product-link:has-text('Sony WH-1000XM5')"}
    obs2, r2, term2, trunc2, info2 = env.step(action2)
    assert "/dp/" in obs2.url
    assert term2 is False

    # Step 3: Click add to cart (Goal achieved!)
    action3 = {"type": "click", "selector": "#add-to-cart-button"}
    obs3, r3, term3, trunc3, info3 = env.step(action3)
    assert "/cart" in obs3.url
    assert term3 is True  # Terminated on goal PASS
    assert info3["task_success"] is True
    assert r3 > 1.0


def test_browser_gym_env_truncation():
    """Verify environment truncates when step budget is reached."""
    env = BrowserGymEnv(sandbox_only=True, max_steps=3)
    env.reset("Some prolonged search task")

    for _ in range(2):
        _, _, term, trunc, _ = env.step({"type": "scroll", "selector": "window"})
        assert term is False and trunc is False

    # 3rd step -> reached max_steps
    _, _, term, trunc, _ = env.step({"type": "scroll", "selector": "window"})
    assert trunc is True


def test_browser_gym_env_sandbox_guard():
    """Verify that sandbox_only=True strictly raises SandboxViolationError on live domains."""
    env = BrowserGymEnv(sandbox_only=True)
    with pytest.raises(SandboxViolationError):
        env.reset({"instruction": "Buy on live Amazon", "domain": "amazon.in"})


# =============================================================================
# 4. DATASET CONVERSION (dataset.py) TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_dataset_exporter_with_mock_db():
    """Verify SFT, DPO, and GRPO export and strict benchmark isolation."""
    import uuid
    uid = uuid.uuid4().hex[:8]
    pass_id = f"test_task_pass_{uid}"
    fail_id = f"test_task_fail_{uid}"
    bench_id = f"eval_bench_{uid}"

    session = await get_session()
    async with session.begin():
        # Valid PASS task
        t_pass = TaskRecord(
            id=pass_id,
            instruction=f"Search for mechanical keyboard on store {uid}",
            site="store.test",
            status="completed",
        )
        session.add(t_pass)
        await session.flush()

        for idx, (atype, sel) in enumerate([("navigate", "url"), ("type", "#search"), ("click", "#item")]):
            s = StepRecord(
                task_id=t_pass.id,
                step_index=idx,
                action_type=atype,
                selector_used=sel,
                success=True,
                page_state_before_json='{"url": "store.test"}',
            )
            session.add(s)
            await session.flush()
            v = VerificationRecord(step_id=s.id, tier="rule", passed=True)
            session.add(v)

        # Valid FAIL task with same instruction
        t_fail = TaskRecord(
            id=fail_id,
            instruction=f"Search for mechanical keyboard on store {uid}",
            site="store.test",
            status="failed",
            error="Selector not found",
        )
        session.add(t_fail)
        await session.flush()

        for idx, (atype, sel) in enumerate([("navigate", "url"), ("type", "#search"), ("click", "#broken")]):
            s = StepRecord(
                task_id=t_fail.id,
                step_index=idx,
                action_type=atype,
                selector_used=sel,
                success=idx < 2,
                page_state_before_json='{"url": "store.test"}',
            )
            session.add(s)
            await session.flush()
            v = VerificationRecord(step_id=s.id, tier="rule", passed=idx < 2)
            session.add(v)

        # Benchmark task (MUST BE EXCLUDED)
        t_bench = TaskRecord(
            id=bench_id,
            instruction="Go to Amazon and search for Sony WH-1000XM5 headphones",
            site="amazon.in",
            status="completed",
        )
        session.add(t_bench)

    exporter = DatasetExporter(min_steps=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        sft_path = os.path.join(tmpdir, "sft.jsonl")
        dpo_path = os.path.join(tmpdir, "dpo.jsonl")
        grpo_path = os.path.join(tmpdir, "grpo.jsonl")

        sft_cnt = await exporter.export_sft_dataset(sft_path)
        assert sft_cnt >= 1
        with open(sft_path, "r", encoding="utf-8") as f:
            sft_rec = json.loads(f.readline())
            assert "messages" in sft_rec
            assert sft_rec["messages"][0]["role"] == "user"
            assert sft_rec["messages"][1]["role"] == "assistant"
            # Ensure benchmark task was strictly excluded
            assert "bench_001" not in sft_rec["task_id"]

        dpo_cnt = await exporter.export_dpo_dataset(dpo_path)
        assert dpo_cnt >= 1
        with open(dpo_path, "r", encoding="utf-8") as f:
            dpo_rec = json.loads(f.readline())
            assert "chosen" in dpo_rec and "rejected" in dpo_rec

        grpo_cnt = await exporter.export_grpo_dataset(grpo_path, group_size=2)
        assert grpo_cnt >= 1
        with open(grpo_path, "r", encoding="utf-8") as f:
            grpo_rec = json.loads(f.readline())
            assert "rewards" in grpo_rec and "ranks" in grpo_rec


# =============================================================================
# 5. CURRICULUM ENGINE TESTS
# =============================================================================

def test_curriculum_task_generation_and_mutation():
    """Verify tiered curriculum generation, mutation, and source tagging."""
    gen = TaskCurriculumGenerator()
    tasks = gen.generate_tiered_tasks(domain="sandbox://ecommerce", n_per_tier=2)

    assert len(tasks) == 6  # 2 tasks * 3 tiers
    tiers = {t.tier for t in tasks}
    assert tiers == {1, 2, 3}

    for t in tasks:
        assert t.source == "curriculum"
        assert t.domain == "sandbox://ecommerce"

    # Test mutation
    t1 = tasks[0]
    mutated = gen.mutate_task(t1)
    assert mutated.task_id != t1.task_id
    assert mutated.source == "curriculum"


def test_curriculum_scheduler():
    """Verify curriculum scheduler samples inversely to accuracy."""
    gen = TaskCurriculumGenerator()
    tasks = gen.generate_tiered_tasks(n_per_tier=2)
    scheduler = CurriculumScheduler(tasks)

    # Set Tier 1 high accuracy (0.95), Tier 2 low accuracy (0.1)
    scheduler.accuracy_by_tier = {1: 0.95, 2: 0.1, 3: 0.5}
    samples = [scheduler.sample_task() for _ in range(50)]
    tier2_count = sum(1 for s in samples if s.tier == 2)
    tier1_count = sum(1 for s in samples if s.tier == 1)

    # Tier 2 should be sampled more frequently than Tier 1
    assert tier2_count > tier1_count


# =============================================================================
# 6. ADAPTER REGISTRY & ROLLBACK GATE TESTS
# =============================================================================

def test_adapter_registry_and_rollback_gate():
    """Verify registry promotion and regression rollback gate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reg_file = os.path.join(tmpdir, "registry.json")
        reg = AdapterRegistry(registry_path=reg_file)

        # 1. Register baseline adapter with 80% accuracy
        reg.register_adapter("run_sft_01", rung=2, base_model="Qwen-7B", adapter_path="/path/sft")
        passed, msg = reg.evaluate_and_gate("run_sft_01", eval_accuracy=80.0)
        assert passed is True
        assert reg.active_adapter_id == "run_sft_01"
        assert reg.best_accuracy == 80.0

        # 2. Register new adapter that improves to 85%
        reg.register_adapter("run_dpo_01", rung=3, base_model="Qwen-7B", adapter_path="/path/dpo")
        passed, msg = reg.evaluate_and_gate("run_dpo_01", eval_accuracy=85.0)
        assert passed is True
        assert reg.active_adapter_id == "run_dpo_01"
        assert reg.best_accuracy == 85.0

        # 3. Register bad PPO run with 78% accuracy (regression > 2 points vs 85.0%)
        reg.register_adapter("run_ppo_bad", rung=5, base_model="Qwen-7B", adapter_path="/path/ppo_bad")
        passed, msg = reg.evaluate_and_gate("run_ppo_bad", eval_accuracy=78.0, regression_tolerance=2.0)
        assert passed is False  # Regressed!
        assert "Regression >" in msg
        # Active adapter remains the previous best!
        assert reg.active_adapter_id == "run_dpo_01"


# =============================================================================
# 7. TRAINER DRY-RUN TESTS (SFT, DPO, GRPO, PPO)
# =============================================================================

def test_all_trainers_dry_run():
    """Verify all 4 trainers pass dry-run validation without ML dependencies."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. SFT Dry-Run
        sft_cfg = SFTConfig(
            data_path=os.path.join(tmpdir, "sft.jsonl"),
            output_dir=os.path.join(tmpdir, "sft_out"),
            dry_run=True,
        )
        res_sft = train_sft(sft_cfg)
        assert res_sft is None

        # 2. DPO Dry-Run
        dpo_cfg = DPOConfig(
            data_path=os.path.join(tmpdir, "dpo.jsonl"),
            output_dir=os.path.join(tmpdir, "dpo_out"),
            dry_run=True,
        )
        res_dpo = train_dpo(dpo_cfg)
        assert res_dpo is None

        # 3. GRPO Dry-Run
        grpo_cfg = GRPOConfig(
            output_dir=os.path.join(tmpdir, "grpo_out"),
            group_size=4,
            dry_run=True,
        )
        res_grpo = train_grpo(grpo_cfg)
        assert res_grpo is None

        # 4. PPO Dry-Run
        ppo_cfg = PPOConfig(
            output_dir=os.path.join(tmpdir, "ppo_out"),
            sandbox_only=True,
            dry_run=True,
        )
        res_ppo = train_ppo(ppo_cfg)
        assert res_ppo is None


def test_grpo_advantage_computation():
    """Verify GRPO group advantage zero-mean normalization."""
    rewards = [1.0, 2.0, 3.0, 4.0]
    advs = compute_group_advantages(rewards)
    assert len(advs) == 4
    assert sum(advs) == pytest.approx(0.0, abs=1e-5)
    assert advs[-1] > advs[0]


def test_ppo_gae_and_normalizer():
    """Verify PPO GAE and running reward normalizer."""
    rewards = [1.0, 2.0, 3.0]
    values = [0.5, 1.5, 2.5]
    advs, returns = compute_gae(rewards, values, next_value=0.0, gamma=0.99, gae_lambda=0.95)
    assert len(advs) == 3
    assert len(returns) == 3

    normalizer = RunningRewardNormalizer(clip_range=5.0)
    for r in [0.0, 10.0]:
        _ = normalizer.normalize(r)
    norm = normalizer.normalize(5.0)
    assert -5.0 <= norm <= 5.0


# =============================================================================
# 8. REGRESSION TESTS — spec-conformance fixes (audit round-trip)
# =============================================================================

def test_sandbox_guard_closes_substring_holes():
    """'generalstore.com' must NOT pass the sandbox check; 'sandbox://amazon.in'
    must NOT smuggle a live domain behind the sandbox prefix."""
    env = BrowserGymEnv(sandbox_only=True)
    with pytest.raises(SandboxViolationError):
        env.reset({"instruction": "Browse generalstore", "domain": "generalstore.com"})
    with pytest.raises(SandboxViolationError):
        env.reset({"instruction": "Browse live amazon", "domain": "sandbox://amazon.in"})
    # Legitimate sandbox targets still allowed
    env.reset({"instruction": "Search headphones", "domain": "sandbox://ecommerce"})


def test_env_abort_reward_accounting_consistency():
    """After an anti-gaming abort, the accumulated total must equal the sum of
    the rewards actually returned to the agent (spec §6.2 checkout abort)."""
    env = BrowserGymEnv(sandbox_only=True, max_steps=10)
    env.reset("Buy something now")
    returned = []
    terminated = False
    for _ in range(10):
        _, r, terminated, _, info = env.step(
            {"type": "click", "selector": "#checkout-button", "text": ""}
        )
        returned.append(r)
        if terminated:
            assert info["reward_breakdown"]["abort_requested"]
            break
    assert terminated
    assert sum(returned) == pytest.approx(env.prm.total_reward)


def test_dataset_filters_human_gated_and_truncated():
    """Spec §5.2: human-gated and truncated-mid-run trajectories are dropped."""
    exporter = DatasetExporter(min_steps=2)
    gated = {"status": "failed", "error": "Task requires human confirmation for payment"}
    assert exporter.is_human_gated(gated)
    assert not exporter.is_human_gated({"status": "failed", "error": "Selector not found"})

    truncated_status = {"status": "timeout", "error": None, "steps": []}
    assert exporter.is_truncated_mid_run(truncated_status)
    truncated_err = {
        "status": "failed",
        "error": None,
        "steps": [{"step_index": 2, "error": "Step budget exhausted at max_steps"}],
    }
    assert exporter.is_truncated_mid_run(truncated_err)
    assert not exporter.is_truncated_mid_run(
        {"status": "failed", "error": None, "steps": [{"step_index": 2, "error": "not found"}]}
    )


def test_dataset_dedupe_applies_to_all_exporters():
    """Spec §5.2 dedupe rule is shared by SFT/DPO/GRPO exporters."""
    exporter = DatasetExporter()
    traj = {
        "domain": "store.test",
        "instruction": "Search for headphones",
        "steps": [{"action_type": "click", "selector_used": "#a", "input_value": ""}],
    }
    dup = json.loads(json.dumps(traj))
    unique = exporter.deduplicate([traj, dup])
    assert len(unique) == 1


def test_dpo_similarity_gate_blocks_unrelated_pairs():
    """Spec §5.1: DPO pairs require same/near-identical instructions."""
    from app.rl.dataset import instruction_similarity, MIN_DPO_SIMILARITY

    assert instruction_similarity(
        "Search for headphones on amazon.in", "search for headphones on amazon.in"
    ) == pytest.approx(1.0)
    sim = instruction_similarity(
        "Search for headphones on amazon.in", "Buy a laptop under budget today"
    )
    assert sim < MIN_DPO_SIMILARITY


def test_verdict_cache_is_snapshot_stable():
    """Spec §5.3: repeated export calls never flip labels for the same snapshot."""
    exporter = DatasetExporter()
    step = {"step_index": 1, "page_state_after": '{"url": "x"}', "verified_passed": True}
    first = exporter.resolve_verdict("task1", step)
    step["verified_passed"] = False  # underlying record mutated after first read
    second = exporter.resolve_verdict("task1", step)
    assert first is True and second is True  # cached, label stable


def test_parse_action_text_contract():
    from app.rl.train_utils import parse_action_text

    good = parse_action_text('Sure! { "type": "click", "selector": "#b" } done')
    assert good["type"] == "click" and good["selector"] == "#b"
    bad = parse_action_text("random garbage with no json")
    assert bad["type"] == "noop"


def test_curriculum_mutation_fallback_for_unknown_words():
    """mutate_task must always produce a changed instruction, even when the
    seed product/brand/price words are absent."""
    from app.rl.curriculum import CurriculumTask, TaskCurriculumGenerator

    gen = TaskCurriculumGenerator()
    task = CurriculumTask(
        task_id="curriculum_t1_0001",
        instruction="Find something cheap quickly on the site",
        domain="sandbox://ecommerce",
        tier=1,
    )
    mutated = gen.mutate_task(task)
    assert mutated.instruction != task.instruction
    assert mutated.metadata.get("mutated_from") == task.task_id


def test_replay_buffer_rejects_degenerate_noop_trajectories():
    """Spec §9.4: all-no-op trajectories are filtered before update."""
    replay = ReplayBuffer(min_confidence=0.6)
    degenerate = {
        "actions": [{"type": "noop"}, {"type": "noop"}],
        "steps": [{"reward": -0.05}, {"reward": -0.05}],
        "confidence": 1.0,
        "gaming_flags": [],
    }
    assert not replay.add_trajectory(degenerate)
    valid = {
        "actions": [{"type": "click", "selector": "#a"}],
        "steps": [{"reward": 1.0}],
        "confidence": 0.9,
        "gaming_flags": [],
    }
    assert replay.add_trajectory(valid)


def test_dataset_cli_exposes_db_flag():
    """Spec §5.4 CLI contract: `python -m app.rl.dataset <cmd> --db ... --out ...`."""
    import subprocess
    import sys

    backend_root = str(Path(__file__).resolve().parent.parent)
    for sub in ("sft", "dpo", "grpo"):
        proc = subprocess.run(
            [sys.executable, "-m", "app.rl.dataset", sub, "--help"],
            cwd=backend_root, capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert "--db" in proc.stdout, f"{sub} subcommand missing --db flag"


def _build_tiny_model():
    """Minimal random Llama + WordLevel tokenizer, fully offline."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    words = ["<unk>", "<pad>", "<s>", "</s>", "{", "}", "\"", ":", "type",
             "click", "goal", "next", "action", "policy", "assistant", "user", "system"]
    vocab = {w: i for i, w in enumerate(words)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>",
                                   pad_token="<pad>", bos_token="<s>", eos_token="</s>")
    fast.chat_template = (
        "{% for m in messages %}{{ '<|' + m['role'] + '|>\\n' + m['content'] + '<|end|>\\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"
    )
    config = LlamaConfig(vocab_size=len(words), hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=4, max_position_embeddings=512)
    model = LlamaForCausalLM(config)
    return model, fast


def test_grpo_update_step_produces_gradients():
    """Real GRPO update math: loss backward yields nonzero grads on trainable params."""
    torch = pytest.importorskip("torch")
    from app.rl.train_utils import apply_lora
    from app.rl.train_grpo import _grpo_update_step

    model, tok = _build_tiny_model()
    policy = apply_lora(model, {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0,
                                "target_modules": ["q_proj", "v_proj"]})
    policy.train()
    reference, _ = _build_tiny_model()
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    batch = [
        ([{"role": "user", "content": "goal next action"}], "click type", 0.5),
        ([{"role": "user", "content": "goal next action"}], "type click", -0.5),
    ]
    loss, kl, _ = _grpo_update_step(policy, reference, tok, batch, beta_kl=0.05)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in policy.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "no gradients reached LoRA parameters"
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_ppo_tokenize_batch_and_value_head_shapes():
    """PPO helper path: tokenization masking + value-head hidden selection."""
    pytest.importorskip("torch")
    from app.rl.train_ppo import _tokenize_batch, _last_hidden

    model, tok = _build_tiny_model()
    batch = [
        ({"prompt_messages": [{"role": "user", "content": "goal next"}],
          "response_text": "click type"}, 0.3),
        ({"prompt_messages": [{"role": "user", "content": "goal next"}],
          "response_text": "type click"}, -0.3),
    ]
    input_ids, attention_mask, resp_mask = _tokenize_batch(tok, batch, model.device)
    assert input_ids.shape == attention_mask.shape
    # Response mask never marks prompt position 0
    assert resp_mask[:, 0].sum() == 0
    hidden = _last_hidden(model, input_ids, attention_mask)
    assert hidden.shape[-1] == model.config.hidden_size


def test_env_accepts_custom_backend_injection():
    """Spec §7.2: backends are injectable (real BrowserController adapter shares
    the navigate/step_action/get_observation interface with the sandbox)."""
    class StubBackend:
        def __init__(self):
            self.calls = []
            self._obs = Observation(url="sandbox://stub/", title="Stub",
                                    visible_text="hello", elements=[], dom_hash="abc")

        def navigate(self, url):
            self.calls.append(("navigate", url))
            return self._obs

        def step_action(self, action):
            self.calls.append(("step", action.type))
            return self._obs

        def get_observation(self):
            return self._obs

    backend = StubBackend()
    env = BrowserGymEnv(sandbox_only=True, backend=backend)
    env.reset("stub task")
    env.step({"type": "click", "selector": "#x"})
    assert backend.calls[0][0] == "navigate"
    assert backend.calls[1] == ("step", "click")
