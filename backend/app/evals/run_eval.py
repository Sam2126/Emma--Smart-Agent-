"""
Held-out Evaluation Runner.

Runs the frozen benchmark suite to evaluate genuine self-improvement accuracy,
measure regression, and test the grounded verification layer.

Usage:
    python -m app.evals.run_eval
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

import structlog

from app.utils.litellm_patch import apply_litellm_patch

apply_litellm_patch()

from app.agent.runner import AgentRunner
from app.browser.controller import BrowserController
from app.browser.page_state import extract_page_state
from app.browser.registry import set_browser_controller
from app.config import get_settings
from app.evals.benchmark_suite import BENCHMARK_TASKS
from app.state.brain import BrainMemory
from app.state.database import init_database
from app.utils.loop import set_main_loop

logger = structlog.get_logger(__name__)


async def run_benchmark():
    print("=" * 75)
    print("🚀 STARTING FROZEN BENCHMARK EVALUATION SUITE")
    print("=" * 75)
    print(f"Total benchmark tasks: {len(BENCHMARK_TASKS)}")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("-" * 75)

    # Browser tools run in worker threads and schedule Playwright work back
    # onto this loop, so it must be registered as the main loop.
    set_main_loop(asyncio.get_running_loop())
    await init_database()

    settings = get_settings()
    browser_ctrl = BrowserController()
    try:
        await browser_ctrl.connect(settings.cdp_endpoint)
    except Exception as e:
        print(f"⚠️ Could not connect to CDP at {settings.cdp_endpoint} ({e}). Launching a standalone browser...")
        await browser_ctrl.connect("http://127.0.0.1:9222")
    set_browser_controller(browser_ctrl)

    runner = AgentRunner()
    brain = BrainMemory()

    results: list[dict] = []
    passed_count = 0

    for i, task in enumerate(BENCHMARK_TASKS, start=1):
        print(f"\n[{i}/{len(BENCHMARK_TASKS)}] Running: {task.name} ({task.task_id})")
        print(f"    Instruction: '{task.instruction}'")

        start_t = time.time()
        try:
            res = await runner.run_task(task.instruction, task_id=f"eval_{task.task_id}", scope="browser")
            duration = time.time() - start_t

            active_page = await browser_ctrl.get_active_page()
            final_state = await extract_page_state(active_page)
            ground_truth_pass = task.ground_truth_assertion(final_state, res.get("summary", ""))

            passed = res.get("success", False) and ground_truth_pass
            if passed:
                passed_count += 1
            status_str = "✅ PASS" if passed else "❌ FAIL"
            print(f"    Result: {status_str} (Execution: {duration:.1f}s | GroundTruth: {ground_truth_pass} | retried: {res.get('retried')})")

            results.append({
                "task_id": task.task_id,
                "name": task.name,
                "passed": passed,
                "duration": duration,
                "ground_truth_pass": ground_truth_pass,
            })
        except Exception as e:
            duration = time.time() - start_t
            print(f"    Result: ❌ ERROR ({str(e)})")
            results.append({
                "task_id": task.task_id,
                "name": task.name,
                "passed": False,
                "duration": duration,
                "error": str(e),
            })

    total = len(BENCHMARK_TASKS)
    accuracy = (passed_count / total) * 100.0 if total > 0 else 0.0

    print("\n" + "=" * 75)
    print("📊 BENCHMARK EVALUATION REPORT")
    print("=" * 75)
    print(f"Total Passed: {passed_count}/{total} ({accuracy:.1f}%)")
    print("-" * 75)
    for r in results:
        mark = "✓" if r["passed"] else "✗"
        print(f"  [{mark}] {r['name']:<45} {r.get('duration', 0):>5.1f}s")
    print("=" * 75)

    stats = await brain.get_brain_stats()
    print(f"Current Brain Skills Learned: {stats.get('total_skills_learned', 0)}")
    print(f"Current Failure Patterns: {stats.get('total_failure_patterns', 0)}")
    print("=" * 75)

    from app.agent.context import drain_background_tasks

    await drain_background_tasks(timeout=20.0)
    await browser_ctrl.disconnect()


if __name__ == "__main__":
    asyncio.run(run_benchmark())
