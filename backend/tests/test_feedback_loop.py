"""
The feedback loop the user asked for (2026-09-16):

  * 👎 asks what went wrong; the note is stored with the task
  * that note is put in front of the agent next time it plans something similar
  * 👍 marks the run as confirmed: its steps are kept and offered as a flow to
    LEARN FROM (adapt it), and it ranks above everything else in recall
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.agent.explain import build_explanation
from app.config import get_settings
from app.state.brain import BrainMemory
from app.state.reflection import (
    ReflectionEngine,
    confirmed_flow_block,
    fallback_strategy,
    user_complaints_block,
)
from app.state.semantic_memory import Experience, step_details
from tests.conftest import ScriptedCompletion, llm_message

PRIME_RUN = [
    {"action_type": "create_folder", "input_value": r"C:\Users\samar\Desktop\MAYANK", "success": True},
    {"action_type": "write_file", "input_value": r"C:\Users\samar\Desktop\MAYANK\prime_code.txt", "success": True},
    {"action_type": "read_file", "input_value": r"C:\Users\samar\Desktop\MAYANK\prime_code.txt", "success": True},
]


@pytest.fixture
def memory_settings(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "semantic_memory_enabled", True)
    monkeypatch.setattr(s, "semantic_min_similarity", 0.2)
    monkeypatch.setattr(s, "llm_reflection_enabled", True)
    return s


def _experience(**overrides) -> Experience:
    base = dict(
        task_id="t1", instruction="make a folder on the desktop and write prime code into a file",
        domain="local", scope="local", success=True, tools=["create_folder", "write_file"],
        lesson="", error="", feedback=0, user_disputed=False,
        created_at=datetime.now(timezone.utc).isoformat(), similarity=0.8,
        steps_detail=step_details(PRIME_RUN),
    )
    base.update(overrides)
    return Experience(**base)


# =============================================================================
# What a run recorded
# =============================================================================

def test_steps_are_recorded_with_their_targets():
    details = step_details(PRIME_RUN + [{"action_type": "run_command", "input_value": "New-Item ...", "success": False}])
    assert details[0].startswith("create_folder(C:\\Users\\samar\\Desktop\\MAYANK)")
    assert details[-1].endswith("[FAILED]")


def test_steps_and_comment_survive_a_round_trip(semantic_memory, memory_settings):
    semantic_memory.upsert_sync(task_id="p1", instruction="make a folder and write prime code", domain="local",
                                scope="local", success=True, trajectory=PRIME_RUN, lesson="use the file tools")
    semantic_memory.record_feedback_sync("p1", -1, "you used powershell instead of the file tools")
    recalled = semantic_memory.recall_sync("make a folder and write prime code")[0]
    assert recalled.steps_detail[0].startswith("create_folder(")
    assert recalled.feedback_comment == "you used powershell instead of the file tools"
    assert recalled.verified is False


def test_a_confirmed_run_ranks_above_the_others(semantic_memory, memory_settings):
    for task_id in ("plain", "confirmed"):
        semantic_memory.upsert_sync(task_id=task_id, instruction="make a folder and write prime code into a file",
                                    domain="local", scope="local", success=True, trajectory=PRIME_RUN, lesson="")
    semantic_memory.record_feedback_sync("confirmed", +1)
    recalled = semantic_memory.recall_sync("make a folder and write prime code into a file")
    assert [e.task_id for e in recalled] == ["confirmed", "plain"]
    assert recalled[0].verified is True and recalled[1].verified is False


# =============================================================================
# What the agent is told next time
# =============================================================================

def test_confirmed_flow_is_offered_as_something_to_learn_from():
    block = confirmed_flow_block([_experience(feedback=1)])
    assert "CONFIRMED AS CORRECT" in block and "adapt" in block
    assert "create_folder(C:\\Users\\samar\\Desktop\\MAYANK)" in block
    assert confirmed_flow_block([_experience()]) == ""


def test_user_complaints_are_passed_on():
    disputed = _experience(feedback=-1, user_disputed=True, feedback_comment="you used powershell instead of the file tools")
    block = user_complaints_block([disputed])
    assert "WENT WRONG" in block and "powershell instead of the file tools" in block
    assert user_complaints_block([_experience(feedback=1)]) == ""


async def test_strategy_prompt_carries_both(memory_settings):
    fake = ScriptedCompletion([llm_message("STRATEGY:\n- use create_folder then write_file")])
    experiences = [
        _experience(task_id="ok", feedback=1),
        _experience(task_id="bad", success=False, feedback=-1, feedback_comment="don't write powershell by hand"),
    ]
    await ReflectionEngine(completion_fn=fake).plan_strategy("make a folder on the desktop with a python file", experiences)
    prompt = fake.calls[0]["messages"][1]["content"]
    assert "CONFIRMED AS CORRECT" in prompt
    assert "don't write powershell by hand" in prompt
    assert "what it did: create_folder(" in prompt


def test_fallback_strategy_carries_both_without_an_llm():
    text = fallback_strategy([
        _experience(task_id="ok", feedback=1),
        _experience(task_id="bad", success=False, feedback=-1, feedback_comment="don't write powershell by hand"),
    ])
    assert "CONFIRMED AS CORRECT" in text and "don't write powershell by hand" in text


def test_explanation_mentions_the_confirmed_flow_and_the_complaint():
    text = build_explanation(
        trajectory=PRIME_RUN, success=True,
        experiences=[
            _experience(task_id="ok", feedback=1),
            _experience(task_id="bad", success=False, feedback=-1, feedback_comment="you used powershell by hand"),
        ],
        strategy="", strategy_from_llm=False, retried=False, first_attempt_error="",
        retry_blocked_reason="", error="",
    )
    assert "run you approved" in text
    assert "you used powershell by hand" in text


# =============================================================================
# The reply the user sees
# =============================================================================

async def test_thumbs_down_reply_repeats_the_note(semantic_memory, memory_settings):
    semantic_memory.upsert_sync(task_id="n1", instruction="make a folder", domain="local", scope="local",
                                success=True, trajectory=PRIME_RUN, lesson="")
    brain = BrainMemory(semantic=semantic_memory, reflection=ReflectionEngine(ScriptedCompletion([])))
    result = await brain.record_feedback("n1", -1, "you used powershell instead of the file tools")
    assert result["applied"] is True
    assert "powershell instead of the file tools" in result["message"]


async def test_thumbs_down_without_a_note_asks_for_one(semantic_memory, memory_settings):
    semantic_memory.upsert_sync(task_id="n2", instruction="make a folder", domain="local", scope="local",
                                success=True, trajectory=PRIME_RUN, lesson="")
    brain = BrainMemory(semantic=semantic_memory, reflection=ReflectionEngine(ScriptedCompletion([])))
    result = await brain.record_feedback("n2", -1)
    assert "Tell me what went wrong" in result["message"]
