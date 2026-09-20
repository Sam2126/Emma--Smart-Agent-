"""
Tests for structured trajectory capture and local-task skill learning.

Every tool call the actor makes goes through toolkit.execute_tool_call, which
returns a structured trajectory step. Local tools report failures as plain
sentences ("No files matching ..."), so those must still be recorded as failed
steps, and a successful local trajectory must produce a SkillRecord.
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import BaseModel
from sqlalchemy import delete, select

from app.agent.toolkit import execute_tool_call, output_indicates_failure, summarize_tool_args
from app.state.brain import BrainMemory
from app.state.database import get_session
from app.state.models import SkillRecord
from app.tools.base import BaseTool


class _EchoInput(BaseModel):
    path: str
    content: str = ""


class _EchoTool(BaseTool):
    name: str = "write_file"
    description: str = "test tool"
    args_schema: type[BaseModel] = _EchoInput

    def _run(self, path: str, content: str = "") -> str:
        return f"Wrote {path} ({len(content)} chars)."


class _NotFoundTool(BaseTool):
    name: str = "find_files"
    description: str = "test tool"
    args_schema: type[BaseModel] = _EchoInput

    def _run(self, path: str, content: str = "") -> str:
        return f"No files matching '{path}' under C:\\Users\\test"


class _RaisingTool(BaseTool):
    name: str = "run_command"
    description: str = "test tool"
    args_schema: type[BaseModel] = _EchoInput

    def _run(self, path: str, content: str = "") -> str:
        raise RuntimeError("boom")


def test_summarize_tool_args_prefers_meaningful_keys():
    assert summarize_tool_args({"path": "/home/user/Desktop/x.txt"}) == "/home/user/Desktop/x.txt"
    assert summarize_tool_args({"name": "notepad"}) == "notepad"
    assert summarize_tool_args({"command": "dir"}) == "dir"
    assert summarize_tool_args("plain string") == "plain string"


async def test_successful_tool_call_becomes_a_success_step():
    tool = _EchoTool()
    result = await execute_tool_call({tool.name: tool}, "write_file", '{"path": "/tmp/a.txt", "content": "hello"}')
    step = result.record
    assert step["action_type"] == "write_file"
    assert step["selector_used"] == "/tmp/a.txt"
    assert step["success"] is True
    assert "Wrote /tmp/a.txt" in result.output


async def test_soft_failure_text_is_detected_without_an_exception():
    tool = _NotFoundTool()
    result = await execute_tool_call({tool.name: tool}, "find_files", {"path": "WhatsApp"})
    assert result.record["success"] is False


async def test_raising_tool_records_failure():
    tool = _RaisingTool()
    result = await execute_tool_call({tool.name: tool}, "run_command", {"path": "x"})
    assert result.record["success"] is False
    assert "boom" in result.output


async def test_unknown_tool_and_bad_json_are_failures_not_crashes():
    tool = _EchoTool()
    unknown = await execute_tool_call({tool.name: tool}, "nope", "{}")
    assert unknown.record["success"] is False and "Unknown tool" in unknown.output
    bad = await execute_tool_call({tool.name: tool}, "write_file", "{not json")
    assert bad.record["success"] is False and "Invalid arguments" in bad.output


async def test_missing_required_argument_is_reported():
    tool = _EchoTool()
    result = await execute_tool_call({tool.name: tool}, "write_file", {})
    assert result.record["success"] is False


def test_failure_scan_only_reads_the_start_of_long_results():
    long_success = "Page text: " + "x" * 400 + " an error occurred in a comment"
    assert output_indicates_failure(long_success) is False
    assert output_indicates_failure("Error executing tool: boom") is True


@pytest.mark.asyncio
async def test_local_task_success_creates_a_skill_record():
    """A successful local trajectory must produce a real SkillRecord."""
    domain = f"local-test-{uuid.uuid4().hex[:8]}"
    brain = BrainMemory()
    trajectory = [{"action_type": "open_app", "selector_used": "whatsapp", "input_value": "whatsapp", "success": True}]

    await brain.learn_from_task(
        task_id=f"task-{uuid.uuid4().hex[:8]}",
        instruction="Open WhatsApp on this local device.",
        domain=domain,
        trajectory=trajectory,
        success=True,
        error=None,
        duration_seconds=2.5,
        raw_result="Launched: whatsapp",
    )

    session = await get_session()
    async with session.begin():
        result = await session.execute(select(SkillRecord).where(SkillRecord.domain_pattern == domain))
        skill = result.scalar_one_or_none()

    assert skill is not None
    assert skill.skill_type == "open_app"
    assert skill.usage_count == 1

    session2 = await get_session()
    async with session2.begin():
        await session2.execute(delete(SkillRecord).where(SkillRecord.domain_pattern == domain))
