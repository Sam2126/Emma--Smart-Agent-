"""
Tests for the LangGraph engine's shape, state schema, routing and protocol parsing.
"""

from app.agent.graph import compile_agent_graph, route_after_verify
from app.agent.state import FailureKind, TaskStatus, create_initial_state
from app.websocket.protocol import (
    ConfirmationResponseMessage,
    TaskCancelMessage,
    TaskSubmitMessage,
    parse_incoming_message,
)


def test_create_initial_state():
    state = create_initial_state("Search for iPhone on Amazon", task_id="test-123")
    assert state["task_id"] == "test-123"
    assert state["task_instruction"] == "Search for iPhone on Amazon"
    assert state["status"] == TaskStatus.PLANNING.value
    assert state["attempt"] == 1
    assert state["plan"] == ""
    assert state["trajectory"] == []
    assert state["experiences"] == []
    assert state["failure_kind"] == FailureKind.NONE.value
    assert state["retried"] is False


def test_initial_state_generates_task_id():
    assert create_initial_state("x")["task_id"]


def test_graph_has_the_unified_pipeline():
    compiled = compile_agent_graph()
    nodes = set(compiled.get_graph().nodes)
    assert {"recall", "planner", "act", "verify", "replan", "learn"} <= nodes
    # The old CrewAI-era nodes are gone.
    assert "perceive" not in nodes and "human_gate" not in nodes


def test_route_after_verify():
    assert route_after_verify({"success": True}) == "learn"
    assert route_after_verify({"success": False, "retry_blocked_reason": "already retried"}) == "learn"
    assert route_after_verify({"success": False, "retry_blocked_reason": ""}) == "replan"


def test_websocket_protocol_parsing():
    msg_submit = parse_incoming_message({"type": "task_submit", "instruction": "Find best mouse"})
    assert isinstance(msg_submit, TaskSubmitMessage)
    assert msg_submit.instruction == "Find best mouse"

    msg_conf = parse_incoming_message({"type": "confirmation_response", "task_id": "123", "confirmed": True})
    assert isinstance(msg_conf, ConfirmationResponseMessage)
    assert msg_conf.confirmed is True

    msg_cancel = parse_incoming_message({"type": "task_cancel", "task_id": "123"})
    assert isinstance(msg_cancel, TaskCancelMessage)
    assert msg_cancel.task_id == "123"

    assert parse_incoming_message({"type": "unknown_message"}) is None
