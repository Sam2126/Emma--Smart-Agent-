"""
LangGraph state machine — the single execution engine for every task.

    START -> recall -> planner -> act -> verify --success--> learn -> END
                                  ^         |
                                  |      failed and a retry is safe
                                  |         v
                                  +----- replan

Extension tasks, Local Agent tasks, voice tasks, wake-word tasks, REST tasks
and the eval runner all run through this graph (via agent/runner.py). It
replaced the CrewAI crew, which ran tasks on a second, separate code path
alongside an older LangGraph graph that the server never used — two paths,
two sets of bugs.

Node names avoid state keys on purpose: LangGraph rejects a node named like a
state channel (e.g. "plan").
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agent.nodes.actor import act_node
from app.agent.nodes.learner import learn_node
from app.agent.nodes.planner import plan_node
from app.agent.nodes.recall import recall_node
from app.agent.nodes.replanner import replan_node
from app.agent.nodes.verifier import verify_node
from app.agent.state import AgentState


def route_after_verify(state: AgentState) -> str:
    """Learn on success, or when a retry is not allowed; otherwise replan."""
    if state.get("success"):
        return "learn"
    if state.get("retry_blocked_reason"):
        return "learn"
    return "replan"


def build_agent_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("recall", recall_node)
    graph.add_node("planner", plan_node)
    graph.add_node("act", act_node)
    graph.add_node("verify", verify_node)
    graph.add_node("replan", replan_node)
    graph.add_node("learn", learn_node)

    graph.add_edge(START, "recall")
    graph.add_edge("recall", "planner")
    graph.add_edge("planner", "act")
    graph.add_edge("act", "verify")
    graph.add_conditional_edges("verify", route_after_verify, {"learn": "learn", "replan": "replan"})
    graph.add_edge("replan", "act")
    graph.add_edge("learn", END)
    return graph


def compile_agent_graph():
    """Compile the graph. No checkpointer: runs are short and hold live objects."""
    return build_agent_graph().compile()
