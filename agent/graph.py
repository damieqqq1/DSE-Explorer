"""LangGraph topology definition."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from agent.nodes.executor import executor_node
from agent.nodes.planner import planner_node
from agent.nodes.query_rewriter import query_rewriter_node
from agent.nodes.reasoner import reasoner_node, synthesizer_node
from agent.state import AgentState


def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("query_rewriter", query_rewriter_node)
    graph.add_node("executor", executor_node)
    graph.add_node("reasoner", reasoner_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "query_rewriter")
    graph.add_edge("query_rewriter", "executor")
    graph.add_edge("executor", "reasoner")
    graph.add_conditional_edges(
        "reasoner",
        should_continue_retrieval,
        {
            "continue": "query_rewriter",
            "finish": "synthesizer",
        },
    )
    graph.add_edge("synthesizer", END)

    return graph.compile()


def should_continue_retrieval(state: AgentState) -> str:
    return "continue" if state.get("needs_more_evidence") else "finish"


def run_agent(
    question: str,
    max_iterations: int = 2,
    conversation_context: str = "",
    long_term_context: str = "",
    planner_context: str = "",
    rewriter_context: str = "",
    reasoner_context: str = "",
    synthesis_context: str = "",
    context_trace: list[dict] | None = None,
) -> AgentState:
    app = build_agent_graph()
    return app.invoke(
        {
            "question": question,
            "max_iterations": max_iterations,
            "conversation_context": conversation_context,
            "long_term_context": long_term_context,
            "planner_context": planner_context,
            "rewriter_context": rewriter_context,
            "reasoner_context": reasoner_context,
            "synthesis_context": synthesis_context,
            "context_trace": context_trace or [],
        }
    )
