"""Executor node for solving planned subtasks through RAG tools."""

from __future__ import annotations

from agent.state import AgentState, EvidenceItem
from utils.logger import get_logger
from utils.mcp_client import call_search_dse_papers_batch


logger = get_logger(__name__)


def executor_node(state: AgentState) -> dict[str, list[EvidenceItem] | list[int]]:
    evidence: list[EvidenceItem] = list(state.get("evidence", []))
    executed_step_ids = list(state.get("executed_step_ids", []))
    executed = set(executed_step_ids)
    pending_steps = [step for step in state.get("plan", []) if step["id"] not in executed]

    requests = [
        {
            "query": step.get("rewritten_query") or step["query"],
            "query_variants": step.get("query_variants", [step["query"]]),
            "top_k": 4,
            "candidate_k": 30,
            "max_chars_per_result": 1000,
        }
        for step in pending_steps
    ]
    tool_results = call_search_dse_papers_batch(requests) if requests else []

    for step, tool_result in zip(pending_steps, tool_results, strict=True):
        logger.info("Executing step %s: %s", step["id"], step["query"])
        evidence.append(
            {
                "step_id": step["id"],
                "task": step["task"],
                "query": step.get("rewritten_query") or step["query"],
                "results": tool_result["results"],
            }
        )
        executed_step_ids.append(step["id"])

    return {"evidence": evidence, "executed_step_ids": executed_step_ids}
