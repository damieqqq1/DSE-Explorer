"""Executor node — routes plan steps to the appropriate MCP tool."""

from __future__ import annotations

from agent.state import AgentState, EvidenceItem
from utils.logger import get_logger
from utils.fs_mcp_client import call_fs_tool
from utils.git_mcp_client import GIT_TOOLS, call_git_tool
from utils.github_mcp_client import GITHUB_TOOLS, call_github_tool
from utils.mcp_client import call_mcp_tool, call_mcp_tools_batch

logger = get_logger(__name__)

_SEARCH_TOOL = "search_dse_papers"


def _build_search_request(step: dict) -> dict:
    """Build a search_dse_papers request from a plan step."""
    return {
        "query": step.get("rewritten_query") or step["query"],
        "query_variants": step.get("query_variants") or [step["query"]],
        "top_k": (step.get("tool_args") or {}).get("top_k", 4),
        "candidate_k": (step.get("tool_args") or {}).get("candidate_k", 30),
        "max_chars_per_result": (step.get("tool_args") or {}).get("max_chars_per_result", 1000),
    }


def _merge_search_results(results: list[dict]) -> list[dict]:
    """Flatten results from multiple search calls into one list."""
    merged: list[dict] = []
    for r in results:
        merged.extend(r.get("results", []))
    return merged


def executor_node(state: AgentState) -> dict:
    evidence: list[EvidenceItem] = list(state.get("evidence", []))
    executed_step_ids = list(state.get("executed_step_ids", []))
    executed = set(executed_step_ids)
    pending_steps = [s for s in state.get("plan", []) if s["id"] not in executed]

    if not pending_steps:
        return {"evidence": evidence, "executed_step_ids": executed_step_ids}

    # Separate search steps from other tool steps
    search_steps = [s for s in pending_steps if s.get("tool", _SEARCH_TOOL) == _SEARCH_TOOL]
    other_steps = [s for s in pending_steps if s.get("tool", _SEARCH_TOOL) != _SEARCH_TOOL]

    # Batch search steps in one MCP session
    if search_steps:
        search_requests = [_build_search_request(s) for s in search_steps]
        search_results_list = call_mcp_tools_batch([(_SEARCH_TOOL, req) for req in search_requests])
        for step, tool_result in zip(search_steps, search_results_list, strict=True):
            logger.info("Search step %s: %s", step["id"], step["query"])
            evidence.append({
                "step_id": step["id"],
                "task": step["task"],
                "query": step.get("rewritten_query") or step["query"],
                "tool": _SEARCH_TOOL,
                "results": tool_result.get("results", []),
            })
            executed_step_ids.append(step["id"])

    # Tools that accept a free-text 'query' as their primary input.
    _QUERY_TOOLS = {"search_dse_papers", "search_research_web", "ask_paper", "design_experiment_template"}

    # Tools served by the filesystem MCP server instead of the DSE server.
    _FS_TOOLS = {
        "read_file", "write_file", "edit_file", "create_directory",
        "list_directory", "directory_tree", "move_file", "search_files",
        "get_file_info", "read_multiple_files", "list_allowed_directories",
    }

    # Route other steps to their specific tools
    for step in other_steps:
        tool = step.get("tool", _SEARCH_TOOL)
        tool_args = dict(step.get("tool_args", {}))

        # Strip 'query' from tool_args for tools that don't accept it
        # (the Planner may incorrectly include it for developer-mode tools).
        if tool not in _QUERY_TOOLS:
            tool_args.pop("query", None)

        # For query-accepting tools, fall back to step query if not provided.
        if tool in _QUERY_TOOLS and "query" not in tool_args:
            tool_args["query"] = step["query"]

        # Route to the correct MCP server.
        is_filesystem = tool in _FS_TOOLS
        is_git = tool in GIT_TOOLS
        is_github = tool in GITHUB_TOOLS

        backend = "dse"
        if is_filesystem:
            backend = "fs"
        elif is_git:
            backend = "git"
        elif is_github:
            backend = "github"

        logger.info("Calling %s [%s] for step %s: %s",
                     tool, backend, step["id"], step["task"])
        try:
            if is_filesystem:
                result = call_fs_tool(tool, tool_args)
            elif is_git:
                result = call_git_tool(tool, tool_args)
            elif is_github:
                result = call_github_tool(tool, tool_args)
            else:
                result = call_mcp_tool(tool, tool_args)
        except Exception:
            logger.exception("Tool %s failed for step %s", tool, step["id"])
            evidence.append({
                "step_id": step["id"],
                "task": step["task"],
                "query": step["query"],
                "tool": tool,
                "results": [{"error": f"Tool call to {tool} failed"}],
            })
            executed_step_ids.append(step["id"])
            continue

        # Normalise result: for tools that return structured data, wrap in results list
        if isinstance(result, dict):
            # Tools that return a list of chunks under "results"
            if "results" in result and isinstance(result["results"], list):
                evidence.append({
                    "step_id": step["id"],
                    "task": step["task"],
                    "query": step["query"],
                    "tool": tool,
                    "results": result["results"],
                })
            else:
                # Wrap the entire tool response as a single evidence item
                evidence.append({
                    "step_id": step["id"],
                    "task": step["task"],
                    "query": step["query"],
                    "tool": tool,
                    "results": [result],
                })
        else:
            evidence.append({
                "step_id": step["id"],
                "task": step["task"],
                "query": step["query"],
                "tool": tool,
                "results": [{"raw": str(result)}],
            })
        executed_step_ids.append(step["id"])

    return {"evidence": evidence, "executed_step_ids": executed_step_ids}
