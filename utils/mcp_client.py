"""MCP client helpers for calling the local DSE Explorer MCP server.

Uses a persistent stdio session — the subprocess is started once and reused
across all calls during the process lifetime, avoiding repeated startup cost.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from collections.abc import Coroutine
from typing import Any, Literal, TypeVar

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from utils.config import PROJECT_ROOT

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Persistent session manager
# ---------------------------------------------------------------------------


class _PersistentSessionManager:
    """Keeps a subprocess + MCP session alive in a background event loop."""

    def __init__(self, params_factory: Any) -> None:
        self._params_factory = params_factory
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session: ClientSession | None = None
        self._started = threading.Event()
        self._init_error: Exception | None = None

    def _ensure_running(self) -> None:
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            self._started.clear()
            self._init_error = None

            def run_loop() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                try:
                    loop.run_until_complete(self._open())
                except Exception as exc:
                    self._init_error = exc
                    self._started.set()
                    loop.close()
                    self._loop = None
                    return
                self._started.set()
                loop.run_forever()
                # run_forever exited — clean up.
                try:
                    loop.run_until_complete(self._close())
                except Exception:
                    pass
                loop.close()
                self._loop = None

            thread = threading.Thread(target=run_loop, daemon=True)
            thread.start()
            if not self._started.wait(timeout=30):
                raise RuntimeError("Timed out waiting for MCP session to start.")
            if self._init_error is not None:
                raise self._init_error

    async def _open(self) -> None:
        params = self._params_factory()
        # Enter both context managers and keep them alive.
        errlog = open(os.devnull, "w", encoding="utf-8")
        self._stdio_ctx = stdio_client(params, errlog=errlog)
        read_stream, write_stream = await self._stdio_ctx.__aenter__()
        self._session_ctx = ClientSession(read_stream, write_stream)
        self._session = await self._session_ctx.__aenter__()
        await self._session.initialize()

    async def _close(self) -> None:
        if self._session_ctx is not None:
            try:
                await self._session_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._session_ctx = None
        if self._stdio_ctx is not None:
            try:
                await self._stdio_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._stdio_ctx = None
        self._session = None

    def call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self._ensure_running()
        assert self._loop is not None
        assert self._session is not None
        coro = self._session.call_tool(tool_name, arguments)
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            result = future.result(timeout=120)
        except Exception:
            # Re-raise so the executor can fall back gracefully.
            raise
        return _decode_tool_result(result)

    def call_batch(self, requests: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        self._ensure_running()
        assert self._loop is not None
        assert self._session is not None
        results: list[dict[str, Any]] = []
        for tool_name, args in requests:
            coro = self._session.call_tool(tool_name, args)
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            results.append(_decode_tool_result(future.result(timeout=120)))
        return results


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

def _dse_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server", "--transport", "stdio"],
        cwd=str(PROJECT_ROOT),
    )


_dse_session: _PersistentSessionManager | None = None


def _get_dse_session() -> _PersistentSessionManager:
    global _dse_session
    if _dse_session is None:
        _dse_session = _PersistentSessionManager(_dse_params)
    return _dse_session


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def call_mcp_tool(tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call any MCP tool by name via the persistent DSE session."""
    return _get_dse_session().call(tool_name, arguments or {})


def call_mcp_tools_batch(requests: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Call multiple MCP tools in one session."""
    return _get_dse_session().call_batch(requests)


# Keep the old run_async for backward compat (used by fs client and any other callers).
def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine from sync code, including when an event loop is already active."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: T | None = None
    error: BaseException | None = None

    def runner() -> None:
        nonlocal result, error
        try:
            result = asyncio.run(coro)
        except BaseException as exc:
            error = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if error is not None:
        raise error
    return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Decoding helpers (exported for reuse)
# ---------------------------------------------------------------------------


def _decode_tool_result(tool_result: Any) -> dict[str, Any]:
    """Convert an MCP CallToolResult into a dict."""
    if getattr(tool_result, "isError", False):
        raise RuntimeError(_extract_text_content(tool_result) or "MCP tool call failed.")

    structured = getattr(tool_result, "structuredContent", None)
    if structured is None:
        structured = getattr(tool_result, "structured_content", None)
    if isinstance(structured, dict):
        return structured

    text = _extract_text_content(tool_result)
    if not text:
        return {}
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {"content": text}


def _extract_text_content(tool_result: Any) -> str:
    texts: list[str] = []
    for item in getattr(tool_result, "content", []) or []:
        text = getattr(item, "text", "")
        if text:
            texts.append(text)
    return "\n".join(texts)


# ---------------------------------------------------------------------------
# Tool-specific convenience wrappers
# ---------------------------------------------------------------------------


def call_search_dse_papers(
    query: str,
    query_variants: list[str] | None = None,
    top_k: int = 5,
    candidate_k: int = 30,
    max_chars_per_result: int = 1200,
    use_hybrid: bool = True,
    use_rerank: bool = True,
) -> dict[str, Any]:
    """Search the local DSE/MOO paper knowledge base."""
    return call_mcp_tool("search_dse_papers", {
        "query": query,
        "query_variants": query_variants or [],
        "top_k": top_k,
        "candidate_k": candidate_k,
        "max_chars_per_result": max_chars_per_result,
        "use_hybrid": use_hybrid,
        "use_rerank": use_rerank,
    })


def call_search_dse_papers_batch(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Call ``search_dse_papers`` for each request dict in one MCP session.

    Kept for backward compatibility with ``agent/nodes/executor.py``.
    """
    return call_mcp_tools_batch([("search_dse_papers", req) for req in requests])


def call_get_rag_corpus_stats() -> dict[str, Any]:
    """Return basic Qdrant collection stats for the local paper RAG corpus."""
    return call_mcp_tool("get_rag_corpus_stats")


def call_generate_dse_method_matrix(
    keyword: str = "",
    category: str = "",
    method_names: list[str] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    limit: int = 20,
    output_format: Literal["markdown", "json", "csv"] = "markdown",
    save_path: str = "",
) -> dict[str, Any]:
    """Generate a DSE method comparison matrix from extracted paper info."""
    return call_mcp_tool("generate_dse_method_matrix", {
        "keyword": keyword,
        "category": category,
        "method_names": method_names or [],
        "year_from": year_from,
        "year_to": year_to,
        "limit": limit,
        "output_format": output_format,
        "save_path": save_path,
    })


def call_search_research_web(
    query: str = "",
    topic: Literal["design_space_exploration", "combinatorial_optimization", "both"] = "both",
    max_results: int = 5,
    provider: Literal["auto", "tavily", "serpapi"] = "auto",
) -> dict[str, Any]:
    """Search the web for DSE/CO research."""
    return call_mcp_tool("search_research_web", {
        "query": query,
        "topic": topic,
        "max_results": max_results,
        "provider": provider,
    })


def call_execute_python_code(
    code: str,
    timeout: int = 30,
) -> dict[str, Any]:
    """Execute Python code in a sandboxed subprocess."""
    return call_mcp_tool("execute_python_code", {
        "code": code,
        "timeout": timeout,
    })


def call_summarize_paper(
    paper_path: str,
    max_chars: int = 30000,
) -> dict[str, Any]:
    """Generate a structured summary of a specific paper."""
    return call_mcp_tool("summarize_paper", {
        "paper_path": paper_path,
        "max_chars": max_chars,
    })


def call_find_related_papers(
    paper_path: str,
    top_k: int = 10,
) -> dict[str, Any]:
    """Find papers related to a given paper."""
    return call_mcp_tool("find_related_papers", {
        "paper_path": paper_path,
        "top_k": top_k,
    })


def call_ask_paper(
    paper_path: str,
    question: str,
    top_k: int = 8,
    max_chars_per_result: int = 1500,
) -> dict[str, Any]:
    """Ask a question about a specific paper and get a page-cited answer."""
    return call_mcp_tool("ask_paper", {
        "paper_path": paper_path,
        "question": question,
        "top_k": top_k,
        "max_chars_per_result": max_chars_per_result,
    })


def call_compute_dse_metrics(
    points: list[list[float]],
    reference_point: list[float] | None = None,
    true_pareto_front: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Compute DSE/MOO evaluation metrics (HV, IGD, GD, Spread)."""
    return call_mcp_tool("compute_dse_metrics", {
        "points": points,
        "reference_point": reference_point,
        "true_pareto_front": true_pareto_front,
    })


def call_generate_pareto_front(
    points: list[list[float]],
    labels: list[str] | None = None,
    title: str = "Pareto Front",
    x_label: str = "Objective 1 (minimise)",
    y_label: str = "Objective 2 (minimise)",
    true_pareto_front: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Generate a 2-D Pareto-front scatter plot."""
    return call_mcp_tool("generate_pareto_front", {
        "points": points,
        "labels": labels,
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "true_pareto_front": true_pareto_front,
    })


def call_design_experiment_template(
    problem_description: str,
    num_design_variables: int | None = None,
    num_objectives: int | None = None,
    variable_type: Literal["continuous", "discrete", "mixed"] = "continuous",
    constraints: list[str] | None = None,
    evaluation_budget: int | None = None,
    evaluation_cost: str = "",
) -> dict[str, Any]:
    """Generate a DSE/MOO experiment design template."""
    return call_mcp_tool("design_experiment_template", {
        "problem_description": problem_description,
        "num_design_variables": num_design_variables,
        "num_objectives": num_objectives,
        "variable_type": variable_type,
        "constraints": constraints or [],
        "evaluation_budget": evaluation_budget,
        "evaluation_cost": evaluation_cost,
    })


def call_fetch_web_page(
    url: str,
    output_format: str = "markdown",
    max_chars: int = 20000,
) -> dict[str, Any]:
    """Fetch a web page and return its content as Markdown or plain text."""
    return call_mcp_tool("fetch_web_page", {
        "url": url,
        "output_format": output_format,
        "max_chars": max_chars,
    })


def call_download_paper_pdf(
    url: str,
    filename: str = "",
    category: str = "_downloaded",
) -> dict[str, Any]:
    """Download a PDF paper from a URL to the local paper corpus."""
    return call_mcp_tool("download_paper_pdf", {
        "url": url,
        "filename": filename,
        "category": category,
    })


def call_search_memory(
    query: str,
    session_id: str = "default",
    kind: str = "",
    limit: int = 10,
) -> dict[str, Any]:
    """Search conversation history and long-term memories."""
    return call_mcp_tool("search_memory", {
        "query": query,
        "session_id": session_id,
        "kind": kind,
        "limit": limit,
    })


def call_save_memory(
    content: str,
    kind: str = "conclusion",
    session_id: str = "default",
    source: str = "agent",
    importance: float = 0.7,
) -> dict[str, Any]:
    """Save information to long-term memory."""
    return call_mcp_tool("save_memory", {
        "content": content,
        "kind": kind,
        "session_id": session_id,
        "source": source,
        "importance": importance,
    })


def call_get_recent_conversations(
    session_id: str = "default",
    limit: int = 5,
) -> dict[str, Any]:
    """Return recent conversation turns."""
    return call_mcp_tool("get_recent_conversations", {
        "session_id": session_id,
        "limit": limit,
    })


def call_get_memory_stats(
    session_id: str = "default",
) -> dict[str, Any]:
    """Return memory statistics."""
    return call_mcp_tool("get_memory_stats", {
        "session_id": session_id,
    })
