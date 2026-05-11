"""MCP client helpers for calling the local DSE Explorer MCP server."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from utils.config import PROJECT_ROOT


T = TypeVar("T")


def call_search_dse_papers_batch(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Call `search_dse_papers` through the local MCP server for each request."""

    return run_async(_call_search_dse_papers_batch(requests))


async def _call_search_dse_papers_batch(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server", "--transport", "stdio"],
        cwd=str(PROJECT_ROOT),
    )

    with open(os.devnull, "w", encoding="utf-8") as errlog:
        async with stdio_client(server_params, errlog=errlog) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                results: list[dict[str, Any]] = []
                for request in requests:
                    tool_result = await session.call_tool("search_dse_papers", request)
                    results.append(decode_tool_result(tool_result))
                return results


def decode_tool_result(tool_result: Any) -> dict[str, Any]:
    """Convert an MCP CallToolResult into the structured dictionary returned by the tool."""

    if getattr(tool_result, "isError", False):
        raise RuntimeError(extract_text_content(tool_result) or "MCP tool call failed.")

    structured = getattr(tool_result, "structuredContent", None)
    if structured is None:
        structured = getattr(tool_result, "structured_content", None)
    if isinstance(structured, dict):
        return structured

    text = extract_text_content(tool_result)
    if not text:
        return {}
    return json.loads(text)


def extract_text_content(tool_result: Any) -> str:
    texts: list[str] = []
    for item in getattr(tool_result, "content", []) or []:
        text = getattr(item, "text", "")
        if text:
            texts.append(text)
    return "\n".join(texts)


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
