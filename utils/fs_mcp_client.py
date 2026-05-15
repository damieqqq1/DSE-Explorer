"""MCP client for the official @modelcontextprotocol/server-filesystem server.

Uses a persistent stdio session — the Node.js subprocess is started once
and reused across calls.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from utils.config import PROJECT_ROOT

# Reuse the persistent session manager from the DSE client.
from utils.mcp_client import _PersistentSessionManager


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

def _fs_params_factory() -> Any:
    from mcp.client.stdio import StdioServerParameters
    from utils.config import get_settings
    settings = get_settings()
    dirs = [str(PROJECT_ROOT), str(settings.paper_root)]
    seen: set[str] = set()
    unique: list[str] = []
    for d in dirs:
        real = os.path.realpath(d)
        if real not in seen:
            seen.add(real)
            unique.append(d)
    return StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", *unique],
        cwd=str(PROJECT_ROOT),
    )


_fs_session: _PersistentSessionManager | None = None


def _get_fs_session() -> _PersistentSessionManager:
    global _fs_session
    if _fs_session is None:
        _fs_session = _PersistentSessionManager(_fs_params_factory)
    return _fs_session


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_fs_tool(tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call any filesystem MCP tool via the persistent session."""
    return _get_fs_session().call(tool_name, arguments or {})


def call_fs_tools_batch(requests: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Call multiple filesystem tools in one session."""
    return _get_fs_session().call_batch(requests)


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def read_file(path: str) -> dict[str, Any]:
    return call_fs_tool("read_file", {"path": path})


def write_file(path: str, content: str) -> dict[str, Any]:
    return call_fs_tool("write_file", {"path": path, "content": content})


def edit_file(path: str, edits: list[dict[str, str]], dry_run: bool = False) -> dict[str, Any]:
    return call_fs_tool("edit_file", {"path": path, "edits": edits, "dryRun": dry_run})


def list_directory(path: str) -> dict[str, Any]:
    return call_fs_tool("list_directory", {"path": path})


def directory_tree(path: str) -> dict[str, Any]:
    return call_fs_tool("directory_tree", {"path": path})


def search_files(path: str, pattern: str, exclude_patterns: list[str] | None = None) -> dict[str, Any]:
    return call_fs_tool("search_files", {
        "path": path,
        "pattern": pattern,
        "excludePatterns": exclude_patterns or [],
    })


def get_file_info(path: str) -> dict[str, Any]:
    return call_fs_tool("get_file_info", {"path": path})


def list_allowed_directories() -> dict[str, Any]:
    return call_fs_tool("list_allowed_directories")
