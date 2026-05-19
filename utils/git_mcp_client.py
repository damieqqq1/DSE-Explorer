"""MCP client for the official mcp-server-git server (Python)."""

from __future__ import annotations

import os
import sys
from typing import Any

from utils.config import PROJECT_ROOT
from utils.mcp_client import _PersistentSessionManager

# Git tools we expose via this client
GIT_TOOLS = {
    "git_status",
    "git_diff_unstaged",
    "git_diff_staged",
    "git_diff",
    "git_commit",
    "git_add",
    "git_reset",
    "git_log",
    "git_create_branch",
    "git_checkout",
    "git_show",
    "git_branch",
}


def _git_params_factory() -> Any:
    from mcp.client.stdio import StdioServerParameters
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server_git", "--repository", str(PROJECT_ROOT)],
        cwd=str(PROJECT_ROOT),
    )


_git_session: _PersistentSessionManager | None = None


def _get_session() -> _PersistentSessionManager:
    global _git_session
    if _git_session is None:
        _git_session = _PersistentSessionManager(_git_params_factory)
    return _git_session


def call_git_tool(tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call any git MCP tool via the persistent session."""
    return _get_session().call(tool_name, arguments or {})


def list_git_tools() -> list[Any]:
    """Return tools exposed by the git MCP server."""
    return _get_session().list_tools()


# ---------------------------------------------------------------------------
# Convenience wrappers — pass repo_path for all tools that accept it
# ---------------------------------------------------------------------------

_REPO = str(PROJECT_ROOT)


def git_status(repo_path: str = _REPO) -> dict[str, Any]:
    return call_git_tool("git_status", {"repo_path": repo_path})


def git_diff_unstaged(repo_path: str = _REPO) -> dict[str, Any]:
    return call_git_tool("git_diff_unstaged", {"repo_path": repo_path})


def git_diff_staged(repo_path: str = _REPO) -> dict[str, Any]:
    return call_git_tool("git_diff_staged", {"repo_path": repo_path})


def git_diff(target: str, repo_path: str = _REPO) -> dict[str, Any]:
    return call_git_tool("git_diff", {"target": target, "repo_path": repo_path})


def git_log(max_count: int = 10, repo_path: str = _REPO) -> dict[str, Any]:
    return call_git_tool("git_log", {"max_count": max_count, "repo_path": repo_path})


def git_show(rev: str, repo_path: str = _REPO) -> dict[str, Any]:
    return call_git_tool("git_show", {"rev": rev, "repo_path": repo_path})


def git_branch(branch_type: str = "all", repo_path: str = _REPO) -> dict[str, Any]:
    return call_git_tool("git_branch", {"branch_type": branch_type, "repo_path": repo_path})


def git_commit(message: str, repo_path: str = _REPO) -> dict[str, Any]:
    return call_git_tool("git_commit", {"message": message, "repo_path": repo_path})


def git_add(files: list[str], repo_path: str = _REPO) -> dict[str, Any]:
    return call_git_tool("git_add", {"files": files, "repo_path": repo_path})


def git_checkout(branch: str, repo_path: str = _REPO) -> dict[str, Any]:
    return call_git_tool("git_checkout", {"branch": branch, "repo_path": repo_path})


def git_create_branch(branch_name: str, base_branch: str = "", repo_path: str = _REPO) -> dict[str, Any]:
    args: dict[str, Any] = {"branch_name": branch_name, "repo_path": repo_path}
    if base_branch:
        args["base_branch"] = base_branch
    return call_git_tool("git_create_branch", args)
