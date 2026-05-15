"""MCP client for the official @modelcontextprotocol/server-github server (Node.js)."""

from __future__ import annotations

import os
from typing import Any

from utils.config import PROJECT_ROOT
from utils.mcp_client import _PersistentSessionManager

# GitHub tools exposed by this client
GITHUB_TOOLS = {
    "create_or_update_file",
    "search_repositories",
    "create_repository",
    "get_file_contents",
    "push_files",
    "create_issue",
    "create_pull_request",
    "fork_repository",
    "create_branch",
    "list_commits",
    "list_issues",
    "update_issue",
    "add_issue_comment",
    "search_code",
    "search_issues",
    "search_users",
    "get_issue",
    "get_pull_request",
    "list_pull_requests",
    "create_pull_request_review",
    "merge_pull_request",
    "get_pull_request_files",
    "get_pull_request_status",
    "update_pull_request_branch",
    "get_pull_request_comments",
    "get_pull_request_reviews",
}


def _get_github_token() -> str:
    """Read GITHUB_PERSONAL_ACCESS_TOKEN from .env or environment."""
    from utils.config import get_settings
    settings = get_settings()
    token = getattr(settings, "github_token", "")
    if token:
        return token
    return os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "") or os.getenv("GITHUB_TOKEN", "")


def _github_params_factory() -> Any:
    from mcp.client.stdio import StdioServerParameters
    token = _get_github_token()
    env = os.environ.copy()
    if token:
        env["GITHUB_PERSONAL_ACCESS_TOKEN"] = token
    return StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        cwd=str(PROJECT_ROOT),
        env=env,
    )


_github_session: _PersistentSessionManager | None = None


def _get_session() -> _PersistentSessionManager:
    global _github_session
    if _github_session is None:
        _github_session = _PersistentSessionManager(_github_params_factory)
    return _github_session


def call_github_tool(tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call any GitHub MCP tool via the persistent session."""
    return _get_session().call(tool_name, arguments or {})


# ---------------------------------------------------------------------------
# Convenience wrappers — read-only operations
# ---------------------------------------------------------------------------


def search_repositories(query: str, page: int = 1, per_page: int = 10) -> dict[str, Any]:
    return call_github_tool("search_repositories", {"query": query, "page": page, "perPage": per_page})


def get_file_contents(owner: str, repo: str, path: str, branch: str = "") -> dict[str, Any]:
    args: dict[str, Any] = {"owner": owner, "repo": repo, "path": path}
    if branch:
        args["branch"] = branch
    return call_github_tool("get_file_contents", args)


def list_commits(owner: str, repo: str, sha: str = "", page: int = 1, per_page: int = 10) -> dict[str, Any]:
    args: dict[str, Any] = {"owner": owner, "repo": repo, "page": page, "perPage": per_page}
    if sha:
        args["sha"] = sha
    return call_github_tool("list_commits", args)


def list_issues(owner: str, repo: str, state: str = "open", page: int = 1, per_page: int = 10) -> dict[str, Any]:
    return call_github_tool("list_issues", {
        "owner": owner, "repo": repo, "state": state, "page": page, "perPage": per_page,
    })


def get_issue(owner: str, repo: str, issue_number: int) -> dict[str, Any]:
    return call_github_tool("get_issue", {"owner": owner, "repo": repo, "issue_number": issue_number})


def get_pull_request(owner: str, repo: str, pull_number: int) -> dict[str, Any]:
    return call_github_tool("get_pull_request", {"owner": owner, "repo": repo, "pull_number": pull_number})


def list_pull_requests(owner: str, repo: str, state: str = "open", page: int = 1, per_page: int = 10) -> dict[str, Any]:
    return call_github_tool("list_pull_requests", {
        "owner": owner, "repo": repo, "state": state, "page": page, "perPage": per_page,
    })


def search_code(query: str, page: int = 1, per_page: int = 10) -> dict[str, Any]:
    return call_github_tool("search_code", {"query": query, "page": page, "perPage": per_page})


def search_issues(query: str, page: int = 1, per_page: int = 10) -> dict[str, Any]:
    return call_github_tool("search_issues", {"query": query, "page": page, "perPage": per_page})


def get_pull_request_files(owner: str, repo: str, pull_number: int) -> dict[str, Any]:
    return call_github_tool("get_pull_request_files", {"owner": owner, "repo": repo, "pull_number": pull_number})


def get_pull_request_status(owner: str, repo: str, pull_number: int) -> dict[str, Any]:
    return call_github_tool("get_pull_request_status", {"owner": owner, "repo": repo, "pull_number": pull_number})
