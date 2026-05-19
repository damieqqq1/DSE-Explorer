"""Dynamic MCP tool discovery, ranking, and routing metadata."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from utils.config import PROJECT_ROOT
from utils.logger import get_logger
from utils.memory import tokenize


logger = get_logger(__name__)

SERVER_DSE = "dse"
SERVER_FS = "fs"
SERVER_GIT = "git"
SERVER_GITHUB = "github"

DEFAULT_TOOL_CONTEXT_LIMIT = 36
TOOL_REGISTRY_CACHE_PATH = PROJECT_ROOT / "memory" / "tool_registry_cache.json"
TOOL_REGISTRY_CACHE_TTL_SECONDS = int(os.getenv("TOOL_REGISTRY_CACHE_TTL_SECONDS", "3600"))
TOOL_REGISTRY_CACHE_VERSION = 2


@dataclass(frozen=True)
class ToolSpec:
    name: str
    server: str
    description: str
    input_schema: dict[str, Any]
    category: str
    risk_level: str

    @property
    def searchable_text(self) -> str:
        schema_text = json.dumps(self.input_schema, ensure_ascii=False, sort_keys=True)
        return f"{self.name} {self.server} {self.category} {self.description} {schema_text}"


def get_relevant_tool_context(question: str, limit: int = DEFAULT_TOOL_CONTEXT_LIMIT) -> str:
    """Render a planner-ready list of tools ranked for the current question."""
    specs = select_relevant_tools(question, limit=limit)
    if not specs:
        return ""

    lines = [
        "Use these MCP tools for this question. Prefer this dynamic list over the static fallback list.",
        "Each tool line includes server, category, risk, purpose, and accepted arguments.",
    ]
    for index, spec in enumerate(specs, start=1):
        args = summarize_schema(spec.input_schema)
        description = " ".join((spec.description or "").split())
        if len(description) > 360:
            description = description[:357] + "..."
        lines.append(
            f"{index}. {spec.name} | server={spec.server} | category={spec.category} | "
            f"risk={spec.risk_level} | args={args}\n"
            f"   {description or 'No description provided by MCP server.'}"
        )
    return "\n".join(lines)


def select_relevant_tools(question: str, limit: int = DEFAULT_TOOL_CONTEXT_LIMIT) -> list[ToolSpec]:
    registry = get_tool_registry()
    question_tokens = tokenize(question)
    selected_names = set(core_tool_names(question))

    scored: list[tuple[float, ToolSpec]] = []
    for spec in registry.values():
        score = tool_relevance_score(spec, question, question_tokens)
        if spec.name in selected_names:
            score += 2.0
        scored.append((score, spec))

    scored.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    selected: list[ToolSpec] = []
    seen: set[str] = set()
    for score, spec in scored:
        if spec.name in seen:
            continue
        if score <= 0 and spec.name not in selected_names:
            continue
        selected.append(spec)
        seen.add(spec.name)
        if len(selected) >= limit:
            break
    return selected


def get_tool_spec(tool_name: str) -> ToolSpec | None:
    return get_tool_registry().get(tool_name)


def missing_required_args(tool_name: str, arguments: dict[str, Any]) -> list[str]:
    spec = get_tool_spec(tool_name)
    if spec is None:
        return []
    required = spec.input_schema.get("required", [])
    if not isinstance(required, list):
        return []
    return [str(name) for name in required if str(name) not in arguments]


@lru_cache(maxsize=1)
def get_tool_registry() -> dict[str, ToolSpec]:
    cached = load_tool_registry_cache(TOOL_REGISTRY_CACHE_PATH, TOOL_REGISTRY_CACHE_TTL_SECONDS)
    if cached:
        return cached

    specs: dict[str, ToolSpec] = {}
    for server, discover, fallback in (
        (SERVER_DSE, discover_dse_tools, fallback_dse_tools),
        (SERVER_FS, discover_fs_tools, fallback_fs_tools),
        (SERVER_GIT, discover_git_tools, fallback_git_tools),
        (SERVER_GITHUB, discover_github_tools, fallback_github_tools),
    ):
        server_specs = discover_with_fallback(server, discover, fallback)
        for spec in server_specs:
            specs[spec.name] = spec
    save_tool_registry_cache(TOOL_REGISTRY_CACHE_PATH, specs)
    return specs


def load_tool_registry_cache(path: Path, ttl_seconds: int) -> dict[str, ToolSpec]:
    if ttl_seconds <= 0 or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("version", 0)) != TOOL_REGISTRY_CACHE_VERSION:
            return {}
        created_at = float(payload.get("created_at", 0.0))
        if time.time() - created_at > ttl_seconds:
            return {}
        tools = payload.get("tools", [])
        specs: dict[str, ToolSpec] = {}
        for item in tools:
            if not isinstance(item, dict):
                continue
            spec = ToolSpec(
                name=str(item.get("name", "")),
                server=str(item.get("server", SERVER_DSE)),
                description=str(item.get("description", "")),
                input_schema=item.get("input_schema") if isinstance(item.get("input_schema"), dict) else {},
                category=str(item.get("category", "research")),
                risk_level=str(item.get("risk_level", "read")),
            )
            if spec.name:
                specs[spec.name] = spec
        if specs:
            logger.info("Loaded %s MCP tool specs from cache.", len(specs))
        return specs
    except Exception:
        logger.exception("Failed to load MCP tool registry cache.")
        return {}


def save_tool_registry_cache(path: Path, specs: dict[str, ToolSpec]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": TOOL_REGISTRY_CACHE_VERSION,
            "created_at": time.time(),
            "tools": [
                {
                    "name": spec.name,
                    "server": spec.server,
                    "description": spec.description,
                    "input_schema": spec.input_schema,
                    "category": spec.category,
                    "risk_level": spec.risk_level,
                }
                for spec in sorted(specs.values(), key=lambda item: (item.server, item.name))
            ],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        logger.exception("Failed to save MCP tool registry cache.")


def discover_with_fallback(
    server: str,
    discover: Callable[[], list[ToolSpec]],
    fallback: Callable[[], list[ToolSpec]],
) -> list[ToolSpec]:
    try:
        specs = discover()
        if specs:
            logger.info("Discovered %s tools from %s MCP server", len(specs), server)
            return specs
    except Exception:
        logger.exception("Failed to discover %s MCP tools; using static fallback.", server)
    return fallback()


def discover_dse_tools() -> list[ToolSpec]:
    from utils.mcp_client import list_mcp_tools

    return [tool_to_spec(tool, SERVER_DSE) for tool in list_mcp_tools()]


def discover_fs_tools() -> list[ToolSpec]:
    from utils.fs_mcp_client import list_fs_tools

    return [tool_to_spec(tool, SERVER_FS) for tool in list_fs_tools()]


def discover_git_tools() -> list[ToolSpec]:
    from utils.git_mcp_client import list_git_tools

    return [tool_to_spec(tool, SERVER_GIT) for tool in list_git_tools()]


def discover_github_tools() -> list[ToolSpec]:
    from utils.github_mcp_client import list_github_tools

    return [tool_to_spec(tool, SERVER_GITHUB) for tool in list_github_tools()]


def tool_to_spec(tool: Any, server: str) -> ToolSpec:
    name = str(getattr(tool, "name", "") or "")
    description = str(getattr(tool, "description", "") or "")
    schema = getattr(tool, "inputSchema", None)
    if schema is None:
        schema = getattr(tool, "input_schema", None)
    if schema is None:
        schema = {}
    if not isinstance(schema, dict):
        schema = {}
    return ToolSpec(
        name=name,
        server=server,
        description=description,
        input_schema=schema,
        category=infer_category(name, description, server),
        risk_level=infer_risk_level(name, description, server),
    )


def tool_relevance_score(spec: ToolSpec, question: str, question_tokens: set[str]) -> float:
    text_tokens = tokenize(spec.searchable_text)
    overlap = len(question_tokens & text_tokens)
    score = float(overlap)
    lowered = question.lower()
    name = spec.name.lower()
    category = spec.category

    if category == "research" and any(term in lowered for term in research_terms()):
        score += 2.0
    if category == "developer" and any(term in lowered for term in developer_terms()):
        score += 2.0
    if category == "memory" and any(term in lowered for term in memory_terms()):
        score += 1.5
    if category == "file" and any(term in lowered for term in file_terms()):
        score += 2.0
    if category in {"git", "github"} and any(term in lowered for term in repo_terms()):
        score += 2.0
    if "ppa" in lowered and "ppa" in spec.searchable_text.lower():
        score += 3.0
    if "paper" in lowered or "论文" in question:
        if any(term in name for term in ("paper", "dse", "method", "rag", "research")):
            score += 1.5
    if "plot" in lowered or "画" in question or "绘" in question:
        if "pareto" in name or "plot" in spec.searchable_text.lower():
            score += 2.0
    return score


def core_tool_names(question: str) -> set[str]:
    lowered = question.lower()
    names = {
        "search_dse_papers",
        "search_memory",
        "get_recent_conversations",
        "get_rag_corpus_stats",
    }
    if "paper" in lowered or "论文" in question:
        names.update({"summarize_paper", "ask_paper", "find_related_papers"})
    if "ppa" in lowered or "power" in lowered or "area" in lowered:
        names.update({"generate_ppa_method_matrix", "synthesize_ppa_literature_review"})
    if any(term in lowered for term in developer_terms()):
        names.update({"execute_python_code", "compute_dse_metrics", "generate_pareto_front"})
    if any(term in lowered for term in file_terms()):
        names.update({"read_file", "search_files", "list_directory"})
    if any(term in lowered for term in repo_terms()):
        names.update({"git_status", "git_diff", "git_log", "git_show"})
    return names


def summarize_schema(schema: dict[str, Any]) -> str:
    properties = schema.get("properties")
    required = set(schema.get("required", []))
    if isinstance(properties, dict) and properties:
        parts: list[str] = []
        for name, detail in properties.items():
            if not isinstance(detail, dict):
                parts.append(str(name))
                continue
            type_name = detail.get("type") or detail.get("anyOf") or detail.get("oneOf") or "any"
            if isinstance(type_name, list):
                type_text = "|".join(str(item) for item in type_name)
            else:
                type_text = short_type_text(type_name)
            marker = " required" if name in required else ""
            parts.append(f"{name} ({type_text}{marker})")
        return ", ".join(parts) if parts else "(no args)"
    if schema:
        return json.dumps(schema, ensure_ascii=False, sort_keys=True)[:240]
    return "(no args)"


def short_type_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            if isinstance(item, dict) and "type" in item:
                texts.append(str(item["type"]))
            else:
                texts.append(str(item))
        return "|".join(texts)
    return str(value)[:80]


def infer_category(name: str, description: str, server: str) -> str:
    lowered = f"{name} {description}".lower()
    if server == SERVER_FS:
        return "file"
    if server == SERVER_GIT:
        return "git"
    if server == SERVER_GITHUB:
        return "github"
    if "memory" in lowered or "conversation" in lowered:
        return "memory"
    if any(term in lowered for term in ("web", "download", "fetch")):
        return "web"
    if any(term in lowered for term in ("paper", "papers", "rag", "method matrix", "literature", "research", "summary", "related")):
        return "research"
    if any(term in lowered for term in ("metric", "pareto", "python", "code", "experiment")):
        return "developer"
    return "research"


def infer_risk_level(name: str, description: str, server: str) -> str:
    lowered = f"{name} {description}".lower()
    if any(term in lowered for term in ("write", "edit", "commit", "create", "delete", "move", "push", "merge", "execute")):
        return "write_or_execute"
    if server == SERVER_GITHUB and any(term in lowered for term in ("issue", "pull", "repository", "branch")):
        return "external"
    if any(term in lowered for term in ("download", "fetch", "web", "search_research_web")):
        return "external"
    return "read"


def research_terms() -> tuple[str, ...]:
    return ("paper", "papers", "literature", "method", "compare", "dse", "moo", "论文", "方法", "对比", "综述")


def developer_terms() -> tuple[str, ...]:
    return ("compute", "metric", "hv", "igd", "pareto", "plot", "python", "code", "experiment", "计算", "画", "绘", "实验")


def memory_terms() -> tuple[str, ...]:
    return ("remember", "memory", "previous", "last", "session", "刚才", "之前", "上次", "记忆")


def file_terms() -> tuple[str, ...]:
    return ("file", "directory", "read", "write", "edit", "path", "文件", "目录", "读取", "修改")


def repo_terms() -> tuple[str, ...]:
    return ("git", "commit", "diff", "branch", "github", "repo", "issue", "pull request", "提交", "分支")


def fallback_dse_tools() -> list[ToolSpec]:
    return [
        fallback_tool("search_memory", SERVER_DSE, "Search conversation history and long-term memories.", {"query": "string", "session_id": "string", "kind": "string", "limit": "integer"}, "memory", "read"),
        fallback_tool("save_memory", SERVER_DSE, "Save useful information to long-term memory.", {"content": "string", "kind": "string", "session_id": "string", "importance": "number"}, "memory", "write_or_execute"),
        fallback_tool("get_recent_conversations", SERVER_DSE, "Return recent conversation turns for a session.", {"session_id": "string", "limit": "integer"}, "memory", "read"),
        fallback_tool("get_memory_stats", SERVER_DSE, "Return memory statistics.", {"session_id": "string"}, "memory", "read"),
        fallback_tool("search_dse_papers", SERVER_DSE, "Search local DSE/MOO and PPA surrogate paper chunks.", {"query": "string", "query_variants": "array", "top_k": "integer"}, "research", "read"),
        fallback_tool("search_research_web", SERVER_DSE, "Search the web for DSE/combinatorial optimization research.", {"query": "string", "topic": "string"}, "web", "external"),
        fallback_tool("fetch_web_page", SERVER_DSE, "Fetch a web page as markdown or text.", {"url": "string", "output_format": "string", "max_chars": "integer"}, "web", "external"),
        fallback_tool("download_paper_pdf", SERVER_DSE, "Download a paper PDF into the local paper directory.", {"url": "string", "filename": "string", "category": "string"}, "web", "external"),
        fallback_tool("summarize_paper", SERVER_DSE, "Summarize a specific local paper.", {"paper_path": "string"}, "research", "read"),
        fallback_tool("ask_paper", SERVER_DSE, "Ask a question about one specific local paper.", {"paper_path": "string", "question": "string"}, "research", "read"),
        fallback_tool("find_related_papers", SERVER_DSE, "Find papers related to a specific local paper.", {"paper_path": "string"}, "research", "read"),
        fallback_tool("generate_dse_method_matrix", SERVER_DSE, "Generate a DSE method comparison matrix.", {"keyword": "string", "category": "string", "method_names": "array"}, "research", "read"),
        fallback_tool("generate_ppa_method_matrix", SERVER_DSE, "Generate a PPA surrogate method comparison matrix.", {"keyword": "string", "model_family": "string", "target": "string"}, "research", "read"),
        fallback_tool("synthesize_ppa_literature_review", SERVER_DSE, "Synthesize a PPA surrogate literature review.", {"topic": "string", "keyword": "string", "model_family": "string", "target": "string"}, "research", "read"),
        fallback_tool("get_rag_corpus_stats", SERVER_DSE, "Return local RAG corpus statistics.", {}, "research", "read"),
        fallback_tool("execute_python_code", SERVER_DSE, "Execute a Python script in a sandbox.", {"code": "string", "timeout": "integer"}, "developer", "write_or_execute"),
        fallback_tool("compute_dse_metrics", SERVER_DSE, "Compute DSE/MOO metrics such as HV, IGD, GD, Spread.", {"points": "array", "reference_point": "array", "true_pareto_front": "array"}, "developer", "read"),
        fallback_tool("generate_pareto_front", SERVER_DSE, "Generate a Pareto-front plot from points.", {"points": "array", "labels": "array", "true_pareto_front": "array"}, "developer", "write_or_execute"),
        fallback_tool("design_experiment_template", SERVER_DSE, "Design an experiment template for a DSE problem.", {"problem_description": "string", "num_design_variables": "integer", "num_objectives": "integer", "variable_type": "string"}, "developer", "read"),
    ]


def fallback_fs_tools() -> list[ToolSpec]:
    return [
        fallback_tool("read_file", SERVER_FS, "Read a file under the allowed directories.", {"path": "string"}, "file", "read"),
        fallback_tool("write_file", SERVER_FS, "Write a file under the allowed directories.", {"path": "string", "content": "string"}, "file", "write_or_execute"),
        fallback_tool("edit_file", SERVER_FS, "Edit a file using text replacements.", {"path": "string", "edits": "array", "dryRun": "boolean"}, "file", "write_or_execute"),
        fallback_tool("list_directory", SERVER_FS, "List a directory.", {"path": "string"}, "file", "read"),
        fallback_tool("directory_tree", SERVER_FS, "Return a recursive directory tree.", {"path": "string"}, "file", "read"),
        fallback_tool("search_files", SERVER_FS, "Search files by pattern.", {"path": "string", "pattern": "string", "excludePatterns": "array"}, "file", "read"),
        fallback_tool("get_file_info", SERVER_FS, "Return file metadata.", {"path": "string"}, "file", "read"),
        fallback_tool("list_allowed_directories", SERVER_FS, "List filesystem roots allowed for this server.", {}, "file", "read"),
    ]


def fallback_git_tools() -> list[ToolSpec]:
    return [
        fallback_tool("git_status", SERVER_GIT, "Show repository status.", {"repo_path": "string"}, "git", "read"),
        fallback_tool("git_diff_unstaged", SERVER_GIT, "Show unstaged diff.", {"repo_path": "string"}, "git", "read"),
        fallback_tool("git_diff_staged", SERVER_GIT, "Show staged diff.", {"repo_path": "string"}, "git", "read"),
        fallback_tool("git_diff", SERVER_GIT, "Show git diff for a target.", {"target": "string", "repo_path": "string"}, "git", "read"),
        fallback_tool("git_log", SERVER_GIT, "Show commit history.", {"max_count": "integer", "repo_path": "string"}, "git", "read"),
        fallback_tool("git_show", SERVER_GIT, "Show a commit or revision.", {"rev": "string", "repo_path": "string"}, "git", "read"),
        fallback_tool("git_branch", SERVER_GIT, "List branches.", {"branch_type": "string", "repo_path": "string"}, "git", "read"),
        fallback_tool("git_add", SERVER_GIT, "Stage files.", {"files": "array", "repo_path": "string"}, "git", "write_or_execute"),
        fallback_tool("git_commit", SERVER_GIT, "Commit staged changes.", {"message": "string", "repo_path": "string"}, "git", "write_or_execute"),
        fallback_tool("git_checkout", SERVER_GIT, "Checkout a branch.", {"branch": "string", "repo_path": "string"}, "git", "write_or_execute"),
        fallback_tool("git_create_branch", SERVER_GIT, "Create a branch.", {"branch_name": "string", "base_branch": "string", "repo_path": "string"}, "git", "write_or_execute"),
    ]


def fallback_github_tools() -> list[ToolSpec]:
    return [
        fallback_tool("search_repositories", SERVER_GITHUB, "Search GitHub repositories.", {"query": "string", "page": "integer", "perPage": "integer"}, "github", "external"),
        fallback_tool("get_file_contents", SERVER_GITHUB, "Read a file from a GitHub repository.", {"owner": "string", "repo": "string", "path": "string", "branch": "string"}, "github", "external"),
        fallback_tool("list_commits", SERVER_GITHUB, "List repository commits.", {"owner": "string", "repo": "string", "sha": "string", "page": "integer", "perPage": "integer"}, "github", "external"),
        fallback_tool("list_issues", SERVER_GITHUB, "List repository issues.", {"owner": "string", "repo": "string", "state": "string", "page": "integer", "perPage": "integer"}, "github", "external"),
        fallback_tool("get_issue", SERVER_GITHUB, "Get a GitHub issue.", {"owner": "string", "repo": "string", "issue_number": "integer"}, "github", "external"),
        fallback_tool("get_pull_request", SERVER_GITHUB, "Get a pull request.", {"owner": "string", "repo": "string", "pull_number": "integer"}, "github", "external"),
        fallback_tool("list_pull_requests", SERVER_GITHUB, "List pull requests.", {"owner": "string", "repo": "string", "state": "string", "page": "integer", "perPage": "integer"}, "github", "external"),
        fallback_tool("search_code", SERVER_GITHUB, "Search GitHub code.", {"query": "string", "page": "integer", "perPage": "integer"}, "github", "external"),
        fallback_tool("search_issues", SERVER_GITHUB, "Search GitHub issues and pull requests.", {"query": "string", "page": "integer", "perPage": "integer"}, "github", "external"),
    ]


def fallback_tool(
    name: str,
    server: str,
    description: str,
    args: dict[str, str],
    category: str,
    risk_level: str,
) -> ToolSpec:
    properties = {key: {"type": value} for key, value in args.items()}
    return ToolSpec(
        name=name,
        server=server,
        description=description,
        input_schema={"type": "object", "properties": properties},
        category=category,
        risk_level=risk_level,
    )
