"""Memory tools — expose conversation and long-term memory as MCP tools."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from utils.config import get_settings
from utils.logger import get_logger
from utils.memory import (
    MEMORY_DB_PATH,
    MemoryStore,
    extract_json,
    format_sources,
    tokenize,
    utc_now,
)

logger = get_logger(__name__)


def _store() -> MemoryStore:
    return MemoryStore(MEMORY_DB_PATH)


def search_memory(
    query: str = "",
    session_id: str = "default",
    kind: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Search conversation history and long-term memories.

    Args:
        query: Search query. Empty string returns recent memories unfiltered.
        session_id: Session to search, or "all" for all sessions.
        kind: Filter by kind: conclusion, user_focus, paper_summary. None means all.
        limit: Maximum results, capped at 30.

    Returns:
        A dictionary with ``query``, ``results`` list, and ``result_count``.
    """
    safe_limit = min(max(1, limit), 30)
    search_text = (query or "").strip()
    safe_kind = (kind or "").strip()
    store = _store()

    # Search long-term memories
    with store._connect() as conn:
        where_parts = ["1=1"]
        params: list[Any] = []
        if session_id != "all":
            where_parts.append("session_id = ?")
            params.append(session_id)
        if safe_kind:
            where_parts.append("kind = ?")
            params.append(safe_kind)

        rows = conn.execute(
            f"SELECT session_id, kind, content, source, importance, created_at "
            f"FROM long_term_memories WHERE {' AND '.join(where_parts)} "
            f"ORDER BY importance DESC, id DESC LIMIT 500",
            params,
        ).fetchall()

    # Token-match scoring
    query_tokens = tokenize(search_text)
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        content_tokens = tokenize(row["content"])
        overlap = len(query_tokens & content_tokens) if query_tokens else 1
        score = overlap + float(row["importance"])
        scored.append((score, {
            "kind": row["kind"],
            "session_id": row["session_id"],
            "content": row["content"],
            "source": row["source"],
            "importance": row["importance"],
            "created_at": row["created_at"],
            "score": round(score, 2),
        }))

    scored.sort(key=lambda item: item[0], reverse=True)
    results = [item[1] for item in scored[:safe_limit]]

    # Also search conversation turn summaries
    with store._connect() as conn:
        conv_params: list[Any] = []
        conv_where = "1=1"
        if session_id != "all":
            conv_where = "session_id = ?"
            conv_params.append(session_id)
        conv_rows = conn.execute(
            f"SELECT session_id, question, summary, created_at "
            f"FROM conversation_turns WHERE {conv_where} "
            f"ORDER BY id DESC LIMIT 200",
            conv_params,
        ).fetchall()

    for row in conv_rows:
        if len(results) >= safe_limit:
            break
        text = f"{row['question']} {row['summary']}"
        tokens = tokenize(text)
        overlap = len(query_tokens & tokens) if query_tokens else 0
        if overlap > 0:
            results.append({
                "kind": "conversation_turn",
                "session_id": row["session_id"],
                "content": row["summary"],
                "source": f"question: {row['question'][:200]}",
                "importance": 0.5,
                "created_at": row["created_at"],
                "score": round(float(overlap), 2),
            })

    results.sort(key=lambda r: r["score"], reverse=True)
    results = results[:safe_limit]

    return {
        "success": True,
        "query": search_text,
        "result_count": len(results),
        "results": results,
    }


def save_memory(
    content: str,
    kind: str | None = None,
    session_id: str = "default",
    source: str = "agent",
    importance: float = 0.7,
) -> dict[str, Any]:
    """Save a piece of information to long-term memory.

    Args:
        content: The information to remember.
        kind: conclusion, user_focus, or paper_summary.
        session_id: Session to save under.
        source: Where this memory came from.
        importance: 0.0 (low) to 1.0 (high), default 0.7.

    Returns:
        A dictionary with success status.
    """
    safe_kind = (kind or "conclusion").strip()
    if safe_kind not in ("conclusion", "user_focus", "paper_summary"):
        return {"success": False, "error": f"Invalid kind: {safe_kind}"}

    content = content.strip()[:1200]
    if not content:
        return {"success": False, "error": "content must not be empty."}

    store = _store()
    store.save_long_term_memory(
        session_id=session_id,
        kind=safe_kind,
        content=content,
        source=source[:300],
        importance=max(0.0, min(1.0, importance)),
    )
    return {
        "success": True,
        "message": f"Saved {safe_kind} memory to session {session_id}.",
        "content_preview": content[:200],
    }


def get_recent_conversations(
    session_id: str = "default",
    limit: int = 5,
) -> dict[str, Any]:
    """Return recent conversation turns for a session.

    Args:
        session_id: Session to query.
        limit: Number of recent turns, capped at 20.

    Returns:
        A dictionary with ``turns`` list and ``turn_count``.
    """
    safe_limit = min(max(1, limit), 20)
    store = _store()
    context_text = store.get_recent_conversation_context(session_id, safe_limit)
    if not context_text:
        return {"success": True, "turns": [], "turn_count": 0}

    # Parse the formatted context back into structured data
    turns: list[dict[str, Any]] = []
    for block in context_text.split("\n\n"):
        if block.strip():
            turns.append({"content": block.strip()})

    return {
        "success": True,
        "session_id": session_id,
        "turn_count": len(turns),
        "turns": turns,
    }


def get_memory_stats(session_id: str = "default") -> dict[str, Any]:
    """Return statistics about stored memories.

    Args:
        session_id: Session to query, or "all" for all sessions.

    Returns:
        A dictionary with counts by kind and session.
    """
    store = _store()
    with store._connect() as conn:
        params: list[Any] = []
        where = "1=1"
        if session_id != "all":
            where = "session_id = ?"
            params.append(session_id)

        mem_count = conn.execute(
            f"SELECT COUNT(*) FROM long_term_memories WHERE {where}", params
        ).fetchone()[0]

        turn_count = conn.execute(
            f"SELECT COUNT(*) FROM conversation_turns WHERE {where}", params
        ).fetchone()[0]

        kind_counts = conn.execute(
            f"SELECT kind, COUNT(*) as cnt FROM long_term_memories WHERE {where} GROUP BY kind", params
        ).fetchall()

    return {
        "success": True,
        "session_id": session_id,
        "long_term_memories": mem_count,
        "conversation_turns": turn_count,
        "by_kind": {row["kind"]: row["cnt"] for row in kind_counts},
    }


def register_memory_tools(mcp: Any) -> None:
    """Register memory tools on a FastMCP server instance."""

    mcp.tool(
        name="search_memory",
        description=(
            "Search the agent's conversation history and long-term memories. "
            "Use this to recall what was discussed in previous turns, find stored "
            "conclusions, user preferences, or paper summaries."
        ),
    )(search_memory)

    mcp.tool(
        name="save_memory",
        description=(
            "Save a piece of information to long-term memory. Use this to "
            "remember important conclusions, user preferences, or paper takeaways "
            "for future conversations."
        ),
    )(save_memory)

    mcp.tool(
        name="get_recent_conversations",
        description=(
            "Return recent conversation turns for a session. Use this to review "
            "what was recently discussed."
        ),
    )(get_recent_conversations)

    mcp.tool(
        name="get_memory_stats",
        description=(
            "Return statistics about stored memories: how many conversations, "
            "long-term memories, and breakdown by kind."
        ),
    )(get_memory_stats)
