"""Conversation and long-term memory backed by local SQLite."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from utils.config import PROJECT_ROOT
from utils.deepseek_llm import invoke_deepseek
from utils.logger import get_logger


logger = get_logger(__name__)

MEMORY_DIR = PROJECT_ROOT / "memory"
MEMORY_DB_PATH = MEMORY_DIR / "dse_explorer_memory.sqlite3"


@dataclass(frozen=True)
class MemoryContext:
    conversation_context: str
    long_term_context: str


class MemoryStore:
    def __init__(self, db_path: Path = MEMORY_DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    summary TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    importance REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_turns_session_created ON conversation_turns(session_id, created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mem_session_kind ON long_term_memories(session_id, kind)"
            )

    def build_context(self, session_id: str, query: str, recent_turns: int = 3) -> MemoryContext:
        return MemoryContext(
            conversation_context=self.get_recent_conversation_context(session_id, recent_turns),
            long_term_context=self.search_long_term_context(session_id, query, limit=5),
        )

    def get_recent_conversation_context(self, session_id: str, limit: int = 3) -> str:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT question, answer, sources_json, summary, created_at
                FROM conversation_turns
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        if not rows:
            return ""

        blocks: list[str] = []
        for index, row in enumerate(reversed(rows), start=1):
            sources = json.loads(row["sources_json"] or "[]")
            source_labels = [
                f"{item.get('source', 'unknown')} page {item.get('page', '?')}"
                for item in sources[:4]
            ]
            blocks.append(
                f"Turn {index} ({row['created_at']}):\n"
                f"User: {row['question']}\n"
                f"Summary: {row['summary']}\n"
                f"Sources: {', '.join(source_labels) if source_labels else 'none'}"
            )
        return "\n\n".join(blocks)

    def search_long_term_context(self, session_id: str, query: str, limit: int = 5) -> str:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT kind, content, source, importance, created_at
                FROM long_term_memories
                WHERE session_id IN (?, 'global')
                ORDER BY importance DESC, id DESC
                LIMIT 200
                """,
                (session_id,),
            ).fetchall()

        if not rows:
            return ""

        query_tokens = tokenize(query)
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            content_tokens = tokenize(row["content"])
            overlap = len(query_tokens & content_tokens)
            if overlap == 0 and query_tokens:
                continue
            score = overlap + float(row["importance"])
            scored.append((score, row))

        if not scored:
            scored = [(float(row["importance"]), row) for row in rows[:limit]]

        scored.sort(key=lambda item: item[0], reverse=True)
        blocks = [
            f"- [{row['kind']}] {row['content']} (source: {row['source']}, importance: {row['importance']:.2f})"
            for _, row in scored[:limit]
        ]
        return "\n".join(blocks)

    def save_turn(
        self,
        *,
        session_id: str,
        question: str,
        answer: str,
        sources: list[dict[str, Any]],
    ) -> None:
        summary = summarize_turn(question, answer, sources)
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_turns(session_id, created_at, question, answer, sources_json, summary)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, now, question, answer, json.dumps(sources, ensure_ascii=False), summary),
            )

        for memory in extract_long_term_memories(question, answer, sources):
            self.save_long_term_memory(
                session_id=session_id,
                kind=memory["kind"],
                content=memory["content"],
                source=memory["source"],
                importance=float(memory["importance"]),
            )

    def save_long_term_memory(
        self,
        *,
        session_id: str,
        kind: str,
        content: str,
        source: str,
        importance: float,
    ) -> None:
        content = content.strip()
        if not content:
            return

        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT id FROM long_term_memories
                WHERE session_id = ? AND kind = ? AND content = ?
                LIMIT 1
                """,
                (session_id, kind, content),
            ).fetchone()
            if existing:
                return
            conn.execute(
                """
                INSERT INTO long_term_memories(session_id, created_at, kind, content, source, importance)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, utc_now(), kind, content, source, max(0.0, min(1.0, importance))),
            )


def summarize_turn(question: str, answer: str, sources: list[dict[str, Any]]) -> str:
    source_text = format_sources(sources[:5])
    prompt = f"""请用中文把这一轮问答压缩成 2-3 句话，保留用户关注点、核心结论和关键论文引用。

用户问题：
{question}

回答：
{answer[:3000]}

引用来源：
{source_text}
"""
    try:
        return invoke_deepseek(prompt).strip()[:1000]
    except Exception:
        logger.exception("Failed to summarize turn; using fallback summary.")
        return fallback_summary(question, answer)


def extract_long_term_memories(
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_text = format_sources(sources[:8])
    prompt = f"""请从下面问答中提取值得长期保存的记忆，返回严格 JSON。

JSON schema:
{{
  "memories": [
    {{
      "kind": "conclusion|user_focus|paper_summary",
      "content": "一条可复用的中文记忆",
      "source": "论文或会话来源",
      "importance": 0.0
    }}
  ]
}}

规则：
- 最多 5 条。
- 只保存对后续研究问答有用的高价值结论。
- importance 取 0.5 到 1.0。
- 不要输出 markdown。

用户问题：
{question}

回答：
{answer[:3500]}

引用来源：
{source_text}
"""
    try:
        raw = invoke_deepseek(prompt)
        payload = json.loads(extract_json(raw))
        memories = payload.get("memories", [])
        if isinstance(memories, list):
            return [normalize_memory_item(item) for item in memories if isinstance(item, dict)]
    except Exception:
        logger.exception("Failed to extract long-term memories; using fallback memory.")

    return [
        {
            "kind": "conclusion",
            "content": fallback_summary(question, answer),
            "source": "conversation",
            "importance": 0.6,
        }
    ]


def normalize_memory_item(item: dict[str, Any]) -> dict[str, Any]:
    kind = str(item.get("kind", "conclusion")).strip()
    if kind not in {"conclusion", "user_focus", "paper_summary"}:
        kind = "conclusion"
    return {
        "kind": kind,
        "content": str(item.get("content", "")).strip()[:1200],
        "source": str(item.get("source", "conversation")).strip()[:300],
        "importance": float(item.get("importance", 0.6) or 0.6),
    }


def fallback_summary(question: str, answer: str) -> str:
    compact_answer = " ".join(answer.split())
    return f"用户询问：{question}。核心回答：{compact_answer[:500]}"


def format_sources(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "无"
    return "\n".join(
        f"- {item.get('source', 'unknown')} page {item.get('page', '?')} score={item.get('score', 0)}"
        for item in sources
    )


def extract_json(raw: str) -> str:
    text = raw.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found.")
    return match.group(0)


def tokenize(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]{2,}", text)
        if len(token.strip()) >= 2
    }


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
