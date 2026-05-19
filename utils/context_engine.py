"""Context collection, scoring, selection, and node routing."""

from __future__ import annotations

import json
import math
import sqlite3
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from utils.config import get_settings
from utils.logger import get_logger
from utils.memory import MemoryStore, tokenize, utc_now


logger = get_logger(__name__)


@dataclass(frozen=True)
class ContextCandidate:
    source_type: str
    content: str
    metadata: dict[str, Any]
    relevance: float
    recency: float
    importance: float
    confidence: float
    token_estimate: int

    @property
    def score(self) -> float:
        return (
            0.45 * self.relevance
            + 0.20 * self.importance
            + 0.15 * self.recency
            + 0.15 * self.confidence
        )


@dataclass(frozen=True)
class ContextBundle:
    conversation_context: str
    long_term_context: str
    planner_context: str
    rewriter_context: str
    reasoner_context: str
    synthesis_context: str
    context_trace: list[dict[str, Any]]


def build_context_bundle(
    store: MemoryStore,
    *,
    session_id: str,
    query: str,
    recent_turns: int = 3,
    use_embeddings: bool = True,
) -> ContextBundle:
    """Build backward-compatible and node-specific context for an agent run."""
    candidates = collect_context_candidates(
        store,
        session_id=session_id,
        query=query,
        recent_turns=recent_turns,
    )
    if use_embeddings:
        candidates = enhance_relevance_with_embeddings(store, query, candidates)
    selected = select_candidates(candidates, max_candidates=12)
    trace = render_context_trace(candidates, selected)

    return ContextBundle(
        conversation_context=render_context(
            selected,
            allowed_types={"conversation"},
            title="Selected conversation context",
            budget_tokens=900,
        ),
        long_term_context=render_context(
            selected,
            allowed_types={"long_term_memory"},
            title="Selected long-term memory",
            budget_tokens=900,
        ),
        planner_context=render_context(
            selected,
            allowed_types={"conversation", "long_term_memory"},
            title="Planner context",
            budget_tokens=1600,
            preferred_types=("long_term_memory", "conversation"),
        ),
        rewriter_context=render_context(
            selected,
            allowed_types={"conversation", "long_term_memory"},
            title="Query rewrite context",
            budget_tokens=1200,
            preferred_types=("conversation", "long_term_memory"),
        ),
        reasoner_context=render_context(
            selected,
            allowed_types={"conversation", "long_term_memory"},
            title="Reasoning context",
            budget_tokens=1400,
        ),
        synthesis_context=render_context(
            selected,
            allowed_types={"conversation", "long_term_memory"},
            title="Synthesis context",
            budget_tokens=1400,
        ),
        context_trace=trace,
    )


def collect_context_candidates(
    store: MemoryStore,
    *,
    session_id: str,
    query: str,
    recent_turns: int,
) -> list[ContextCandidate]:
    candidates: list[ContextCandidate] = []
    candidates.extend(collect_conversation_candidates(store, session_id, query, recent_turns))
    candidates.extend(collect_long_term_candidates(store, session_id, query))
    return candidates


def collect_conversation_candidates(
    store: MemoryStore,
    session_id: str,
    query: str,
    recent_turns: int,
    scan_limit: int = 80,
) -> list[ContextCandidate]:
    with store._connect() as conn:
        rows = conn.execute(
            """
            SELECT id, question, answer, sources_json, summary, created_at
            FROM conversation_turns
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, max(recent_turns, scan_limit)),
        ).fetchall()

    candidates: list[ContextCandidate] = []
    for position, row in enumerate(rows, start=1):
        sources = _load_sources(row)
        source_labels = [
            f"{item.get('source', 'unknown')} page {item.get('page', '?')}"
            for item in sources[:4]
            if isinstance(item, dict)
        ]
        content = (
            f"Conversation turn ({row['created_at']}):\n"
            f"User: {row['question']}\n"
            f"Summary: {row['summary']}\n"
            f"Sources: {', '.join(source_labels) if source_labels else 'none'}"
        )
        relevance = relevance_score(query, f"{row['question']} {row['summary']}")
        recency = recency_score(row["created_at"])
        if position <= recent_turns:
            recency = max(recency, 0.95)
            relevance = max(relevance, 0.35)
        candidates.append(
            ContextCandidate(
                source_type="conversation",
                content=content,
                metadata={
                    "id": row["id"],
                    "session_id": session_id,
                    "created_at": row["created_at"],
                    "recent_rank": position,
                    "lexical_relevance": round(relevance, 4),
                    "relevance_method": "lexical",
                },
                relevance=relevance,
                recency=recency,
                importance=0.55,
                confidence=0.75,
                token_estimate=estimate_tokens(content),
            )
        )
    return candidates


def collect_long_term_candidates(
    store: MemoryStore,
    session_id: str,
    query: str,
    scan_limit: int = 200,
) -> list[ContextCandidate]:
    with store._connect() as conn:
        rows = conn.execute(
            """
            SELECT id, session_id, kind, content, source, importance, created_at
            FROM long_term_memories
            WHERE session_id IN (?, 'global')
            ORDER BY importance DESC, id DESC
            LIMIT ?
            """,
            (session_id, scan_limit),
        ).fetchall()

    candidates: list[ContextCandidate] = []
    for row in rows:
        content = (
            f"[{row['kind']}] {row['content']}\n"
            f"Source: {row['source']}; importance={float(row['importance']):.2f}; "
            f"created_at={row['created_at']}"
        )
        candidates.append(
            ContextCandidate(
                source_type="long_term_memory",
                content=content,
                metadata={
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "kind": row["kind"],
                    "source": row["source"],
                    "created_at": row["created_at"],
                    "lexical_relevance": round(relevance_score(query, row["content"]), 4),
                    "relevance_method": "lexical",
                },
                relevance=relevance_score(query, row["content"]),
                recency=recency_score(row["created_at"]),
                importance=max(0.0, min(1.0, float(row["importance"]))),
                confidence=0.8,
                token_estimate=estimate_tokens(content),
            )
        )
    return candidates


def enhance_relevance_with_embeddings(
    store: MemoryStore,
    query: str,
    candidates: list[ContextCandidate],
) -> list[ContextCandidate]:
    """Blend lexical relevance with embedding cosine similarity when available."""
    if not query.strip() or not candidates:
        return candidates

    try:
        from utils.embeddings import EmbeddingClient

        settings = get_settings()
        embedder = EmbeddingClient(settings)
        model_key = (
            f"{settings.embedding.model_type}:"
            f"{embedder.model_name}:"
            f"{settings.qdrant.vector_size}"
        )
        query_vector = get_or_create_embedding(store, embedder, model_key, query)
        candidate_vectors = get_or_create_embeddings(
            store,
            embedder,
            model_key,
            [candidate.content for candidate in candidates],
        )
    except Exception:
        logger.exception("Embedding relevance failed; falling back to lexical relevance.")
        return candidates

    enhanced: list[ContextCandidate] = []
    for candidate, vector in zip(candidates, candidate_vectors, strict=True):
        semantic = normalized_cosine_similarity(query_vector, vector)
        lexical = float(candidate.metadata.get("lexical_relevance", candidate.relevance))
        relevance = max(0.0, min(1.0, 0.75 * semantic + 0.25 * lexical))
        metadata = {
            **candidate.metadata,
            "semantic_relevance": round(semantic, 4),
            "lexical_relevance": round(lexical, 4),
            "relevance_method": "embedding+lexical",
            "embedding_model": model_key,
        }
        enhanced.append(
            ContextCandidate(
                source_type=candidate.source_type,
                content=candidate.content,
                metadata=metadata,
                relevance=relevance,
                recency=candidate.recency,
                importance=candidate.importance,
                confidence=candidate.confidence,
                token_estimate=candidate.token_estimate,
            )
        )
    return enhanced


def get_or_create_embedding(
    store: MemoryStore,
    embedder: Any,
    model_key: str,
    text: str,
) -> list[float]:
    return get_or_create_embeddings(store, embedder, model_key, [text])[0]


def get_or_create_embeddings(
    store: MemoryStore,
    embedder: Any,
    model_key: str,
    texts: list[str],
) -> list[list[float]]:
    ensure_embedding_cache_table(store)
    cached = load_cached_embeddings(store, model_key, texts)
    missing_indices = [index for index, vector in enumerate(cached) if vector is None]
    if missing_indices:
        missing_texts = [texts[index] for index in missing_indices]
        vectors = embedder.embed_documents(missing_texts, batch_size=min(len(missing_texts), 10))
        save_cached_embeddings(store, model_key, missing_texts, vectors)
        for index, vector in zip(missing_indices, vectors, strict=True):
            cached[index] = vector
    return [vector for vector in cached if vector is not None]


def ensure_embedding_cache_table(store: MemoryStore) -> None:
    with store._connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS context_embedding_cache (
                model_key TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                text_preview TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (model_key, text_hash)
            )
            """
        )


def load_cached_embeddings(
    store: MemoryStore,
    model_key: str,
    texts: list[str],
) -> list[list[float] | None]:
    cached: list[list[float] | None] = []
    with store._connect() as conn:
        for text in texts:
            row = conn.execute(
                """
                SELECT vector_json FROM context_embedding_cache
                WHERE model_key = ? AND text_hash = ?
                LIMIT 1
                """,
                (model_key, text_hash(text)),
            ).fetchone()
            if not row:
                cached.append(None)
                continue
            try:
                vector = json.loads(row["vector_json"])
            except json.JSONDecodeError:
                cached.append(None)
                continue
            cached.append(vector if isinstance(vector, list) else None)
    return cached


def save_cached_embeddings(
    store: MemoryStore,
    model_key: str,
    texts: list[str],
    vectors: list[list[float]],
) -> None:
    with store._connect() as conn:
        for text, vector in zip(texts, vectors, strict=True):
            conn.execute(
                """
                INSERT OR REPLACE INTO context_embedding_cache(
                    model_key, text_hash, text_preview, vector_json, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    model_key,
                    text_hash(text),
                    " ".join(text.split())[:240],
                    json.dumps(vector),
                    utc_now(),
                ),
            )


def select_candidates(
    candidates: list[ContextCandidate],
    *,
    max_candidates: int,
) -> list[ContextCandidate]:
    selected: list[ContextCandidate] = []
    seen_fingerprints: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        fingerprint = context_fingerprint(candidate.content)
        if fingerprint in seen_fingerprints:
            continue
        if candidate.relevance <= 0 and candidate.recency < 0.9 and candidate.importance < 0.75:
            continue
        selected.append(candidate)
        seen_fingerprints.add(fingerprint)
        if len(selected) >= max_candidates:
            break
    return selected


def render_context(
    candidates: list[ContextCandidate],
    *,
    allowed_types: set[str],
    title: str,
    budget_tokens: int,
    preferred_types: tuple[str, ...] = (),
) -> str:
    filtered = [item for item in candidates if item.source_type in allowed_types]
    if preferred_types:
        priority = {source_type: index for index, source_type in enumerate(preferred_types)}
        filtered.sort(
            key=lambda item: (
                priority.get(item.source_type, len(priority)),
                -item.score,
            )
        )
    else:
        filtered.sort(key=lambda item: item.score, reverse=True)

    blocks: list[str] = []
    used_tokens = 0
    for item in filtered:
        block = (
            f"- type={item.source_type}; score={item.score:.2f}; "
            f"relevance={item.relevance:.2f}; recency={item.recency:.2f}; "
            f"importance={item.importance:.2f}\n{item.content}"
        )
        block_tokens = estimate_tokens(block)
        if used_tokens + block_tokens > budget_tokens and blocks:
            continue
        blocks.append(block)
        used_tokens += block_tokens
        if used_tokens >= budget_tokens:
            break

    if not blocks:
        return ""
    return f"{title}:\n" + "\n\n".join(blocks)


def render_context_trace(
    candidates: list[ContextCandidate],
    selected: list[ContextCandidate],
) -> list[dict[str, Any]]:
    selected_keys = {context_fingerprint(item.content) for item in selected}
    trace: list[dict[str, Any]] = []
    for item in sorted(candidates, key=lambda candidate: candidate.score, reverse=True):
        trace.append(
            {
                "selected": context_fingerprint(item.content) in selected_keys,
                "source_type": item.source_type,
                "score": round(item.score, 3),
                "relevance": round(item.relevance, 3),
                "recency": round(item.recency, 3),
                "importance": round(item.importance, 3),
                "confidence": round(item.confidence, 3),
                "token_estimate": item.token_estimate,
                "metadata": item.metadata,
                "preview": " ".join(item.content.split())[:220],
            }
        )
    return trace


def relevance_score(query: str, content: str) -> float:
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0
    content_tokens = tokenize(content)
    if not content_tokens:
        return 0.0
    overlap = len(query_tokens & content_tokens)
    if overlap == 0:
        return 0.0
    return min(1.0, overlap / math.sqrt(len(query_tokens) * len(content_tokens)))


def normalized_cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    cosine = dot / (left_norm * right_norm)
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def recency_score(created_at: str) -> float:
    try:
        created = datetime.fromisoformat(str(created_at))
    except ValueError:
        return 0.0
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    age_days = max(0.0, (datetime.now(UTC) - created.astimezone(UTC)).total_seconds() / 86400)
    return 1.0 / (1.0 + age_days / 14.0)


def estimate_tokens(text: str) -> int:
    # Conservative enough for routing budgets without adding a tokenizer dependency.
    return max(1, len(text) // 4)


def context_fingerprint(text: str) -> str:
    return " ".join(text.lower().split())[:500]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_sources(row: sqlite3.Row) -> list[dict[str, Any]]:
    try:
        sources = json.loads(row["sources_json"] or "[]")
    except json.JSONDecodeError:
        return []
    return sources if isinstance(sources, list) else []
