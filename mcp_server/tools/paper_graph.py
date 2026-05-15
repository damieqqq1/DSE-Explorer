"""Paper relationship graph tool — find related papers by semantic and metadata similarity."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from typing import Any

from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from rag_pipeline.ingest import create_qdrant_client
from rag_pipeline.paper_info import PAPER_INFO_DB_PATH
from utils.config import get_settings
from utils.embeddings import EmbeddingClient
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TOP_K = 10
MAX_TOP_K = 30
REPRESENTATIVE_MAX_CHARS = 8000
RELATIONSHIP_WEIGHTS = {
    "same_benchmark": 0.35,
    "similar_method": 0.30,
    "same_problem": 0.25,
    "same_category": 0.10,
}


def find_related_papers(
    paper_path: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Find papers related to a given paper using semantic and metadata similarity.

    Args:
        paper_path: Relative path or filename of the source paper, e.g. ``DSE/DAC23_GRL_DSE.pdf``.
        top_k: Number of related papers to return, capped at 30.

    Returns:
        A dictionary with the source paper info and a ranked list of related papers,
        each annotated with relationship types and shared aspects.
    """
    safe_top_k = min(max(1, top_k), MAX_TOP_K)
    settings = get_settings()
    client = create_qdrant_client(settings)
    embedder = EmbeddingClient(settings)

    # Resolve the paper
    source_info = _resolve_paper(client, settings, paper_path)
    if source_info is None:
        return {"success": False, "error": f"Paper not found in corpus: {paper_path}"}

    source_path = source_info["relative_path"]
    source_chunks = _scroll_paper_chunks(client, settings.qdrant.collection, source_path)

    if not source_chunks:
        return {"success": False, "error": f"No chunks found for paper: {paper_path}"}

    # Build representative text
    rep_text = _build_representative_text(source_chunks, REPRESENTATIVE_MAX_CHARS)
    if not rep_text.strip():
        return {"success": False, "error": "Could not extract representative text from paper."}

    # Embed and search for similar chunks, excluding the source paper itself
    rep_vector = embedder.embed_query(rep_text)
    exclude_filter = Filter(
        must=[
            FieldCondition(key="project", match=MatchValue(value="dse-explorer")),
        ],
        must_not=[
            FieldCondition(key="relative_path", match=MatchValue(value=source_path)),
        ],
    )

    search_result = client.query_points(
        collection_name=settings.qdrant.collection,
        query=rep_vector,
        limit=200,
        query_filter=exclude_filter,
        with_payload=True,
    )

    # Aggregate scores by paper
    paper_scores: dict[str, float] = defaultdict(float)
    paper_hit_counts: dict[str, int] = defaultdict(int)
    for point in search_result.points:
        payload = point.payload or {}
        rp = str(payload.get("relative_path", ""))
        if not rp:
            continue
        paper_scores[rp] += float(point.score)
        paper_hit_counts[rp] += 1

    # Normalize: average score weighted by log(hits) to favor multiple hits
    for rp in paper_scores:
        hits = paper_hit_counts[rp]
        paper_scores[rp] = paper_scores[rp] / (1 + hits) * (1 + min(hits, 10) * 0.5)

    ranked = sorted(paper_scores.items(), key=lambda item: item[1], reverse=True)[:safe_top_k + 5]

    # Look up metadata from paper_info SQLite
    paper_meta = _load_paper_metadata([rp for rp, _ in ranked])
    source_meta = paper_meta.get(source_path) or _empty_meta()

    # Classify relationships
    related: list[dict[str, Any]] = []
    for rp, score in ranked:
        if rp == source_path:
            continue
        meta = paper_meta.get(rp) or _empty_meta()
        rel = _classify_relationship(source_meta, meta)
        related.append({
            "paper_path": rp,
            "file_name": meta.get("file_name", rp.rsplit("/", 1)[-1]),
            "title": meta.get("title", ""),
            "method_name": meta.get("method_name", ""),
            "year": meta.get("year", ""),
            "relevance_score": round(score, 4),
            "primary_relationship": rel["primary"],
            "relationship_types": rel["types"],
            "shared_aspects": rel["shared"],
        })

    related.sort(key=lambda r: r["relevance_score"], reverse=True)

    return {
        "success": True,
        "source_paper": {
            "paper_path": source_path,
            "file_name": source_info["file_name"],
            "title": source_meta.get("title", ""),
            "method_name": source_meta.get("method_name", ""),
            "year": source_meta.get("year", ""),
        },
        "related_papers": related[:safe_top_k],
    }


def _resolve_paper(client: Any, settings: Any, paper_path: str) -> dict[str, Any] | None:
    """Find a paper by path or filename in Qdrant."""
    pf = Filter(
        must=[
            FieldCondition(key="project", match=MatchValue(value="dse-explorer")),
            FieldCondition(key="relative_path", match=MatchValue(value=paper_path)),
        ]
    )
    points, _ = client.scroll(
        collection_name=settings.qdrant.collection,
        scroll_filter=pf,
        limit=1,
        with_payload=True,
    )
    if points:
        payload = points[0].payload or {}
        return {"relative_path": str(payload.get("relative_path", "")), "file_name": str(payload.get("file_name", ""))}

    # Try filename match
    name = paper_path.replace("\\", "/").rsplit("/", 1)[-1]
    pf2 = Filter(
        must=[
            FieldCondition(key="project", match=MatchValue(value="dse-explorer")),
            FieldCondition(key="file_name", match=MatchValue(value=name)),
        ]
    )
    points, _ = client.scroll(
        collection_name=settings.qdrant.collection,
        scroll_filter=pf2,
        limit=1,
        with_payload=True,
    )
    if points:
        payload = points[0].payload or {}
        return {"relative_path": str(payload.get("relative_path", "")), "file_name": str(payload.get("file_name", ""))}

    return None


def _scroll_paper_chunks(client: Any, collection: str, paper_path: str) -> list[dict[str, Any]]:
    pf = Filter(
        must=[
            FieldCondition(key="project", match=MatchValue(value="dse-explorer")),
            FieldCondition(key="relative_path", match=MatchValue(value=paper_path)),
        ]
    )
    chunks: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection, scroll_filter=pf,
            limit=256, offset=offset, with_payload=True, with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            text = str(payload.get("text", ""))
            if text.strip():
                chunks.append({"page": payload.get("page", 0), "chunk_index": payload.get("chunk_index", 0), "text": text})
        if offset is None:
            break
    chunks.sort(key=lambda c: (c["page"], c["chunk_index"]))
    return chunks


def _build_representative_text(chunks: list[dict[str, Any]], max_chars: int) -> str:
    parts: list[str] = []
    total = 0
    # Take first 5 chunks + every 10th chunk to sample across the paper
    first_n = 5
    step = 10
    for i, chunk in enumerate(chunks):
        if i >= first_n and (i - first_n) % step != 0:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        t = chunk["text"]
        if total + len(t) > max_chars:
            parts.append(t[:remaining])
            break
        parts.append(t)
        total += len(t)
    return "\n\n".join(parts)


def _load_paper_metadata(paths: list[str]) -> dict[str, dict[str, Any]]:
    if not paths:
        return {}
    try:
        conn = sqlite3.connect(PAPER_INFO_DB_PATH)
        conn.row_factory = sqlite3.Row
        placeholders = ",".join(["?" for _ in paths])
        rows = conn.execute(
            f"SELECT relative_path, file_name, title, method_name, year, problem, optimization_algorithm, benchmarks_json, objectives_json FROM paper_info WHERE relative_path IN ({placeholders})",
            paths,
        ).fetchall()
        conn.close()
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            d = dict(row)
            rp = d.pop("relative_path")
            for key in ("benchmarks_json", "objectives_json"):
                try:
                    d[key.replace("_json", "s")] = json.loads(d.pop(key, "[]") or "[]")
                except json.JSONDecodeError:
                    d[key.replace("_json", "s")] = []
            result[rp] = d
        return result
    except Exception:
        logger.exception("Failed to load paper metadata")
        return {}


def _empty_meta() -> dict[str, Any]:
    return {
        "file_name": "", "title": "", "method_name": "", "year": "",
        "problem": "", "optimization_algorithm": "", "benchmarks": [], "objectives": [],
    }


def _classify_relationship(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    types: list[str] = []
    shared: dict[str, Any] = {}
    scores: dict[str, float] = {}

    # same_benchmark
    src_benchmarks = set(b.lower() for b in source.get("benchmarks", []))
    tgt_benchmarks = set(b.lower() for b in target.get("benchmarks", []))
    common_benchmarks = src_benchmarks & tgt_benchmarks
    if common_benchmarks:
        types.append("same_benchmark")
        shared["benchmarks"] = sorted(common_benchmarks)
        scores["same_benchmark"] = RELATIONSHIP_WEIGHTS["same_benchmark"]

    # similar_method (overlap in optimization algorithm keywords)
    src_algo = set(source.get("optimization_algorithm", "").lower().split())
    tgt_algo = set(target.get("optimization_algorithm", "").lower().split())
    common_algo = src_algo & tgt_algo - {"and", "or", "the", "with", "based", "for", "a", "an", "of", "in", "on", "to"}
    if common_algo:
        types.append("similar_method")
        shared["algorithm_terms"] = sorted(common_algo)
        scores["similar_method"] = RELATIONSHIP_WEIGHTS["similar_method"]

    # same_problem
    src_problem = set(source.get("problem", "").lower().split())
    tgt_problem = set(target.get("problem", "").lower().split())
    common_problem = src_problem & tgt_problem - {"and", "or", "the", "with", "based", "for", "a", "an", "of", "in", "on", "to", "design", "optimization"}
    if len(common_problem) >= 3:
        types.append("same_problem")
        shared["problem_terms"] = sorted(common_problem)
        scores["same_problem"] = RELATIONSHIP_WEIGHTS["same_problem"]

    if not types:
        types.append("semantic_similarity")
        scores["semantic_similarity"] = 0.5

    # Determine primary relationship (highest weight)
    primary = max(scores, key=scores.get) if scores else "semantic_similarity"

    return {"primary": primary, "types": types, "shared": shared}


def register_paper_graph_tools(mcp: Any) -> None:
    """Register paper relationship graph tools on a FastMCP server instance."""

    mcp.tool(
        name="find_related_papers",
        description=(
            "Find papers related to a given paper using semantic similarity and "
            "metadata overlap. Provide the paper's relative path or filename, e.g. "
            "'DSE/DAC23_GRL_DSE.pdf'. Returns ranked related papers annotated with "
            "relationship types such as same_benchmark, similar_method, same_problem, "
            "or semantic_similarity. Use this when the user asks 'what papers are "
            "similar to X' or 'find related work to Y'."
        ),
    )(find_related_papers)
