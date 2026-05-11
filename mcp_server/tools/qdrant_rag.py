"""Qdrant-backed RAG retrieval tools for MCP."""

from __future__ import annotations

from typing import Any

from rag_pipeline.ingest import create_qdrant_client
from rag_pipeline.retriever import QdrantRetriever
from utils.config import get_settings
from utils.logger import get_logger


logger = get_logger(__name__)

DEFAULT_TOP_K = 5
MAX_TOP_K = 20
DEFAULT_MAX_CHARS_PER_RESULT = 1200
MAX_CHARS_PER_RESULT = 4000


def search_dse_papers(
    query: str,
    query_variants: list[str] | None = None,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = 30,
    max_chars_per_result: int = DEFAULT_MAX_CHARS_PER_RESULT,
    use_hybrid: bool = True,
    use_rerank: bool = True,
) -> dict[str, Any]:
    """Search local DSE/MOO paper chunks from the Qdrant-backed hybrid RAG corpus.

    Args:
        query: Natural-language search query.
        query_variants: Optional rewritten or expanded retrieval queries.
        top_k: Number of chunks to return. Capped at 20.
        candidate_k: First-stage retrieval candidates before reranking.
        max_chars_per_result: Maximum text length for each returned chunk.
        use_hybrid: Whether to combine dense retrieval with sparse BM25 retrieval.
        use_rerank: Whether to apply lightweight reranking after fusion.

    Returns:
        A JSON-serializable dictionary containing ranked chunks and metadata.
    """

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty.")

    safe_top_k = min(max(1, top_k), MAX_TOP_K)
    safe_candidate_k = min(max(safe_top_k, candidate_k), 100)
    safe_max_chars = min(max(200, max_chars_per_result), MAX_CHARS_PER_RESULT)

    settings = get_settings()
    retriever = QdrantRetriever(settings)
    results = retriever.search(
        normalized_query,
        top_k=safe_top_k,
        query_variants=query_variants,
        candidate_k=safe_candidate_k,
        use_hybrid=use_hybrid,
        use_rerank=use_rerank,
    )

    formatted_results: list[dict[str, Any]] = []
    for index, result in enumerate(results, start=1):
        text = " ".join(result.text.split())
        formatted_results.append(
            {
                "rank": index,
                "score": result.score,
                "text": text[:safe_max_chars],
                "metadata": result.metadata,
            }
        )

    logger.info("RAG search returned %s results for query: %s", len(formatted_results), normalized_query)
    return {
        "query": normalized_query,
        "query_variants": query_variants or [],
        "collection": settings.qdrant.collection,
        "top_k": safe_top_k,
        "candidate_k": safe_candidate_k,
        "retrieval_mode": "hybrid" if use_hybrid else "dense",
        "rerank": use_rerank,
        "result_count": len(formatted_results),
        "results": formatted_results,
    }


def get_rag_corpus_stats() -> dict[str, Any]:
    """Return basic Qdrant corpus stats for the DSE paper knowledge base."""

    settings = get_settings()
    client = create_qdrant_client(settings)
    info = client.get_collection(settings.qdrant.collection)

    return {
        "collection": settings.qdrant.collection,
        "vector_size": settings.qdrant.vector_size,
        "distance": settings.qdrant.distance,
        "points_count": info.points_count,
        "indexed_vectors_count": info.indexed_vectors_count,
        "status": str(info.status),
    }


def register_qdrant_rag_tools(mcp: Any) -> None:
    """Register Qdrant RAG tools on a FastMCP server instance."""

    mcp.tool(
        name="search_dse_papers",
        description=(
            "Search the local DSE/MOO paper knowledge base stored in Qdrant. "
            "Use this when answering questions about design space exploration, "
            "microarchitecture optimization, MOO methods, or papers in Paper_ljh."
        ),
    )(search_dse_papers)

    mcp.tool(
        name="get_rag_corpus_stats",
        description="Return basic Qdrant collection stats for the local paper RAG corpus.",
    )(get_rag_corpus_stats)
