"""Single-paper interactive Q&A tool — ask questions about a specific paper."""

from __future__ import annotations

from typing import Any

from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from rag_pipeline.ingest import create_qdrant_client
from utils.config import get_settings
from utils.deepseek_llm import invoke_deepseek
from utils.embeddings import EmbeddingClient
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TOP_K = 8
MAX_TOP_K = 20
DEFAULT_MAX_CHARS_PER_RESULT = 1500

PAPER_QA_SYSTEM_PROMPT = """\
You are a research assistant answering questions about a specific DSE/MOO paper. \
Answer using ONLY the provided paper excerpts. Cite the page number for every claim.

Format citations as [page X] at the end of each sentence that uses that source.

If the provided excerpts do not contain enough information to answer the question, \
say so clearly and suggest what part of the paper might contain the answer.

Keep answers concise, technical, and grounded in the text. Do not invent information \
beyond what is provided.\
"""


def ask_paper(
    paper_path: str,
    question: str,
    top_k: int = DEFAULT_TOP_K,
    max_chars_per_result: int = DEFAULT_MAX_CHARS_PER_RESULT,
) -> dict[str, Any]:
    """Ask a question about a specific paper and get a page-cited answer.

    Args:
        paper_path: Relative path or filename of the paper, e.g. ``DSE/DAC23_GRL_DSE.pdf``.
        question: The question to ask about the paper.
        top_k: Number of relevant chunks to retrieve, capped at 20.
        max_chars_per_result: Maximum characters per retrieved chunk.

    Returns:
        A dictionary with the answer, sources (page, text, score), and paper info.
    """
    q = question.strip()
    if not q:
        raise ValueError("question must not be empty.")

    safe_top_k = min(max(1, top_k), MAX_TOP_K)
    safe_max_chars = min(max(200, max_chars_per_result), 4000)
    settings = get_settings()
    client = create_qdrant_client(settings)
    embedder = EmbeddingClient(settings)

    # Resolve the paper
    paper_info = _resolve_paper(client, settings, paper_path)
    if paper_info is None:
        return {"success": False, "error": f"Paper not found in corpus: {paper_path}"}

    source_path = paper_info["relative_path"]

    # Search within this paper only
    question_vector = embedder.embed_query(q)
    paper_filter = Filter(
        must=[
            FieldCondition(key="project", match=MatchValue(value="dse-explorer")),
            FieldCondition(key="relative_path", match=MatchValue(value=source_path)),
        ]
    )

    search_result = client.query_points(
        collection_name=settings.qdrant.collection,
        query=question_vector,
        limit=safe_top_k,
        query_filter=paper_filter,
        with_payload=True,
    )

    if not search_result.points:
        return {
            "success": True,
            "paper_path": source_path,
            "file_name": paper_info["file_name"],
            "question": q,
            "answer": (
                f"No relevant excerpts found in {paper_info['file_name']} for this question. "
                "The paper may not cover this topic, or try rephrasing the question."
            ),
            "sources": [],
        }

    # Collect results
    excerpts: list[dict[str, Any]] = []
    context_parts: list[str] = []
    for rank, point in enumerate(search_result.points, start=1):
        payload = point.payload or {}
        text = " ".join(str(payload.get("text", "")).split())
        page = payload.get("page", "?")
        score = float(point.score)
        snippets = text[:safe_max_chars]
        excerpts.append({"rank": rank, "page": page, "text": snippets, "score": round(score, 4)})
        context_parts.append(f"[page {page}] (score={score:.4f}):\n{snippets}")

    context = "\n\n---\n\n".join(context_parts)

    try:
        answer = invoke_deepseek(
            f"Paper: {paper_info['file_name']}\n\nQuestion: {q}\n\nPaper excerpts:\n\n{context}",
            system_prompt=PAPER_QA_SYSTEM_PROMPT,
            settings=settings,
        )
    except Exception:
        logger.exception("LLM call failed for ask_paper: %s", paper_path)
        return {"success": False, "error": "LLM call failed during Q&A."}

    return {
        "success": True,
        "paper_path": source_path,
        "file_name": paper_info["file_name"],
        "question": q,
        "answer": answer.strip(),
        "sources": excerpts,
    }


def _resolve_paper(client: Any, settings: Any, paper_path: str) -> dict[str, Any] | None:
    pf = Filter(
        must=[
            FieldCondition(key="project", match=MatchValue(value="dse-explorer")),
            FieldCondition(key="relative_path", match=MatchValue(value=paper_path)),
        ]
    )
    points, _ = client.scroll(
        collection_name=settings.qdrant.collection,
        scroll_filter=pf, limit=1, with_payload=True,
    )
    if points:
        payload = points[0].payload or {}
        return {"relative_path": str(payload.get("relative_path", "")), "file_name": str(payload.get("file_name", ""))}

    name = paper_path.replace("\\", "/").rsplit("/", 1)[-1]
    pf2 = Filter(
        must=[
            FieldCondition(key="project", match=MatchValue(value="dse-explorer")),
            FieldCondition(key="file_name", match=MatchValue(value=name)),
        ]
    )
    points, _ = client.scroll(
        collection_name=settings.qdrant.collection,
        scroll_filter=pf2, limit=1, with_payload=True,
    )
    if points:
        payload = points[0].payload or {}
        return {"relative_path": str(payload.get("relative_path", "")), "file_name": str(payload.get("file_name", ""))}

    return None


def register_paper_qa_tools(mcp: Any) -> None:
    """Register paper Q&A tools on a FastMCP server instance."""

    mcp.tool(
        name="ask_paper",
        description=(
            "Ask a question about a specific paper from the DSE/MOO corpus and get "
            "an answer with page-level citations. Provide the paper's path or filename "
            "and your question. Use this for deep-dive questions about a single paper, "
            "e.g. 'what is the core idea of this paper' or 'how does their method differ "
            "from prior work'."
        ),
    )(ask_paper)
