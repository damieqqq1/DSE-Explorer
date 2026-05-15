"""Paper summarization tool for generating structured TL;DR summaries."""

from __future__ import annotations

import json
import re
from typing import Any

from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from rag_pipeline.ingest import create_qdrant_client
from utils.config import get_settings
from utils.deepseek_llm import invoke_deepseek
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_CHARS = 30000

SUMMARIZE_SYSTEM_PROMPT = """\
You are a research paper summarizer for the DSE (Design Space Exploration) and \
MOO (Multi-Objective Optimization) domain. Summarize the provided paper text in \
a structured JSON format.

Output only a JSON object with these fields:
{
  "title": "paper title (inferred from text)",
  "authors": ["author 1", "author 2"],
  "year": "publication year",
  "problem": "1-2 sentence description of the research problem",
  "method": "core method / approach described in 3-5 sentences",
  "contributions": ["key contribution 1", "key contribution 2"],
  "results": "key experimental results and benchmarks",
  "limitations": "noted limitations or future work",
  "dse_takeaway": "one-sentence takeaway for a DSE/MOO researcher"
}

Rules:
- Use empty string "" or empty list [] for unknown fields.
- Keep the total output under 500 words.
- Only use information present in the provided text — do not fabricate.
- Preserve exact method names and acronyms (e.g. "GRL-DSE", "NSGA-II").\
"""


def summarize_paper(
    paper_path: str,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> dict[str, Any]:
    """Generate a structured TL;DR summary of a specific paper from the local corpus.

    Args:
        paper_path: Relative path or filename of the paper, e.g. ``DSE/DAC23_GRL_DSE.pdf``.
        max_chars: Maximum characters of paper text sent to the summarizer (default 30000).

    Returns:
        A dictionary with the structured summary under ``summary``, plus metadata
        about the retrieval (paper_path, total_chunks, chars_used).
    """
    settings = get_settings()
    client = create_qdrant_client(settings)

    chunks = _fetch_chunks_by_path(client, settings, paper_path)
    if not chunks:
        chunks = _fetch_chunks_by_filename(client, settings, paper_path)

    if not chunks:
        return {
            "success": False,
            "error": f"No chunks found for paper: {paper_path}",
            "paper_path": paper_path,
        }

    chunks.sort(key=lambda c: (c["page"], c["chunk_index"]))

    full_text, total_chars = _assemble_text(chunks, max_chars)

    try:
        raw = invoke_deepseek(
            f"Summarize the following research paper text:\n\n{full_text}",
            system_prompt=SUMMARIZE_SYSTEM_PROMPT,
            settings=settings,
        )
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        summary = json.loads(match.group()) if match else {"raw_summary": raw}
    except Exception:
        logger.exception("LLM summarization failed for %s", paper_path)
        return {
            "success": False,
            "error": "LLM call failed during summarization.",
            "paper_path": paper_path,
        }

    return {
        "success": True,
        "paper_path": paper_path,
        "total_chunks": len(chunks),
        "chars_used": total_chars,
        "summary": summary,
    }


def _fetch_chunks_by_path(client: Any, settings: Any, paper_path: str) -> list[dict[str, Any]]:
    pf = Filter(
        must=[
            FieldCondition(key="project", match=MatchValue(value="dse-explorer")),
            FieldCondition(key="relative_path", match=MatchValue(value=paper_path)),
        ]
    )
    return _scroll_chunks(client, settings.qdrant.collection, pf)


def _fetch_chunks_by_filename(client: Any, settings: Any, paper_path: str) -> list[dict[str, Any]]:
    name = paper_path.replace("\\", "/").rsplit("/", 1)[-1]
    pf = Filter(
        must=[
            FieldCondition(key="project", match=MatchValue(value="dse-explorer")),
            FieldCondition(key="file_name", match=MatchValue(value=name)),
        ]
    )
    return _scroll_chunks(client, settings.qdrant.collection, pf)


def _scroll_chunks(client: Any, collection: str, query_filter: Filter) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=query_filter,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            text = str(payload.get("text", ""))
            if text.strip():
                chunks.append({
                    "page": payload.get("page", 0),
                    "chunk_index": payload.get("chunk_index", 0),
                    "text": text,
                })
        if offset is None:
            break
    return chunks


def _assemble_text(chunks: list[dict[str, Any]], max_chars: int) -> tuple[str, int]:
    parts: list[str] = []
    total = 0
    for chunk in chunks:
        remaining = max_chars - total
        if remaining <= 0:
            break
        text = chunk["text"]
        if total + len(text) > max_chars:
            if remaining > 200:
                parts.append(text[:remaining] + "\n[...truncated]")
            break
        parts.append(text)
        total += len(text)
    return "\n\n".join(parts), total


def register_paper_summary_tools(mcp: Any) -> None:
    """Register paper summary tools on a FastMCP server instance."""

    mcp.tool(
        name="summarize_paper",
        description=(
            "Generate a structured summary of a specific paper from the DSE/MOO "
            "corpus. Provide the paper's relative path or filename, e.g. "
            "'DSE/DAC23_GRL_DSE.pdf'. Returns the title, authors, year, problem, "
            "method, contributions, results, limitations, and a one-sentence DSE "
            "takeaway. Use this when the user asks 'what is paper X about' or "
            "'summarize paper Y'."
        ),
    )(summarize_paper)
