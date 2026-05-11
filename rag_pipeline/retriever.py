"""Hybrid retrieval logic for the RAG pipeline."""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from rag_pipeline.ingest import create_qdrant_client
from utils.config import AppSettings, get_settings
from utils.embeddings import EmbeddingClient
from utils.logger import configure_logging, get_logger


logger = get_logger(__name__)

RRF_K = 60
DEFAULT_CANDIDATE_K = 30
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "which",
    "with",
}
_SPARSE_INDEX_CACHE: dict[str, "SparseKeywordIndex"] = {}


@dataclass(frozen=True)
class RetrievalResult:
    score: float
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CorpusDocument:
    doc_id: str
    text: str
    metadata: dict[str, Any]


@dataclass
class RetrievalCandidate:
    doc_id: str
    text: str
    metadata: dict[str, Any]
    dense_score: float = 0.0
    sparse_score: float = 0.0
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fused_score: float = 0.0
    rerank_score: float = 0.0


class SparseKeywordIndex:
    """In-memory BM25 index over Qdrant payload text."""

    def __init__(self, documents: list[CorpusDocument]) -> None:
        self.documents = documents
        self.doc_lengths: list[int] = []
        self.inverted_index: dict[str, list[tuple[int, int]]] = defaultdict(list)
        total_length = 0

        for doc_index, document in enumerate(documents):
            counts = Counter(tokenize(document.text))
            length = sum(counts.values())
            self.doc_lengths.append(length)
            total_length += length
            for token, frequency in counts.items():
                self.inverted_index[token].append((doc_index, frequency))

        self.avg_doc_length = total_length / max(len(documents), 1)

    def search(self, query: str, limit: int) -> list[RetrievalCandidate]:
        query_tokens = [token for token in tokenize(query) if token not in STOPWORDS]
        if not query_tokens:
            return []

        scores: dict[int, float] = defaultdict(float)
        unique_query_tokens = set(query_tokens)
        total_docs = len(self.documents)
        k1 = 1.5
        b = 0.75

        for token in unique_query_tokens:
            postings = self.inverted_index.get(token, [])
            if not postings:
                continue
            doc_frequency = len(postings)
            idf = math.log(1 + (total_docs - doc_frequency + 0.5) / (doc_frequency + 0.5))
            for doc_index, term_frequency in postings:
                doc_length = self.doc_lengths[doc_index] or 1
                denominator = term_frequency + k1 * (
                    1 - b + b * doc_length / max(self.avg_doc_length, 1e-9)
                )
                scores[doc_index] += idf * (term_frequency * (k1 + 1)) / denominator

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        candidates: list[RetrievalCandidate] = []
        for rank, (doc_index, score) in enumerate(ranked, start=1):
            document = self.documents[doc_index]
            candidates.append(
                RetrievalCandidate(
                    doc_id=document.doc_id,
                    text=document.text,
                    metadata=document.metadata,
                    sparse_score=score,
                    sparse_rank=rank,
                )
            )
        return candidates


class QdrantRetriever:
    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.embedder = EmbeddingClient(self.settings)
        self.client = create_qdrant_client(self.settings)

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        query_variants: list[str] | None = None,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        use_hybrid: bool = True,
        use_rerank: bool = True,
    ) -> list[RetrievalResult]:
        safe_candidate_k = max(candidate_k, top_k)
        queries = clean_queries([query, *(query_variants or [])])
        dense_candidates = self._dense_search(queries, safe_candidate_k)
        if not use_hybrid:
            candidates = dense_candidates
        else:
            sparse_query = " ".join(queries)
            sparse_candidates = self._sparse_search(sparse_query, safe_candidate_k)
            candidates = fuse_candidates(dense_candidates, sparse_candidates)

        if use_rerank:
            candidates = rerank_candidates(query, queries, candidates)
        else:
            candidates.sort(key=lambda item: item.fused_score or item.dense_score, reverse=True)

        return [candidate_to_result(candidate) for candidate in candidates[:top_k]]

    def _dense_search(self, queries: list[str], candidate_k: int) -> list[RetrievalCandidate]:
        vectors = self.embedder.embed_documents(queries, batch_size=min(len(queries), 10))
        candidates: dict[str, RetrievalCandidate] = {}

        for vector in vectors:
            result = self.client.query_points(
                collection_name=self.settings.qdrant.collection,
                query=vector,
                limit=candidate_k,
                query_filter=project_filter(),
                with_payload=True,
            )
            for rank, point in enumerate(result.points, start=1):
                payload = point.payload or {}
                doc_id = str(point.id)
                candidate = candidates.get(doc_id)
                metadata = {key: value for key, value in payload.items() if key != "text"}
                metadata["_point_id"] = doc_id
                if candidate is None:
                    candidates[doc_id] = RetrievalCandidate(
                        doc_id=doc_id,
                        text=str(payload.get("text", "")),
                        metadata=metadata,
                        dense_score=float(point.score),
                        dense_rank=rank,
                    )
                    continue
                if float(point.score) > candidate.dense_score:
                    candidate.dense_score = float(point.score)
                    candidate.dense_rank = rank

        ranked = sorted(candidates.values(), key=lambda item: item.dense_score, reverse=True)
        for rank, candidate in enumerate(ranked, start=1):
            candidate.dense_rank = rank
        return ranked[:candidate_k]

    def _sparse_search(self, query: str, candidate_k: int) -> list[RetrievalCandidate]:
        index = get_sparse_index(self.client, self.settings)
        return index.search(query, candidate_k)


def get_sparse_index(client: Any, settings: AppSettings) -> SparseKeywordIndex:
    cache_key = f"{settings.qdrant.url}|{settings.qdrant.collection}"
    cached = _SPARSE_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    documents = load_sparse_documents(client, settings)
    index = SparseKeywordIndex(documents)
    _SPARSE_INDEX_CACHE[cache_key] = index
    logger.info("Built sparse keyword index with %s documents", len(documents))
    return index


def load_sparse_documents(client: Any, settings: AppSettings) -> list[CorpusDocument]:
    documents: list[CorpusDocument] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=settings.qdrant.collection,
            scroll_filter=project_filter(),
            limit=1024,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            text = str(payload.get("text", ""))
            if not text:
                continue
            metadata = {key: value for key, value in payload.items() if key != "text"}
            metadata["_point_id"] = str(point.id)
            documents.append(CorpusDocument(doc_id=str(point.id), text=text, metadata=metadata))
        if offset is None:
            break
    return documents


def project_filter() -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key="project",
                match=MatchValue(value="dse-explorer"),
            )
        ]
    )


def fuse_candidates(
    dense_candidates: list[RetrievalCandidate],
    sparse_candidates: list[RetrievalCandidate],
) -> list[RetrievalCandidate]:
    candidates: dict[str, RetrievalCandidate] = {}

    for candidate in dense_candidates:
        candidates[candidate.doc_id] = candidate
    for candidate in sparse_candidates:
        existing = candidates.get(candidate.doc_id)
        if existing is None:
            candidates[candidate.doc_id] = candidate
            continue
        existing.sparse_score = candidate.sparse_score
        existing.sparse_rank = candidate.sparse_rank

    for candidate in candidates.values():
        score = 0.0
        if candidate.dense_rank is not None:
            score += 1 / (RRF_K + candidate.dense_rank)
        if candidate.sparse_rank is not None:
            score += 1 / (RRF_K + candidate.sparse_rank)
        candidate.fused_score = score

    ranked = sorted(candidates.values(), key=lambda item: item.fused_score, reverse=True)
    return ranked


def rerank_candidates(
    original_query: str,
    query_variants: list[str],
    candidates: list[RetrievalCandidate],
) -> list[RetrievalCandidate]:
    if not candidates:
        return []

    query_text = " ".join([original_query, *query_variants])
    query_tokens = {token for token in tokenize(query_text) if token not in STOPWORDS}
    exact_terms = extract_exact_terms(query_text)
    max_fused = max((candidate.fused_score for candidate in candidates), default=0.0) or 1.0

    for candidate in candidates:
        text = candidate.text
        text_lower = text.lower()
        text_tokens = set(tokenize(text))
        overlap = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
        exact_hits = sum(1 for term in exact_terms if term.lower() in text_lower)
        phrase_hits = sum(1 for variant in query_variants if len(variant) > 12 and variant.lower() in text_lower)
        source_text = " ".join(
            str(candidate.metadata.get(key, ""))
            for key in ("relative_path", "file_name", "category")
        ).lower()
        source_hits = sum(1 for term in exact_terms if term.lower() in source_text)
        normalized_fused = candidate.fused_score / max_fused
        candidate.rerank_score = (
            0.50 * normalized_fused
            + 0.30 * overlap
            + 0.12 * min(exact_hits, 3) / 3
            + 0.05 * min(source_hits, 2) / 2
            + 0.03 * min(phrase_hits, 2) / 2
        )

    candidates.sort(key=lambda item: item.rerank_score, reverse=True)
    return candidates


def candidate_to_result(candidate: RetrievalCandidate) -> RetrievalResult:
    metadata = dict(candidate.metadata)
    metadata["retrieval"] = {
        "dense_score": candidate.dense_score,
        "sparse_score": candidate.sparse_score,
        "fused_score": candidate.fused_score,
        "rerank_score": candidate.rerank_score,
        "dense_rank": candidate.dense_rank,
        "sparse_rank": candidate.sparse_rank,
    }
    return RetrievalResult(
        score=candidate.rerank_score or candidate.fused_score or candidate.dense_score,
        text=candidate.text,
        metadata=metadata,
    )


def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_\-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]{2,}", text)
        if len(token.strip()) >= 2
    ]


def extract_exact_terms(text: str) -> list[str]:
    terms = re.findall(r"\b[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+\b|\b[A-Z]{2,}[A-Za-z0-9-]*\b", text)
    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(term)
    return unique


def clean_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for query in queries:
        value = " ".join(str(query).split())
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(value)
    return cleaned or [""]


def format_result(result: RetrievalResult, index: int) -> str:
    source = result.metadata.get("relative_path") or result.metadata.get("file_name") or "<unknown>"
    page = result.metadata.get("page", "?")
    snippet = " ".join(result.text.split())[:600]
    retrieval = result.metadata.get("retrieval", {})
    return (
        f"[{index}] score={result.score:.4f} source={source} page={page} "
        f"dense={retrieval.get('dense_score', 0):.4f} "
        f"sparse={retrieval.get('sparse_score', 0):.4f}\n{snippet}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search DSE paper chunks from Qdrant.")
    parser.add_argument("query", type=str, help="Search query.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results.")
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K, help="First-stage candidates.")
    parser.add_argument("--dense-only", action="store_true", help="Disable BM25 sparse retrieval.")
    parser.add_argument("--no-rerank", action="store_true", help="Disable lightweight reranking.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    retriever = QdrantRetriever(settings)
    results = retriever.search(
        args.query,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        use_hybrid=not args.dense_only,
        use_rerank=not args.no_rerank,
    )
    logger.info("Found %s results", len(results))
    for index, result in enumerate(results, start=1):
        print(format_result(result, index))
        print()


if __name__ == "__main__":
    main()
