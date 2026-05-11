"""PDF ingestion script: parse, chunk, embed, and store papers in Qdrant."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PayloadSchemaType, PointStruct, VectorParams

from utils.config import AppSettings, get_settings
from utils.embeddings import EmbeddingClient
from utils.logger import configure_logging, get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    text: str
    metadata: dict[str, str | int]


PARSED_CACHE_DIR = Path("rag_pipeline/data/parsed_cache")


def find_pdfs(paper_root: Path, limit: int | None = None) -> list[Path]:
    pdfs = sorted(path for path in paper_root.rglob("*.pdf") if path.is_file())
    if limit is not None:
        return pdfs[:limit]
    return pdfs


def extract_pdf_chunks(
    pdf_path: Path,
    *,
    paper_root: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    reader = PdfReader(str(pdf_path))
    chunks: list[DocumentChunk] = []
    relative_path = pdf_path.relative_to(paper_root).as_posix()
    category = pdf_path.parent.relative_to(paper_root).as_posix()

    for page_index, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if not text:
            continue
        for chunk_index, chunk_text in enumerate(split_text(text, chunk_size, chunk_overlap)):
            chunk_id = stable_chunk_id(relative_path, page_index, chunk_index, chunk_text)
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    text=chunk_text,
                    metadata={
                        "source": str(pdf_path),
                        "relative_path": relative_path,
                        "file_name": pdf_path.name,
                        "category": category,
                        "page": page_index,
                        "chunk_index": chunk_index,
                        "project": "dse-explorer",
                    },
                )
            )
    return chunks


def load_or_extract_pdf_chunks(
    pdf_path: Path,
    *,
    settings: AppSettings,
    chunk_size: int,
    chunk_overlap: int,
    use_cache: bool,
) -> list[DocumentChunk]:
    cache_path = get_cache_path(settings, pdf_path, chunk_size, chunk_overlap)
    if use_cache and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if is_cache_valid(cached, pdf_path, chunk_size, chunk_overlap):
                return [
                    DocumentChunk(
                        id=item["id"],
                        text=item["text"],
                        metadata=item["metadata"],
                    )
                    for item in cached["chunks"]
                ]
        except (OSError, json.JSONDecodeError):
            logger.warning("Ignoring invalid parse cache: %s", cache_path)
            cache_path.unlink(missing_ok=True)

    chunks = extract_pdf_chunks(
        pdf_path,
        paper_root=settings.paper_root,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": str(pdf_path),
            "relative_path": pdf_path.relative_to(settings.paper_root).as_posix(),
            "file_size": pdf_path.stat().st_size,
            "mtime_ns": pdf_path.stat().st_mtime_ns,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunks": [
                {"id": chunk.id, "text": chunk.text, "metadata": chunk.metadata}
                for chunk in chunks
            ],
        }
        cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return chunks


def get_cache_path(
    settings: AppSettings,
    pdf_path: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> Path:
    relative_path = pdf_path.relative_to(settings.paper_root).as_posix()
    raw = f"{relative_path}|{chunk_size}|{chunk_overlap}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return settings.project_root / PARSED_CACHE_DIR / f"{digest}.json"


def is_cache_valid(
    cached: dict[str, Any],
    pdf_path: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> bool:
    stat = pdf_path.stat()
    return (
        cached.get("file_size") == stat.st_size
        and cached.get("mtime_ns") == stat.st_mtime_ns
        and cached.get("chunk_size") == chunk_size
        and cached.get("chunk_overlap") == chunk_overlap
    )


def normalize_text(text: str) -> str:
    cleaned = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    return " ".join(cleaned.replace("\x00", " ").split())


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size.")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - chunk_overlap
    return chunks


def stable_chunk_id(relative_path: str, page: int, chunk_index: int, text: str) -> str:
    raw = f"{relative_path}|{page}|{chunk_index}|{text[:128]}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def create_qdrant_client(settings: AppSettings) -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant.url,
        api_key=settings.qdrant.api_key or None,
        timeout=settings.qdrant.timeout,
    )


def ensure_collection(client: QdrantClient, settings: AppSettings) -> None:
    collection = settings.qdrant.collection
    if not client.collection_exists(collection):
        distance_name = settings.qdrant.distance.upper()
        distance = getattr(Distance, distance_name, Distance.COSINE)
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=settings.qdrant.vector_size, distance=distance),
        )

    for field_name in ("project", "relative_path", "category"):
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:
            message = str(exc)
            if "already exists" not in message.lower():
                raise


def upsert_chunks(
    *,
    client: QdrantClient,
    embedder: EmbeddingClient,
    settings: AppSettings,
    chunks: list[DocumentChunk],
    batch_size: int,
    start_offset: int,
) -> None:
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embedder.embed_documents([chunk.text for chunk in batch], batch_size=batch_size)
        points = [
            PointStruct(
                id=chunk.id,
                vector=vector,
                payload={**chunk.metadata, "text": chunk.text},
            )
            for chunk, vector in zip(batch, vectors, strict=True)
        ]
        client.upsert(collection_name=settings.qdrant.collection, points=points)
        logger.info(
            "Upserted chunks %s-%s",
            start_offset + start + 1,
            start_offset + start + len(batch),
        )


def ingest_pdfs(
    *,
    settings: AppSettings,
    limit: int | None,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    dry_run: bool = False,
    use_cache: bool = True,
) -> int:
    settings.validate_for_papers()
    pdfs = find_pdfs(settings.paper_root, limit=limit)
    if not pdfs:
        logger.warning("No PDF files found under %s", settings.paper_root)
        return 0

    embedder = None
    client = None
    if not dry_run:
        embedder = EmbeddingClient(settings)
        client = create_qdrant_client(settings)
        ensure_collection(client, settings)

    total_chunks = 0
    for pdf_path in pdfs:
        try:
            pdf_chunks = load_or_extract_pdf_chunks(
                pdf_path,
                settings=settings,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                use_cache=use_cache,
            )
        except Exception:
            logger.exception("Failed to parse PDF: %s", pdf_path)
            continue
        logger.info("Parsed %s chunks from %s", len(pdf_chunks), pdf_path.name)
        if not dry_run and pdf_chunks:
            assert embedder is not None
            assert client is not None
            upsert_chunks(
                client=client,
                embedder=embedder,
                settings=settings,
                chunks=pdf_chunks,
                batch_size=batch_size,
                start_offset=total_chunks,
            )
        total_chunks += len(pdf_chunks)

    logger.info("Prepared %s chunks from %s PDF files", total_chunks, len(pdfs))
    return total_chunks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest PDF papers into Qdrant.")
    parser.add_argument("--limit", type=int, default=3, help="Number of PDFs to ingest.")
    parser.add_argument("--all", action="store_true", help="Ingest all PDFs under PAPER_ROOT.")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Chunk size in characters.")
    parser.add_argument("--chunk-overlap", type=int, default=180, help="Chunk overlap in characters.")
    parser.add_argument("--batch-size", type=int, default=8, help="Embedding/upsert batch size.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and chunk without embedding/upsert.")
    parser.add_argument("--no-cache", action="store_true", help="Disable parsed PDF cache.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    limit = None if args.all else args.limit
    count = ingest_pdfs(
        settings=settings,
        limit=limit,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        use_cache=not args.no_cache,
    )
    logger.info("Ingestion finished with %s chunks", count)


if __name__ == "__main__":
    main()
