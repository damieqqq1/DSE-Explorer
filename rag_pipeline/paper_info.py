"""Extract structured metadata and method information from local papers."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from rag_pipeline.ingest import find_pdfs, normalize_text
from utils.config import AppSettings, PROJECT_ROOT, get_settings
from utils.deepseek_llm import invoke_deepseek
from utils.logger import configure_logging, get_logger


logger = get_logger(__name__)

PAPER_INFO_DB_PATH = PROJECT_ROOT / "rag_pipeline" / "data" / "paper_info.sqlite3"
DEFAULT_MAX_PAGES = 4
DEFAULT_MAX_CHARS = 16000

PAPER_INFO_SYSTEM_PROMPT = """You extract structured information from DSE/MOO research papers.
Return only valid JSON. Do not include markdown fences or commentary.

JSON schema:
{
  "metadata": {
    "title": "",
    "authors": [],
    "year": "",
    "venue": "",
    "abstract": "",
    "keywords": []
  },
  "method": {
    "method_name": "",
    "problem": "",
    "core_idea": "",
    "optimization_algorithm": "",
    "model_or_agent": "",
    "design_variables": [],
    "objectives": [],
    "benchmarks": [],
    "baselines": [],
    "metrics": [],
    "main_results": [],
    "limitations": []
  },
  "tags": []
}

Rules:
- Use concise English technical terms when possible.
- If a field is unknown from the provided text, use an empty string or empty list.
- Preserve exact method names and acronyms such as GRL-DSE, BOOM-Explorer, ArchGym.
- Focus on design space exploration, multi-objective optimization, architecture search, and combinatorial optimization details."""


@dataclass(frozen=True)
class PaperInfo:
    relative_path: str
    file_name: str
    category: str
    metadata: dict[str, Any]
    method: dict[str, Any]
    tags: list[str]
    extraction_model: str


class PaperInfoStore:
    def __init__(self, db_path: Path = PAPER_INFO_DB_PATH) -> None:
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
                CREATE TABLE IF NOT EXISTS paper_info (
                    relative_path TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    authors_json TEXT NOT NULL,
                    year TEXT NOT NULL,
                    venue TEXT NOT NULL,
                    abstract TEXT NOT NULL,
                    keywords_json TEXT NOT NULL,
                    method_name TEXT NOT NULL,
                    problem TEXT NOT NULL,
                    core_idea TEXT NOT NULL,
                    optimization_algorithm TEXT NOT NULL,
                    model_or_agent TEXT NOT NULL,
                    design_variables_json TEXT NOT NULL,
                    objectives_json TEXT NOT NULL,
                    benchmarks_json TEXT NOT NULL,
                    baselines_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    main_results_json TEXT NOT NULL,
                    limitations_json TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    extraction_model TEXT NOT NULL,
                    raw_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_info_method ON paper_info(method_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_info_year ON paper_info(year)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_info_category ON paper_info(category)")

    def has_paper(self, relative_path: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM paper_info WHERE relative_path = ? LIMIT 1",
                (relative_path,),
            ).fetchone()
        return row is not None

    def upsert(self, info: PaperInfo) -> None:
        metadata = info.metadata
        method = info.method
        raw = {
            "metadata": metadata,
            "method": method,
            "tags": info.tags,
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO paper_info(
                    relative_path, file_name, category, title, authors_json, year, venue,
                    abstract, keywords_json, method_name, problem, core_idea,
                    optimization_algorithm, model_or_agent, design_variables_json,
                    objectives_json, benchmarks_json, baselines_json, metrics_json,
                    main_results_json, limitations_json, tags_json, extraction_model, raw_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(relative_path) DO UPDATE SET
                    file_name = excluded.file_name,
                    category = excluded.category,
                    title = excluded.title,
                    authors_json = excluded.authors_json,
                    year = excluded.year,
                    venue = excluded.venue,
                    abstract = excluded.abstract,
                    keywords_json = excluded.keywords_json,
                    method_name = excluded.method_name,
                    problem = excluded.problem,
                    core_idea = excluded.core_idea,
                    optimization_algorithm = excluded.optimization_algorithm,
                    model_or_agent = excluded.model_or_agent,
                    design_variables_json = excluded.design_variables_json,
                    objectives_json = excluded.objectives_json,
                    benchmarks_json = excluded.benchmarks_json,
                    baselines_json = excluded.baselines_json,
                    metrics_json = excluded.metrics_json,
                    main_results_json = excluded.main_results_json,
                    limitations_json = excluded.limitations_json,
                    tags_json = excluded.tags_json,
                    extraction_model = excluded.extraction_model,
                    raw_json = excluded.raw_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    info.relative_path,
                    info.file_name,
                    info.category,
                    str(metadata.get("title", "")),
                    json.dumps(as_list(metadata.get("authors")), ensure_ascii=False),
                    str(metadata.get("year", "")),
                    str(metadata.get("venue", "")),
                    str(metadata.get("abstract", "")),
                    json.dumps(as_list(metadata.get("keywords")), ensure_ascii=False),
                    str(method.get("method_name", "")),
                    str(method.get("problem", "")),
                    str(method.get("core_idea", "")),
                    str(method.get("optimization_algorithm", "")),
                    str(method.get("model_or_agent", "")),
                    json.dumps(as_list(method.get("design_variables")), ensure_ascii=False),
                    json.dumps(as_list(method.get("objectives")), ensure_ascii=False),
                    json.dumps(as_list(method.get("benchmarks")), ensure_ascii=False),
                    json.dumps(as_list(method.get("baselines")), ensure_ascii=False),
                    json.dumps(as_list(method.get("metrics")), ensure_ascii=False),
                    json.dumps(as_list(method.get("main_results")), ensure_ascii=False),
                    json.dumps(as_list(method.get("limitations")), ensure_ascii=False),
                    json.dumps(info.tags, ensure_ascii=False),
                    info.extraction_model,
                    json.dumps(raw, ensure_ascii=False),
                ),
            )

    def list_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT relative_path, title, method_name, year, venue, optimization_algorithm, updated_at
                FROM paper_info
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]


def extract_paper_info(
    pdf_path: Path,
    *,
    settings: AppSettings,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> PaperInfo:
    context = extract_front_matter_text(pdf_path, max_pages=max_pages, max_chars=max_chars)
    payload = call_paper_info_extractor(context, pdf_path, settings)
    normalized = normalize_payload(payload)
    relative_path = pdf_path.relative_to(settings.paper_root).as_posix()
    category = pdf_path.parent.relative_to(settings.paper_root).as_posix()
    return PaperInfo(
        relative_path=relative_path,
        file_name=pdf_path.name,
        category=category,
        metadata=normalized["metadata"],
        method=normalized["method"],
        tags=as_list(normalized.get("tags")),
        extraction_model=settings.llm.model_id,
    )


def extract_front_matter_text(pdf_path: Path, *, max_pages: int, max_chars: int) -> str:
    reader = PdfReader(str(pdf_path))
    blocks: list[str] = []
    for page_index, page in enumerate(reader.pages[:max_pages], start=1):
        text = normalize_text(page.extract_text() or "")
        if text:
            blocks.append(f"[page {page_index}]\n{text}")
    return "\n\n".join(blocks)[:max_chars]


def call_paper_info_extractor(context: str, pdf_path: Path, settings: AppSettings) -> dict[str, Any]:
    prompt = f"""PDF file name:
{pdf_path.name}

Extract structured metadata and method information from the paper text below.

Paper text:
{context}
"""
    raw = invoke_deepseek(prompt, system_prompt=PAPER_INFO_SYSTEM_PROMPT, settings=settings)
    return json.loads(extract_json(raw))


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    method = payload.get("method") if isinstance(payload.get("method"), dict) else {}
    return {
        "metadata": {
            "title": str(metadata.get("title", "")).strip(),
            "authors": as_list(metadata.get("authors")),
            "year": str(metadata.get("year", "")).strip(),
            "venue": str(metadata.get("venue", "")).strip(),
            "abstract": str(metadata.get("abstract", "")).strip(),
            "keywords": as_list(metadata.get("keywords")),
        },
        "method": {
            "method_name": str(method.get("method_name", "")).strip(),
            "problem": str(method.get("problem", "")).strip(),
            "core_idea": str(method.get("core_idea", "")).strip(),
            "optimization_algorithm": str(method.get("optimization_algorithm", "")).strip(),
            "model_or_agent": str(method.get("model_or_agent", "")).strip(),
            "design_variables": as_list(method.get("design_variables")),
            "objectives": as_list(method.get("objectives")),
            "benchmarks": as_list(method.get("benchmarks")),
            "baselines": as_list(method.get("baselines")),
            "metrics": as_list(method.get("metrics")),
            "main_results": as_list(method.get("main_results")),
            "limitations": as_list(method.get("limitations")),
        },
        "tags": as_list(payload.get("tags")),
    }


def extract_json(raw: str) -> str:
    text = raw.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in paper info extractor output.")
    return match.group(0)


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def extract_papers(
    *,
    settings: AppSettings,
    limit: int | None,
    max_pages: int,
    max_chars: int,
    dry_run: bool = False,
    force: bool = False,
) -> list[PaperInfo]:
    settings.validate_for_papers()
    store = PaperInfoStore()
    pdfs = find_pdfs(settings.paper_root, limit=limit)
    results: list[PaperInfo] = []

    for pdf_path in pdfs:
        relative_path = pdf_path.relative_to(settings.paper_root).as_posix()
        if not force and not dry_run and store.has_paper(relative_path):
            logger.info("Skipping existing paper info: %s", relative_path)
            continue
        try:
            info = extract_paper_info(
                pdf_path,
                settings=settings,
                max_pages=max_pages,
                max_chars=max_chars,
            )
        except Exception:
            logger.exception("Failed to extract paper info: %s", relative_path)
            continue
        results.append(info)
        if dry_run:
            print(json.dumps(paper_info_to_dict(info), indent=2, ensure_ascii=False))
        else:
            store.upsert(info)
            logger.info(
                "Saved paper info: %s | method=%s",
                relative_path,
                info.method.get("method_name", ""),
            )

    logger.info("Extracted structured info for %s papers", len(results))
    return results


def paper_info_to_dict(info: PaperInfo) -> dict[str, Any]:
    return {
        "relative_path": info.relative_path,
        "file_name": info.file_name,
        "category": info.category,
        "metadata": info.metadata,
        "method": info.method,
        "tags": info.tags,
        "extraction_model": info.extraction_model,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract structured paper metadata and method information.")
    parser.add_argument("--limit", type=int, default=3, help="Number of PDFs to process.")
    parser.add_argument("--all", action="store_true", help="Process all PDFs under PAPER_ROOT.")
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES, help="Front-matter pages to read.")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="Maximum text chars sent to LLM.")
    parser.add_argument("--dry-run", action="store_true", help="Print extracted JSON without writing SQLite.")
    parser.add_argument("--force", action="store_true", help="Re-extract papers already present in SQLite.")
    parser.add_argument("--list", action="store_true", help="List recently extracted paper records.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)

    if args.list:
        store = PaperInfoStore()
        print(json.dumps(store.list_recent(limit=args.limit), indent=2, ensure_ascii=False))
        return

    limit = None if args.all else args.limit
    extract_papers(
        settings=settings,
        limit=limit,
        max_pages=args.max_pages,
        max_chars=args.max_chars,
        dry_run=args.dry_run,
        force=args.force,
    )


if __name__ == "__main__":
    main()
