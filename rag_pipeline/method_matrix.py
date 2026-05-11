"""Generate DSE method comparison matrices from extracted paper information."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from rag_pipeline.paper_info import PAPER_INFO_DB_PATH
from utils.config import PROJECT_ROOT
from utils.logger import configure_logging, get_logger


logger = get_logger(__name__)

MatrixFormat = Literal["markdown", "json", "csv"]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports"
DEFAULT_COLUMNS = [
    "method_name",
    "title",
    "year",
    "problem",
    "core_idea",
    "optimization_algorithm",
    "model_or_agent",
    "objectives",
    "benchmarks",
    "baselines",
    "metrics",
    "main_results",
    "limitations",
    "source",
]


@dataclass(frozen=True)
class MethodMatrixRow:
    method_name: str
    title: str
    year: str
    venue: str
    problem: str
    core_idea: str
    optimization_algorithm: str
    model_or_agent: str
    design_variables: list[str]
    objectives: list[str]
    benchmarks: list[str]
    baselines: list[str]
    metrics: list[str]
    main_results: list[str]
    limitations: list[str]
    tags: list[str]
    source: str


def generate_method_matrix(
    *,
    keyword: str = "",
    category: str = "",
    method_names: list[str] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    limit: int = 20,
    output_format: MatrixFormat = "markdown",
    save_path: str = "",
    columns: list[str] | None = None,
    db_path: Path = PAPER_INFO_DB_PATH,
) -> dict[str, Any]:
    rows = load_method_rows(db_path)
    filters = {
        "keyword": keyword,
        "category": category,
        "method_names": method_names or [],
        "year_from": year_from,
        "year_to": year_to,
        "limit": limit,
    }
    filtered = filter_rows(
        rows,
        keyword=keyword,
        category=category,
        method_names=method_names or [],
        year_from=year_from,
        year_to=year_to,
    )[: max(1, min(limit, 100))]
    selected_columns = columns or DEFAULT_COLUMNS
    matrix = render_matrix(filtered, output_format=output_format, columns=selected_columns)
    resolved_output_path = ""
    if save_path:
        output_path = resolve_output_path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(matrix, encoding="utf-8")
        resolved_output_path = str(output_path)

    return {
        "format": output_format,
        "paper_count": len(filtered),
        "available_paper_count": len(rows),
        "filters": filters,
        "columns": selected_columns,
        "matrix": matrix,
        "rows": [asdict(row) for row in filtered],
        "output_path": resolved_output_path,
    }


def load_method_rows(db_path: Path = PAPER_INFO_DB_PATH) -> list[MethodMatrixRow]:
    if not db_path.exists():
        logger.warning("Paper info database does not exist: %s", db_path)
        return []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                relative_path, title, year, venue, method_name, problem, core_idea,
                optimization_algorithm, model_or_agent, design_variables_json,
                objectives_json, benchmarks_json, baselines_json, metrics_json,
                main_results_json, limitations_json, tags_json
            FROM paper_info
            ORDER BY year DESC, method_name ASC, title ASC
            """
        ).fetchall()

    return [
        MethodMatrixRow(
            method_name=row["method_name"],
            title=row["title"],
            year=row["year"],
            venue=row["venue"],
            problem=row["problem"],
            core_idea=row["core_idea"],
            optimization_algorithm=row["optimization_algorithm"],
            model_or_agent=row["model_or_agent"],
            design_variables=json_list(row["design_variables_json"]),
            objectives=json_list(row["objectives_json"]),
            benchmarks=json_list(row["benchmarks_json"]),
            baselines=json_list(row["baselines_json"]),
            metrics=json_list(row["metrics_json"]),
            main_results=json_list(row["main_results_json"]),
            limitations=json_list(row["limitations_json"]),
            tags=json_list(row["tags_json"]),
            source=row["relative_path"],
        )
        for row in rows
    ]


def filter_rows(
    rows: list[MethodMatrixRow],
    *,
    keyword: str,
    category: str,
    method_names: list[str],
    year_from: int | None,
    year_to: int | None,
) -> list[MethodMatrixRow]:
    keyword_terms = [term.lower() for term in keyword.split() if term.strip()]
    method_set = {name.lower() for name in method_names if name.strip()}
    category_value = category.strip().lower()

    filtered: list[MethodMatrixRow] = []
    for row in rows:
        if category_value and category_value not in row.source.lower():
            continue
        if method_set and row.method_name.lower() not in method_set:
            continue
        year = parse_year(row.year)
        if year_from is not None and (year is None or year < year_from):
            continue
        if year_to is not None and (year is None or year > year_to):
            continue
        haystack = row_search_text(row)
        if keyword_terms and not all(term in haystack for term in keyword_terms):
            continue
        filtered.append(row)
    return filtered


def render_matrix(rows: list[MethodMatrixRow], *, output_format: MatrixFormat, columns: list[str]) -> str:
    if output_format == "json":
        return json.dumps([asdict(row) for row in rows], indent=2, ensure_ascii=False)
    if output_format == "csv":
        return render_csv(rows, columns)
    if output_format == "markdown":
        return render_markdown(rows, columns)
    raise ValueError(f"Unsupported matrix format: {output_format}")


def render_markdown(rows: list[MethodMatrixRow], columns: list[str]) -> str:
    if not rows:
        return "| Result |\n|---|\n| No paper_info records matched the filters. |\n"

    labels = [column_label(column) for column in columns]
    lines = [
        "| " + " | ".join(escape_markdown(label) for label in labels) + " |",
        "| " + " | ".join("---" for _ in labels) + " |",
    ]
    for row in rows:
        values = [format_cell(row, column, compact=True) for column in columns]
        lines.append("| " + " | ".join(escape_markdown(value) for value in values) + " |")
    return "\n".join(lines) + "\n"


def render_csv(rows: list[MethodMatrixRow], columns: list[str]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([column_label(column) for column in columns])
    for row in rows:
        writer.writerow([format_cell(row, column, compact=False) for column in columns])
    return buffer.getvalue()


def format_cell(row: MethodMatrixRow, column: str, *, compact: bool) -> str:
    value = getattr(row, column, "")
    if isinstance(value, list):
        text = ", ".join(value[:4] if compact else value)
        if compact and len(value) > 4:
            text += ", ..."
        return text
    text = str(value or "")
    if compact:
        return truncate(text, 180)
    return text


def json_list(raw: str) -> list[str]:
    try:
        value = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def row_search_text(row: MethodMatrixRow) -> str:
    values: list[str] = []
    for value in asdict(row).values():
        if isinstance(value, list):
            values.extend(value)
        else:
            values.append(str(value))
    return " ".join(values).lower()


def parse_year(value: str) -> int | None:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) < 4:
        return None
    try:
        return int(digits[:4])
    except ValueError:
        return None


def truncate(text: str, max_length: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3].rstrip() + "..."


def escape_markdown(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def column_label(column: str) -> str:
    labels = {
        "method_name": "Method",
        "title": "Paper",
        "year": "Year",
        "venue": "Venue",
        "problem": "Problem",
        "core_idea": "Core Idea",
        "optimization_algorithm": "Optimizer",
        "model_or_agent": "Model/Agent",
        "design_variables": "Design Variables",
        "objectives": "Objectives",
        "benchmarks": "Benchmarks",
        "baselines": "Baselines",
        "metrics": "Metrics",
        "main_results": "Main Results",
        "limitations": "Limitations",
        "tags": "Tags",
        "source": "Source",
    }
    return labels.get(column, column)


def resolve_output_path(save_path: str) -> Path:
    path = Path(save_path)
    if not path.is_absolute():
        path = DEFAULT_OUTPUT_DIR / path
    resolved = path.resolve()
    project_root = PROJECT_ROOT.resolve()
    if project_root not in resolved.parents and resolved != project_root:
        raise ValueError(f"save_path must stay inside the project directory: {PROJECT_ROOT}")
    return resolved


def parse_method_names(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a DSE method comparison matrix.")
    parser.add_argument("--keyword", default="", help="Keyword filter over extracted paper information.")
    parser.add_argument("--category", default="", help="Category/path filter such as DSE or llm.")
    parser.add_argument("--method-names", default="", help="Comma-separated exact method names.")
    parser.add_argument("--year-from", type=int, default=None, help="Minimum publication year.")
    parser.add_argument("--year-to", type=int, default=None, help="Maximum publication year.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum rows.")
    parser.add_argument("--format", choices=("markdown", "json", "csv"), default="markdown")
    parser.add_argument("--save-path", default="", help="Optional output path under the project directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging("INFO")
    result = generate_method_matrix(
        keyword=args.keyword,
        category=args.category,
        method_names=parse_method_names(args.method_names),
        year_from=args.year_from,
        year_to=args.year_to,
        limit=args.limit,
        output_format=args.format,
        save_path=args.save_path,
    )
    print(result["matrix"])
    if result["output_path"]:
        logger.info("Saved method matrix: %s", result["output_path"])


if __name__ == "__main__":
    main()
