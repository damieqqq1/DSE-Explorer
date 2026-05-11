"""DSE method comparison matrix tools for MCP."""

from __future__ import annotations

from typing import Any, Literal

from rag_pipeline.method_matrix import generate_method_matrix


MatrixFormat = Literal["markdown", "json", "csv"]


def generate_dse_method_matrix(
    keyword: str = "",
    category: str = "",
    method_names: list[str] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    limit: int = 20,
    output_format: MatrixFormat = "markdown",
    save_path: str = "",
) -> dict[str, Any]:
    """Generate a DSE method comparison matrix from extracted paper_info records.

    Args:
        keyword: Optional keyword filter over titles, methods, objectives, baselines, metrics, and tags.
        category: Optional paper category/path filter such as DSE or llm.
        method_names: Optional exact method names to include, for example ["GRL-DSE"].
        year_from: Optional minimum publication year.
        year_to: Optional maximum publication year.
        limit: Maximum matrix rows. Capped at 100.
        output_format: Matrix format: markdown, json, or csv.
        save_path: Optional output file path under the project directory, commonly reports/*.md.

    Returns:
        JSON-serializable dictionary containing the matrix, structured rows, filters, and output path.
    """

    safe_limit = min(max(1, limit), 100)
    return generate_method_matrix(
        keyword=keyword,
        category=category,
        method_names=method_names or [],
        year_from=year_from,
        year_to=year_to,
        limit=safe_limit,
        output_format=output_format,
        save_path=save_path,
    )


def register_method_matrix_tools(mcp: Any) -> None:
    """Register DSE method matrix tools on a FastMCP server instance."""

    mcp.tool(
        name="generate_dse_method_matrix",
        description=(
            "Generate a structured DSE/MOO method comparison matrix from extracted "
            "paper metadata and method information stored in SQLite. Use this when "
            "the user asks to compare DSE methods, optimization algorithms, objectives, "
            "benchmarks, baselines, metrics, or paper-level method characteristics."
        ),
    )(generate_dse_method_matrix)
