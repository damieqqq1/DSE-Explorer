"""MCP server entry point."""

from __future__ import annotations

import argparse
import sys

from fastmcp import FastMCP

from mcp_server.tools.method_matrix import register_method_matrix_tools
from mcp_server.tools.qdrant_rag import register_qdrant_rag_tools
from mcp_server.tools.web_search import register_web_search_tools
from utils.config import get_settings
from utils.logger import configure_logging


SERVER_NAME = "dse-explorer-mcp"


def create_mcp_server() -> FastMCP:
    """Create and configure the DSE Explorer MCP server."""

    mcp = FastMCP(
        name=SERVER_NAME,
        instructions=(
            "Tools for searching a local DSE/MOO paper corpus and supporting "
            "agentic design-space-exploration research workflows."
        ),
    )
    register_qdrant_rag_tools(mcp)
    register_method_matrix_tools(mcp)
    register_web_search_tools(mcp)
    return mcp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DSE Explorer MCP server.")
    parser.add_argument(
        "--transport",
        choices=("stdio", "http", "sse"),
        default="stdio",
        help="MCP transport to use.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP/SSE transports.")
    parser.add_argument("--port", type=int, default=8765, help="Port for HTTP/SSE transports.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    configure_logging(settings.log_level, stream=sys.stderr if args.transport == "stdio" else sys.stdout)

    mcp = create_mcp_server()
    if args.transport == "stdio":
        mcp.run(transport="stdio", show_banner=False)
        return

    mcp.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
