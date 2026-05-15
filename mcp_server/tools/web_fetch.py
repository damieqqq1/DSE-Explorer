"""Web fetch tools — fetch URLs as Markdown, download papers."""

from __future__ import annotations

import ipaddress
import os
import re
import urllib.parse
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from html2text import HTML2Text

from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

FETCH_TIMEOUT = 30
MAX_CONTENT_BYTES = 5 * 1024 * 1024  # 5 MB

_PRIVATE_NETS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
]


def _is_safe_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.hostname
    if not host:
        return False
    host_lower = host.lower()
    # IPv4
    try:
        addr = ipaddress.ip_address(host_lower)
        if addr.version == 4:
            return not any(addr in net for net in _PRIVATE_NETS)
        return False  # block raw IPv6 for now
    except ValueError:
        pass
    if host_lower in ("localhost", "0.0.0.0", "[::1]"):
        return False
    return True


def _markdown_from_html(html: str, base_url: str) -> str:
    converter = HTML2Text()
    converter.body_width = 0
    converter.ignore_links = False
    converter.ignore_images = True
    converter.ignore_emphasis = False
    converter.protect_links = True
    return converter.handle(html)


def _plain_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return " ".join(soup.get_text().split())


def fetch_web_page(
    url: str,
    output_format: str = "markdown",
    max_chars: int = 20000,
) -> dict[str, Any]:
    """Fetch a web page and return its content as Markdown or plain text.

    Args:
        url: The URL to fetch (http/https only).
        output_format: "markdown" (default) or "text".
        max_chars: Maximum characters to return, capped at 50000.

    Returns:
        A dictionary with ``url``, ``title``, ``content``, ``content_type``, and ``char_count``.
    """
    if not _is_safe_url(url):
        return {"success": False, "error": "URL blocked: internal/private addresses are not allowed.", "url": url}

    safe_max_chars = min(max(100, max_chars), 50000)

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "dse-explorer/1.0 (research-assistant)"},
            timeout=FETCH_TIMEOUT,
            stream=True,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Fetch failed for %s: %s", url, exc)
        return {"success": False, "error": str(exc), "url": url}

    content_type = resp.headers.get("Content-Type", "")
    raw = b""
    for chunk in resp.iter_content(chunk_size=8192):
        raw += chunk
        if len(raw) > MAX_CONTENT_BYTES:
            break

    if b"text/html" in content_type.encode():
        html = raw.decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")
        title = soup.title.string.strip() if soup.title else ""
        if output_format == "text":
            content = _plain_text_from_html(html)
        else:
            content = _markdown_from_html(html, url)
    else:
        title = ""
        content = raw.decode("utf-8", errors="replace")

    content = content[:safe_max_chars]

    return {
        "success": True,
        "url": url,
        "title": title,
        "content": content,
        "content_type": content_type,
        "char_count": len(content),
    }


def download_paper_pdf(
    url: str,
    filename: str = "",
    category: str = "_downloaded",
) -> dict[str, Any]:
    """Download a PDF paper from a URL and save it to the paper corpus.

    Args:
        url: The URL to download from (http/https, should be a PDF link).
        filename: Optional filename. If empty, inferred from the URL.
        category: Subdirectory under PAPER_ROOT to save to. Default ``_downloaded``.

    Returns:
        A dictionary with ``url``, ``saved_path``, ``file_size``.
    """
    if not _is_safe_url(url):
        return {"success": False, "error": "URL blocked: internal/private addresses are not allowed.", "url": url}

    settings = get_settings()
    dest_dir = settings.paper_root / category
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not filename:
        parsed = urllib.parse.urlparse(url)
        fname = os.path.basename(parsed.path)
        if not fname or not fname.endswith(".pdf"):
            fname = "paper.pdf"
        filename = fname

    dest_path = dest_dir / filename

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "dse-explorer/1.0 (research-assistant)"},
            timeout=FETCH_TIMEOUT,
            stream=True,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        return {"success": False, "error": str(exc), "url": url}

    total = 0
    try:
        with open(dest_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)
                total += len(chunk)
    except OSError as exc:
        return {"success": False, "error": str(exc), "url": url}

    relative_path = dest_path.relative_to(settings.paper_root).as_posix()
    return {
        "success": True,
        "url": url,
        "saved_path": relative_path,
        "file_size": total,
        "message": f"Downloaded {total} bytes to {relative_path}",
    }


def register_web_fetch_tools(mcp: Any) -> None:
    """Register web fetch tools on a FastMCP server instance."""

    mcp.tool(
        name="fetch_web_page",
        description=(
            "Fetch a web page and return its content as Markdown or plain text. "
            "Use this to read the full text of a paper abstract page, a blog post, "
            "or documentation. Internal/private IPs are blocked. Content is capped "
            "at 50000 characters."
        ),
    )(fetch_web_page)

    mcp.tool(
        name="download_paper_pdf",
        description=(
            "Download a PDF from a URL and save it into the local paper corpus "
            "under Paper_ljh/_downloaded/. Use this to add new papers to the "
            "knowledge base for later ingestion."
        ),
    )(download_paper_pdf)
