"""Web research search tools for MCP."""

from __future__ import annotations

from typing import Any, Literal

import requests

from utils.config import get_settings
from utils.logger import get_logger


logger = get_logger(__name__)

ResearchTopic = Literal["design_space_exploration", "combinatorial_optimization", "both"]
SearchProvider = Literal["auto", "tavily", "serpapi"]

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
SERPAPI_SEARCH_URL = "https://serpapi.com/search.json"
DEFAULT_MAX_RESULTS = 5
MAX_RESULTS = 10


def search_research_web(
    query: str = "",
    topic: ResearchTopic = "both",
    max_results: int = DEFAULT_MAX_RESULTS,
    provider: SearchProvider = "auto",
) -> dict[str, Any]:
    """Search the web for research about DSE or combinatorial optimization.

    Args:
        query: Optional user query. It will be augmented with research-focused terms.
        topic: Search scope: design space exploration, combinatorial optimization, or both.
        max_results: Number of results to return. Capped at 10.
        provider: Search provider. `auto` prefers Tavily and falls back to SerpApi.

    Returns:
        JSON-serializable search results with title, url, snippet, source provider, and score/rank.
    """

    settings = get_settings()
    safe_max_results = min(max(1, max_results), MAX_RESULTS)
    augmented_query = build_research_query(query, topic)

    if provider == "tavily":
        return tavily_search(augmented_query, safe_max_results, settings.web_search.tavily_api_key)
    if provider == "serpapi":
        return serpapi_search(augmented_query, safe_max_results, settings.web_search.serpapi_api_key)

    if settings.web_search.tavily_api_key:
        try:
            return tavily_search(augmented_query, safe_max_results, settings.web_search.tavily_api_key)
        except Exception:
            if not settings.web_search.serpapi_api_key:
                raise
            logger.exception("Tavily search failed; falling back to SerpApi.")

    if settings.web_search.serpapi_api_key:
        return serpapi_search(augmented_query, safe_max_results, settings.web_search.serpapi_api_key)

    raise ValueError("Missing TAVILY_API_KEY or SERPAPI_API_KEY in .env.")


def build_research_query(query: str, topic: ResearchTopic) -> str:
    topic_terms = {
        "design_space_exploration": '"design space exploration"',
        "combinatorial_optimization": '"combinatorial optimization"',
        "both": '("design space exploration" OR "combinatorial optimization")',
    }[topic]

    user_query = query.strip()
    if user_query:
        return f"{user_query} {topic_terms} research paper survey arXiv IEEE ACM"
    return f"{topic_terms} research paper survey arXiv IEEE ACM recent methods"


def tavily_search(query: str, max_results: int, api_key: str) -> dict[str, Any]:
    if not api_key:
        raise ValueError("Missing TAVILY_API_KEY in .env.")

    response = requests.post(
        TAVILY_SEARCH_URL,
        json={
            "api_key": api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    results = [
        {
            "rank": index,
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("content", ""),
            "score": item.get("score"),
            "published_date": item.get("published_date"),
            "provider": "tavily",
        }
        for index, item in enumerate(payload.get("results", []), start=1)
    ]
    return {
        "provider": "tavily",
        "query": query,
        "result_count": len(results),
        "results": results,
    }


def serpapi_search(query: str, max_results: int, api_key: str) -> dict[str, Any]:
    if not api_key:
        raise ValueError("Missing SERPAPI_API_KEY in .env.")

    response = requests.get(
        SERPAPI_SEARCH_URL,
        params={
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": max_results,
            "hl": "en",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    results = [
        {
            "rank": index,
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
            "score": None,
            "published_date": item.get("date"),
            "provider": "serpapi",
        }
        for index, item in enumerate(payload.get("organic_results", [])[:max_results], start=1)
    ]
    return {
        "provider": "serpapi",
        "query": query,
        "result_count": len(results),
        "results": results,
    }


def register_web_search_tools(mcp: Any) -> None:
    """Register web research search tools on a FastMCP server instance."""

    mcp.tool(
        name="search_research_web",
        description=(
            "Search the internet for research about design space exploration "
            "or combinatorial optimization. Use it for recent papers, surveys, "
            "methods, and external references beyond the local PDF corpus."
        ),
    )(search_research_web)
