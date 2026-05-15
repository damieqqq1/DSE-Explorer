"""Global configuration loader.

This module is intentionally dependency-light. It loads values from `.env`
once, exposes typed settings, and avoids printing secrets by default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_PAPER_ROOT = PROJECT_ROOT / "Paper_ljh"


def _load_env() -> None:
    load_dotenv(ENV_PATH, override=False)


def _get_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def _get_int(name: str, default: int) -> int:
    raw = _get_str(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc


def _get_float(name: str, default: float) -> float:
    raw = _get_str(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number.") from exc


def _mask_secret(value: str) -> str:
    if not value:
        return "<missing>"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


@dataclass(frozen=True)
class LLMSettings:
    model_id: str
    api_key: str
    base_url: str
    timeout: float
    temperature: float


@dataclass(frozen=True)
class QdrantSettings:
    url: str
    api_key: str
    collection: str
    vector_size: int
    distance: str
    timeout: float


@dataclass(frozen=True)
class WebSearchSettings:
    tavily_api_key: str
    serpapi_api_key: str


@dataclass(frozen=True)
class EmbeddingSettings:
    model_type: str
    model_name: str
    api_key: str
    base_url: str
    dashscope_api_key: str


@dataclass(frozen=True)
class AppSettings:
    project_root: Path
    env_path: Path
    paper_root: Path
    log_level: str
    github_token: str
    llm: LLMSettings
    qdrant: QdrantSettings
    web_search: WebSearchSettings
    embedding: EmbeddingSettings

    def validate_for_llm(self) -> None:
        if not self.llm.api_key:
            raise ValueError("Missing LLM_API_KEY or DEEPSEEK_API_KEY in .env.")

    def validate_for_papers(self) -> None:
        if not self.paper_root.exists():
            raise FileNotFoundError(f"Paper directory does not exist: {self.paper_root}")

    def masked_summary(self) -> dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "env_path": str(self.env_path),
            "paper_root": str(self.paper_root),
            "log_level": self.log_level,
            "github_token": _mask_secret(self.github_token),
            "llm": {
                "model_id": self.llm.model_id,
                "base_url": self.llm.base_url,
                "api_key": _mask_secret(self.llm.api_key),
                "timeout": self.llm.timeout,
                "temperature": self.llm.temperature,
            },
            "qdrant": {
                "url": self.qdrant.url,
                "api_key": _mask_secret(self.qdrant.api_key),
                "collection": self.qdrant.collection,
                "vector_size": self.qdrant.vector_size,
                "distance": self.qdrant.distance,
                "timeout": self.qdrant.timeout,
            },
            "web_search": {
                "tavily_api_key": _mask_secret(self.web_search.tavily_api_key),
                "serpapi_api_key": _mask_secret(self.web_search.serpapi_api_key),
            },
            "embedding": {
                "model_type": self.embedding.model_type,
                "model_name": self.embedding.model_name,
                "api_key": _mask_secret(self.embedding.api_key),
                "base_url": self.embedding.base_url,
                "dashscope_api_key": _mask_secret(self.embedding.dashscope_api_key),
            },
        }


def get_settings() -> AppSettings:
    _load_env()

    llm_api_key = _get_str("LLM_API_KEY") or _get_str("DEEPSEEK_API_KEY")
    embed_api_key = _get_str("EMBED_API_KEY") or _get_str("DASHSCOPE_API_KEY")
    paper_root = Path(_get_str("PAPER_ROOT", str(DEFAULT_PAPER_ROOT))).expanduser()
    if not paper_root.is_absolute():
        paper_root = PROJECT_ROOT / paper_root

    return AppSettings(
        project_root=PROJECT_ROOT,
        env_path=ENV_PATH,
        paper_root=paper_root,
        log_level=_get_str("LOG_LEVEL", "INFO").upper(),
        github_token=_get_str("GITHUB_PERSONAL_ACCESS_TOKEN") or _get_str("GITHUB_TOKEN"),
        llm=LLMSettings(
            model_id=_get_str("LLM_MODEL_ID", "deepseek-chat"),
            api_key=llm_api_key,
            base_url=_get_str("LLM_BASE_URL", "https://api.deepseek.com"),
            timeout=_get_float("LLM_TIMEOUT", 60.0),
            temperature=_get_float("LLM_TEMPERATURE", 0.2),
        ),
        qdrant=QdrantSettings(
            url=_get_str("QDRANT_URL", "http://localhost:6333"),
            api_key=_get_str("QDRANT_API_KEY"),
            collection=_get_str("QDRANT_COLLECTION", "dse_papers"),
            vector_size=_get_int("QDRANT_VECTOR_SIZE", 1536),
            distance=_get_str("QDRANT_DISTANCE", "Cosine"),
            timeout=_get_float("QDRANT_TIMEOUT", 30.0),
        ),
        web_search=WebSearchSettings(
            tavily_api_key=_get_str("TAVILY_API_KEY"),
            serpapi_api_key=_get_str("SERPAPI_API_KEY"),
        ),
        embedding=EmbeddingSettings(
            model_type=_get_str("EMBED_MODEL_TYPE", "openai_compatible"),
            model_name=_get_str("EMBED_MODEL_NAME", "text-embedding-v1"),
            api_key=embed_api_key,
            base_url=_get_str("EMBED_BASE_URL"),
            dashscope_api_key=_get_str("DASHSCOPE_API_KEY"),
        ),
    )
