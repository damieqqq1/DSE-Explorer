"""Embedding clients used by the RAG pipeline."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Iterable
from typing import Any

import requests

from utils.config import AppSettings, get_settings


DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_DASHSCOPE_MODEL = "text-embedding-v3"


class EmbeddingClient:
    """Small OpenAI-compatible embedding wrapper.

    DashScope's compatible mode follows the OpenAI `/embeddings` shape, so the
    same implementation also works for other compatible providers.
    """

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model_type = self.settings.embedding.model_type.lower()
        self.model_name = self.settings.embedding.model_name
        self.api_key = self.settings.embedding.api_key
        self.base_url = self.settings.embedding.base_url
        self.vector_size = self.settings.qdrant.vector_size
        self.timeout = self.settings.qdrant.timeout

        if self.model_type == "dashscope":
            self.model_name = self.model_name or DEFAULT_DASHSCOPE_MODEL
            self.base_url = self.base_url or DEFAULT_DASHSCOPE_BASE_URL
            self.max_batch_size = 10
        elif self.model_type in {"openai", "openai_compatible"}:
            if not self.model_name:
                raise ValueError("EMBED_MODEL_NAME is required for OpenAI-compatible embeddings.")
            if not self.base_url:
                raise ValueError("EMBED_BASE_URL is required for OpenAI-compatible embeddings.")
            self.max_batch_size = 128
        elif self.model_type == "hash":
            self.model_name = self.model_name or "hash-embedding"
            self.max_batch_size = 1024
        else:
            raise ValueError(f"Unsupported EMBED_MODEL_TYPE: {self.model_type}")

        if self.model_type != "hash" and not self.api_key:
            raise ValueError("Missing EMBED_API_KEY or DASHSCOPE_API_KEY in .env.")

    def embed_documents(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        vectors: list[list[float]] = []
        effective_batch_size = min(batch_size, self.max_batch_size)
        for batch in _batched(texts, effective_batch_size):
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text], batch_size=1)[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.model_type == "hash":
            return [_hash_embedding(text, self.vector_size) for text in texts]

        url = f"{self.base_url.rstrip('/')}/embeddings"
        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": texts,
        }
        if self.vector_size:
            payload["dimensions"] = self.vector_size

        response = None
        for attempt in range(3):
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code < 500:
                break
            time.sleep(1.5 * (attempt + 1))
        if response is None:
            raise RuntimeError("Embedding request was not sent.")
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(f"Embedding request failed: {response.text[:1000]}") from exc
        data = response.json()
        items = sorted(data["data"], key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in items]


def _batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def _hash_embedding(text: str, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = -1.0 if digest[4] % 2 else 1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
