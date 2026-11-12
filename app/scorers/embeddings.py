"""Embedding provider abstraction used by the semantic-similarity scorer.

* ``local``  - dependency-free hashed bag-of-words + character trigrams.
               Crude, but offline and deterministic (good for tests/demos).
* ``openai`` - any OpenAI-compatible ``/embeddings`` endpoint.
* ``ollama`` - local Ollama ``/api/embed`` endpoint.

Spec strings: ``"local"``, ``"openai"``, ``"openai/text-embedding-3-small"``,
``"ollama/nomic-embed-text"``.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

import httpx
import numpy as np

from app.config.settings import Settings, get_settings
from app.providers.base import ProviderError, raise_for_status


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[np.ndarray]: ...


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class LocalHashingEmbedding(EmbeddingProvider):
    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim)
        lowered = text.lower()
        features = re.findall(r"\w+", lowered)
        compact = re.sub(r"\s+", " ", lowered)
        features += [compact[i : i + 3] for i in range(max(0, len(compact) - 2))]
        for f in features:
            h = int(hashlib.md5(f.encode()).hexdigest()[:8], 16)
            vec[h % self.dim] += 1.0
        return vec

    async def embed(self, texts: list[str]) -> list[np.ndarray]:
        return [self._vector(t) for t in texts]


class OpenAIEmbedding(EmbeddingProvider):
    def __init__(self, api_key: str | None, base_url: str, model: str, timeout_s: float = 60):
        self.api_key, self.base_url, self.model, self.timeout_s = api_key, base_url.rstrip("/"), model, timeout_s

    async def embed(self, texts: list[str]) -> list[np.ndarray]:
        if not self.api_key:
            raise ProviderError("openai embeddings: OPENAI_API_KEY is not set")
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": texts},
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        raise_for_status(r, "openai-embeddings")
        return [np.array(d["embedding"]) for d in r.json()["data"]]


class OllamaEmbedding(EmbeddingProvider):
    def __init__(self, base_url: str, model: str, timeout_s: float = 60):
        self.base_url, self.model, self.timeout_s = base_url.rstrip("/"), model, timeout_s

    async def embed(self, texts: list[str]) -> list[np.ndarray]:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(f"{self.base_url}/api/embed", json={"model": self.model, "input": texts})
        raise_for_status(r, "ollama-embeddings")
        return [np.array(v) for v in r.json()["embeddings"]]


def get_embedding_provider(spec: str, settings: Settings | None = None) -> EmbeddingProvider:
    s = settings or get_settings()
    kind, _, model = spec.partition("/")
    if kind == "local":
        return LocalHashingEmbedding()
    if kind == "openai":
        return OpenAIEmbedding(s.openai_api_key, s.openai_base_url, model or s.openai_embedding_model, s.timeout_s)
    if kind == "ollama":
        return OllamaEmbedding(s.ollama_base_url, model or s.ollama_embedding_model, s.timeout_s)
    raise ValueError(f"Unknown embedding provider '{spec}' (use local, openai or ollama)")
