"""Provider adapter interface.

Every provider implements ``_request`` (a single HTTP call). The shared
``generate`` method adds what every provider needs, so the runner never has
to know which provider it is talking to:

* timeout per attempt (``asyncio.wait_for``)
* retries with exponential backoff + jitter for *retryable* failures only
* latency measurement
"""

from __future__ import annotations

import asyncio
import random
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

from app.models.results import LLMResponse


class ProviderError(Exception):
    """Non-retryable provider failure (bad request, auth error, bad config)."""


class RetryableProviderError(ProviderError):
    """Transient failure worth retrying (timeout, 429, 5xx, connection reset)."""


RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def raise_for_status(response: httpx.Response, provider: str) -> None:
    if response.status_code < 400:
        return
    body = response.text[:300]
    msg = f"{provider} HTTP {response.status_code}: {body}"
    if response.status_code in RETRYABLE_STATUS:
        raise RetryableProviderError(msg)
    raise ProviderError(msg)


class LLMProvider(ABC):
    """Base class for all provider adapters."""

    name: str = "base"
    supports_structured_output: bool = False

    def __init__(
        self,
        default_model: str | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        backoff_base_s: float = 0.5,
        backoff_max_s: float = 8.0,
    ):
        self.default_model = default_model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0,
        max_tokens: int | None = None,
        response_schema: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> LLMResponse:
        """Generate a completion with timeout, retries and latency tracking.

        ``response_schema`` requests structured (JSON-schema constrained) output
        from providers that support it; others ignore it.
        """
        model = model or self.default_model
        if not model:
            raise ProviderError(f"{self.name}: no model specified and no default configured")

        attempt = 0
        while True:
            start = time.perf_counter()
            try:
                response = await asyncio.wait_for(
                    self._request(prompt, model, temperature, max_tokens, response_schema, seed),
                    timeout=self.timeout_s,
                )
                if not response.latency_ms:
                    response.latency_ms = (time.perf_counter() - start) * 1000
                return response
            except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
                error: ProviderError = RetryableProviderError(
                    f"{self.name}: timed out after {self.timeout_s}s ({type(exc).__name__})"
                )
            except httpx.TransportError as exc:
                error = RetryableProviderError(f"{self.name}: transport error: {exc}")
            except RetryableProviderError as exc:
                error = exc
            # ProviderError (non-retryable) propagates immediately.

            if attempt >= self.max_retries:
                raise ProviderError(f"{error} (gave up after {attempt + 1} attempts)") from error
            await asyncio.sleep(self._backoff(attempt))
            attempt += 1

    def _backoff(self, attempt: int) -> float:
        delay = min(self.backoff_max_s, self.backoff_base_s * (2**attempt))
        return delay * (0.5 + random.random() / 2)  # jitter in [50%, 100%]

    @abstractmethod
    async def _request(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int | None,
        response_schema: dict[str, Any] | None,
        seed: int | None,
    ) -> LLMResponse:
        """Perform exactly one request. Raise RetryableProviderError for transient errors."""

    async def aclose(self) -> None:  # pragma: no cover - overridden by HTTP providers
        return None
