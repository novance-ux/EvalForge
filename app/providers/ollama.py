"""Adapter for a local Ollama server (https://ollama.com)."""

from __future__ import annotations

from typing import Any

import httpx

from app.models.results import LLMResponse
from app.providers.base import LLMProvider, ProviderError, raise_for_status
from app.utils.misc import estimate_tokens


class OllamaProvider(LLMProvider):
    name = "ollama"
    supports_structured_output = True  # via the `format` field (JSON schema)

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str | None = None,
        client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ):
        super().__init__(default_model=default_model, **kwargs)
        self.base_url = base_url.rstrip("/")
        self._client = client

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    async def _request(self, prompt, model, temperature, max_tokens, response_schema, seed):
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if seed is not None:
            options["seed"] = seed
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if response_schema is not None:
            payload["format"] = response_schema

        resp = await self.client.post(f"{self.base_url}/api/generate", json=payload)
        raise_for_status(resp, self.name)
        data = resp.json()
        if "response" not in data:
            raise ProviderError(f"ollama: unexpected response shape: {str(data)[:200]}")
        text = data["response"]
        return LLMResponse(
            text=text,
            model=data.get("model", model),
            provider=self.name,
            input_tokens=data.get("prompt_eval_count") or estimate_tokens(prompt),
            output_tokens=data.get("eval_count") or estimate_tokens(text),
            raw=data,
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
