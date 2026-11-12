"""Adapter for any OpenAI-compatible Chat Completions API.

Works with OpenAI and with the many servers that mimic its API (vLLM,
LM Studio, Groq, Together, OpenRouter, ...) by changing ``OPENAI_BASE_URL``.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.models.results import LLMResponse
from app.providers.base import LLMProvider, ProviderError, raise_for_status
from app.utils.misc import estimate_tokens


class OpenAICompatibleProvider(LLMProvider):
    name = "openai"
    supports_structured_output = True

    def __init__(
        self,
        api_key: str | None,
        base_url: str = "https://api.openai.com/v1",
        default_model: str | None = None,
        client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ):
        super().__init__(default_model=default_model, **kwargs)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = client

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    async def _request(self, prompt, model, temperature, max_tokens, response_schema, seed):
        if not self.api_key:
            raise ProviderError("openai: OPENAI_API_KEY is not set (see .env.example)")
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if seed is not None:
            payload["seed"] = seed
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": response_schema, "strict": False},
            }

        resp = await self.client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        raise_for_status(resp, self.name)
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"openai: unexpected response shape: {str(data)[:200]}") from exc
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=data.get("model", model),
            provider=self.name,
            input_tokens=usage.get("prompt_tokens", estimate_tokens(prompt)),
            output_tokens=usage.get("completion_tokens", estimate_tokens(text)),
            raw=data,
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
