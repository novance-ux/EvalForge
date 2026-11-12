"""The single path through which every LLM call flows.

Both the runner (model under test) and the LLM judge call
``Generator.generate``, so caching, cost accounting and concurrency limits
apply uniformly and provider-agnostically.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.models.results import LLMResponse
from app.models.suite import ModelSpec
from app.providers.registry import ProviderRegistry
from app.runner.cache import ResponseCache, cache_key
from app.runner.cost import PricingTable


class Generator:
    def __init__(
        self,
        registry: ProviderRegistry,
        pricing: PricingTable,
        cache: ResponseCache | None = None,
        max_concurrency: int = 4,
    ):
        self.registry = registry
        self.pricing = pricing
        self.cache = cache
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def generate(
        self,
        spec: ModelSpec | str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_schema: dict[str, Any] | None = None,
        input_vars: dict[str, Any] | None = None,
        sample_index: int = 0,
    ) -> LLMResponse:
        spec = ModelSpec.parse(spec)
        provider = self.registry.get(spec.provider)
        model = self.registry.resolve_model(spec.provider, spec.model)
        params = {"temperature": temperature, "max_tokens": max_tokens, "response_schema": response_schema}

        key = cache_key(provider.name, model, prompt, params, input_vars, sample_index)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return hit

        async with self.semaphore:
            response = await provider.generate(
                prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_schema=response_schema if provider.supports_structured_output else None,
            )
        response.cost = self.pricing.cost(provider.name, model, response.input_tokens, response.output_tokens)
        if self.cache is not None:
            self.cache.put(key, response)
        return response
