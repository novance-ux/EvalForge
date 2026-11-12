import asyncio
import json

import httpx
import pytest

from app.models.results import LLMResponse
from app.providers.base import LLMProvider, ProviderError, RetryableProviderError
from app.providers.mock import MockProvider
from app.providers.ollama import OllamaProvider
from app.providers.openai_compat import OpenAICompatibleProvider
from app.providers.registry import ProviderRegistry, available_providers


class Flaky(LLMProvider):
    name = "flaky"

    def __init__(self, failures: int, error=RetryableProviderError, **kw):
        super().__init__(default_model="m", backoff_base_s=0.001, **kw)
        self.failures, self.error, self.calls = failures, error, 0

    async def _request(self, prompt, model, temperature, max_tokens, response_schema, seed):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error("boom")
        return LLMResponse(text="ok", model=model, provider=self.name)


async def test_retries_then_succeeds():
    p = Flaky(failures=2, max_retries=3)
    r = await p.generate("hi")
    assert r.text == "ok" and p.calls == 3
    assert r.latency_ms > 0


async def test_gives_up_after_max_retries():
    p = Flaky(failures=10, max_retries=2)
    with pytest.raises(ProviderError, match="gave up after 3 attempts"):
        await p.generate("hi")
    assert p.calls == 3


async def test_non_retryable_error_is_not_retried():
    p = Flaky(failures=10, error=ProviderError, max_retries=5)
    with pytest.raises(ProviderError):
        await p.generate("hi")
    assert p.calls == 1


async def test_timeout_is_retryable():
    class Slow(Flaky):
        async def _request(self, *a):
            self.calls += 1
            if self.calls == 1:
                await asyncio.sleep(1)
            return LLMResponse(text="ok", model="m", provider="slow")

    p = Slow(failures=0, max_retries=1, timeout_s=0.05)
    assert (await p.generate("hi")).text == "ok"
    assert p.calls == 2


def test_backoff_is_exponential_and_capped():
    p = Flaky(failures=0)
    p.backoff_base_s, p.backoff_max_s = 1.0, 4.0
    assert 0.5 <= p._backoff(0) <= 1.0
    assert 2.0 <= p._backoff(2) <= 4.0
    assert p._backoff(10) <= 4.0


# ------------------------------------------------------------------ HTTP adapters (no network)
def transport(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_openai_adapter_parses_response_and_sends_schema():
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json={
            "model": "gpt-x", "choices": [{"message": {"content": '{"category":"billing"}'}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5},
        })

    p = OpenAICompatibleProvider(api_key="sk-test", default_model="gpt-x", client=transport(handler))
    r = await p.generate("hello", response_schema={"type": "object"})
    assert r.text == '{"category":"billing"}' and r.input_tokens == 12 and r.output_tokens == 5
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["response_format"]["type"] == "json_schema"


async def test_openai_adapter_retries_429():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    p = OpenAICompatibleProvider(api_key="k", default_model="m", client=transport(handler), backoff_base_s=0.001)
    assert (await p.generate("x")).text == "ok"
    assert calls["n"] == 2


async def test_openai_adapter_does_not_retry_401():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(401, json={"error": "bad key"})

    p = OpenAICompatibleProvider(api_key="k", default_model="m", client=transport(handler))
    with pytest.raises(ProviderError, match="401"):
        await p.generate("x")
    assert calls["n"] == 1


async def test_openai_requires_api_key():
    p = OpenAICompatibleProvider(api_key=None, default_model="m")
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        await p.generate("x")


async def test_ollama_adapter():
    def handler(request):
        body = json.loads(request.content)
        assert body["stream"] is False and body["options"]["temperature"] == 0
        return httpx.Response(200, json={"model": "llama", "response": "hi", "prompt_eval_count": 7, "eval_count": 2})

    p = OllamaProvider(default_model="llama", client=transport(handler))
    r = await p.generate("x")
    assert (r.text, r.input_tokens, r.output_tokens, r.provider) == ("hi", 7, 2, "ollama")


async def test_mock_provider_is_deterministic_at_temperature_zero():
    p = MockProvider(time_scale=0)
    prompt = 'Classify into billing, account. Return ONLY valid JSON.\nTicket: """I was charged twice"""'
    a, b = await p.generate(prompt), await p.generate(prompt)
    assert a.text == b.text and a.latency_ms == b.latency_ms
    assert json.loads(a.text)["category"] in {"billing", "account"}


def test_registry():
    assert {"openai", "ollama", "mock"} <= set(available_providers())
    reg = ProviderRegistry()
    assert reg.get("mock") is reg.get("mock")
    with pytest.raises(ProviderError, match="Unknown provider"):
        reg.get("does-not-exist")
