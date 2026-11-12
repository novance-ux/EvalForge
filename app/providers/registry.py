"""Provider registry / factory.

The runner asks the registry for "a provider called X" and never imports a
concrete adapter. To add a provider, write an ``LLMProvider`` subclass and
register a factory for it:

    from app.providers.registry import register_provider

    @register_provider("anthropic")
    def _make(settings):
        return MyAnthropicProvider(api_key=..., default_model=...)
"""

from __future__ import annotations

from typing import Callable

from app.config.settings import Settings, get_settings
from app.providers.base import LLMProvider, ProviderError

ProviderFactory = Callable[[Settings], LLMProvider]
_FACTORIES: dict[str, ProviderFactory] = {}


def register_provider(name: str) -> Callable[[ProviderFactory], ProviderFactory]:
    def deco(fn: ProviderFactory) -> ProviderFactory:
        _FACTORIES[name] = fn
        return fn

    return deco


def available_providers() -> list[str]:
    return sorted(_FACTORIES)


class ProviderRegistry:
    """Creates providers lazily and reuses one instance per provider name."""

    def __init__(self, settings: Settings | None = None, overrides: dict[str, LLMProvider] | None = None):
        self.settings = settings or get_settings()
        self._instances: dict[str, LLMProvider] = dict(overrides or {})

    def get(self, name: str) -> LLMProvider:
        if name not in self._instances:
            if name not in _FACTORIES:
                raise ProviderError(
                    f"Unknown provider '{name}'. Available: {', '.join(available_providers())}"
                )
            self._instances[name] = _FACTORIES[name](self.settings)
        return self._instances[name]

    def resolve_model(self, provider_name: str, model: str | None) -> str:
        provider = self.get(provider_name)
        resolved = model or provider.default_model
        if not resolved:
            raise ProviderError(f"No model given for provider '{provider_name}'")
        return resolved

    async def aclose(self) -> None:
        for p in self._instances.values():
            await p.aclose()


# --------------------------------------------------------------------------- #
# Built-in providers
# --------------------------------------------------------------------------- #
def _common(s: Settings) -> dict:
    return {"timeout_s": s.timeout_s, "max_retries": s.max_retries}


@register_provider("openai")
def _openai(s: Settings) -> LLMProvider:
    from app.providers.openai_compat import OpenAICompatibleProvider

    return OpenAICompatibleProvider(
        api_key=s.openai_api_key, base_url=s.openai_base_url, default_model=s.openai_model, **_common(s)
    )


@register_provider("ollama")
def _ollama(s: Settings) -> LLMProvider:
    from app.providers.ollama import OllamaProvider

    return OllamaProvider(base_url=s.ollama_base_url, default_model=s.ollama_model, **_common(s))


@register_provider("mock")
def _mock(s: Settings) -> LLMProvider:
    from app.providers.mock import MockProvider

    return MockProvider(default_model="mock-small", **_common(s))
