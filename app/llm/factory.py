"""Construct LLM providers from settings. Lazy imports keep optional SDKs optional."""
from __future__ import annotations

from ..config import ProviderName, Settings, get_settings
from .base import LLMProvider


class ProviderConfigError(RuntimeError):
    pass


def _build(provider: ProviderName, s: Settings, *, vision: bool) -> LLMProvider:
    if provider == "openai":
        if not s.openai_api_key:
            raise ProviderConfigError("OPENAI_API_KEY is not set")
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(
            api_key=s.openai_api_key,
            text_model=s.openai_model,
            vision_model=s.openai_vision_model,
        )

    if provider == "azure":
        if not (s.azure_openai_api_key and s.azure_openai_endpoint):
            raise ProviderConfigError("Azure OpenAI key/endpoint not set")
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(
            api_key=s.azure_openai_api_key,
            text_model=s.azure_openai_deployment or "",
            vision_model=s.azure_openai_vision_deployment
            or s.azure_openai_deployment
            or "",
            azure=True,
            azure_endpoint=s.azure_openai_endpoint,
            azure_api_version=s.azure_openai_api_version,
        )

    if provider == "anthropic":
        if not s.anthropic_api_key:
            raise ProviderConfigError("ANTHROPIC_API_KEY is not set")
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            api_key=s.anthropic_api_key,
            text_model=s.anthropic_model,
            vision_model=s.anthropic_vision_model,
        )

    if provider == "ollama":
        from .ollama_provider import OllamaProvider

        return OllamaProvider(
            base_url=s.ollama_base_url,
            text_model=s.ollama_model,
            vision_model=s.ollama_vision_model,
        )

    raise ProviderConfigError(f"Unknown provider: {provider}")


def build_llm_provider(settings: Settings | None = None) -> LLMProvider:
    s = settings or get_settings()
    return _build(s.llm_provider, s, vision=False)


def build_vision_provider(settings: Settings | None = None) -> LLMProvider:
    s = settings or get_settings()
    return _build(s.effective_vision_provider, s, vision=True)
