import pytest

from app.config import Settings
from app.llm.factory import ProviderConfigError, build_llm_provider, build_vision_provider


def test_missing_openai_key_raises():
    s = Settings(llm_provider="openai", openai_api_key=None)
    with pytest.raises(ProviderConfigError):
        build_llm_provider(s)


def test_missing_anthropic_key_raises():
    s = Settings(llm_provider="anthropic", anthropic_api_key=None)
    with pytest.raises(ProviderConfigError):
        build_llm_provider(s)


def test_azure_requires_endpoint():
    s = Settings(llm_provider="azure", azure_openai_api_key="k", azure_openai_endpoint=None)
    with pytest.raises(ProviderConfigError):
        build_llm_provider(s)


def test_ollama_needs_no_key():
    s = Settings(llm_provider="ollama")
    provider = build_llm_provider(s)
    assert provider.name == "ollama"


def test_vision_provider_defaults_to_llm_provider():
    s = Settings(llm_provider="ollama", vision_provider=None)
    assert s.effective_vision_provider == "ollama"
    assert build_vision_provider(s).name == "ollama"


def test_vision_provider_can_differ():
    s = Settings(llm_provider="ollama", vision_provider="ollama")
    assert s.effective_vision_provider == "ollama"
