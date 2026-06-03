"""Central configuration. All knobs are env-driven (12-factor)."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["openai", "anthropic", "azure", "ollama"]
ParserBackend = Literal["auto", "docling", "llamaparse", "vision_llm"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # App
    app_env: str = "production"
    log_level: str = "INFO"
    max_upload_mb: int = 50

    # Provider selection
    llm_provider: ProviderName = "anthropic"
    vision_provider: ProviderName | None = None  # falls back to llm_provider

    # OpenAI
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.4"
    openai_vision_model: str = "gpt-5.4"

    # Anthropic
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-4-6"
    anthropic_vision_model: str = "claude-opus-4-6"

    # Azure OpenAI
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str = "2025-04-01-preview"
    azure_openai_deployment: str | None = None
    azure_openai_vision_deployment: str | None = None

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.3"
    ollama_vision_model: str = "llama3.2-vision"

    # Parsing
    parser_backend: ParserBackend = "auto"
    llama_cloud_api_key: str | None = None
    parser_confidence_threshold: float = Field(0.62, ge=0.0, le=1.0)
    force_vision_llm: bool = False
    vision_image_detail: Literal["auto", "original"] = "original"
    vision_render_dpi: int = 200

    @property
    def effective_vision_provider(self) -> ProviderName:
        return self.vision_provider or self.llm_provider


@lru_cache
def get_settings() -> Settings:
    return Settings()
