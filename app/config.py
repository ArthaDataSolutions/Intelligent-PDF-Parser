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
    # Persistence — run history, per-run logs, and Q&A chat live in this SQLite file.
    db_path: str = "./data/app.db"
    # Default analyst-question count used by the run API when not specified.
    default_num_questions: int = 12
    # Default comma-separated comparable pharma companies used to seed analyst
    # peer-comparison questions when a run doesn't specify its own list.
    default_peers: str = ""
    # Ground peer-comparison questions with live web search (Anthropic web-search
    # tool) so they carry real citations + how a comparable company answered.
    # Requires an Anthropic text provider; silently skipped for other providers.
    # NOTE: each searched question incurs Anthropic web-search billing.
    peer_web_search: bool = True
    # Max web searches Claude may run per peer question.
    peer_web_search_max_uses: int = Field(5, ge=1, le=20)

    # Provider selection
    llm_provider: ProviderName = "anthropic"
    vision_provider: ProviderName | None = None  # falls back to llm_provider

    # OpenAI (current frontier line, June 2026: GPT-5.5 flagship, vision-capable)
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.5"
    openai_vision_model: str = "gpt-5.5"

    # Anthropic — strongest model for handwriting/vision, sonnet for cheaper Q&A
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"
    anthropic_vision_model: str = "claude-opus-4-8"

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
    handwriting_scan_char_threshold: int = Field(20, ge=0)
    handwriting_image_area_threshold: float = Field(0.15, ge=0.0, le=1.0)
    handwriting_text_area_threshold: float = Field(0.35, ge=0.0, le=1.0)
    force_vision_llm: bool = False
    vision_image_detail: Literal["auto", "original"] = "original"
    vision_render_dpi: int = 200
    vision_max_concurrency: int = Field(4, ge=1, le=16)

    @property
    def effective_vision_provider(self) -> ProviderName:
        return self.vision_provider or self.llm_provider


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Operational knobs that may be changed from the Settings UI and persisted to
# the DB. Secrets (API keys, endpoints) are deliberately excluded — they stay
# env-only and are never written to or returned from SQLite.
OVERRIDE_WHITELIST: frozenset[str] = frozenset({
    "log_level", "max_upload_mb", "default_num_questions", "default_peers",
    "peer_web_search", "peer_web_search_max_uses",
    "llm_provider", "vision_provider",
    "parser_backend", "force_vision_llm", "parser_confidence_threshold",
    "handwriting_scan_char_threshold", "handwriting_image_area_threshold",
    "handwriting_text_area_threshold", "vision_image_detail",
    "vision_render_dpi", "vision_max_concurrency",
    "openai_model", "openai_vision_model",
    "anthropic_model", "anthropic_vision_model",
    "ollama_base_url", "ollama_model", "ollama_vision_model",
    "azure_openai_deployment", "azure_openai_vision_deployment",
    "azure_openai_endpoint", "azure_openai_api_version",
})

# Secret credentials that may be set from the UI. They are handled separately
# from OVERRIDE_WHITELIST: they are write-only (never returned to the client —
# only a "present" boolean is exposed) and stored in the same override table.
SECRET_OVERRIDE_KEYS: frozenset[str] = frozenset({
    "openai_api_key", "anthropic_api_key", "azure_openai_api_key",
    "llama_cloud_api_key",
})


def get_active_settings() -> Settings:
    """Env settings with persisted UI overrides applied on top.

    Rebuilt per call (one cheap SQLite read) so changes made via ``PUT
    /settings`` take effect immediately without a restart. Falls back to the
    cached env settings if the override store is empty or unavailable.
    """
    base = get_settings()
    try:
        from .storage import get_repository

        overrides = get_repository(base.db_path).get_overrides()
    except Exception:  # pragma: no cover - never let the store break a request
        return base
    allowed = OVERRIDE_WHITELIST | SECRET_OVERRIDE_KEYS
    filtered = {
        k: v for k, v in overrides.items()
        if k in allowed and v is not None
    }
    if not filtered:
        return base
    merged = base.model_dump()
    merged.update(filtered)
    return Settings(**merged)
