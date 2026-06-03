"""Provider-agnostic LLM interface.

Every provider implements two calls:
  * `chat`   — text in, text out (used by the Q&A generator)
  * `vision` — text + page images in, text out (used by the vision-LLM parser)

Keeping the surface this small lets us swap OpenAI / Anthropic / Azure / Ollama
without touching the pipeline.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMProvider(abc.ABC):
    name: str = "base"
    text_model: str = ""
    vision_model: str = ""

    @abc.abstractmethod
    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        """Return assistant text for a text-only conversation."""

    @abc.abstractmethod
    async def vision(
        self,
        prompt: str,
        images_png: list[bytes],
        *,
        system: str | None = None,
        detail: str = "auto",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Return assistant text given a prompt and one or more PNG page images."""

    async def aclose(self) -> None:  # pragma: no cover - optional cleanup hook
        return None
