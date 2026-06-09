"""OpenAI + Azure OpenAI provider.

Both share the same SDK; Azure differs only in client construction and in using
deployment names instead of model names. GPT-5.4 vision guidance (June 2026):
send `detail="original"` for handwriting / low-quality scans.
"""
from __future__ import annotations

import base64

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMMessage, LLMProvider


def _png_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str,
        text_model: str,
        vision_model: str,
        *,
        azure: bool = False,
        azure_endpoint: str | None = None,
        azure_api_version: str | None = None,
    ) -> None:
        from openai import AsyncAzureOpenAI, AsyncOpenAI

        self.text_model = text_model
        self.vision_model = vision_model
        if azure:
            self.name = "azure"
            self._client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint or "",
                api_version=azure_api_version or "2025-04-01-preview",
            )
        else:
            self._client = AsyncOpenAI(api_key=api_key)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20))
    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        # NOTE: current GPT-5.x reasoning models reject `temperature`/`top_p` and
        # require `max_completion_tokens` instead of `max_tokens`. We target those
        # models, so we omit `temperature` and use `max_completion_tokens`.
        # `temperature` stays in the signature only for cross-provider parity.
        kwargs: dict = {
            "model": self.text_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_completion_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20))
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
        content: list[dict] = [{"type": "text", "text": prompt}]
        for png in images_png:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _png_data_url(png), "detail": detail},
                }
            )
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        # See chat(): omit `temperature`, use `max_completion_tokens` for GPT-5.x.
        resp = await self._client.chat.completions.create(
            model=self.vision_model,
            messages=messages,
            max_completion_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    async def aclose(self) -> None:
        await self._client.close()
