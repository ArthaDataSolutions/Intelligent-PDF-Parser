"""Anthropic (Claude) provider. Strong handwriting/vision performance."""
from __future__ import annotations

import base64

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMMessage, LLMProvider


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str, text_model: str, vision_model: str) -> None:
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key)
        self.text_model = text_model
        self.vision_model = vision_model

    @staticmethod
    def _split_system(messages: list[LLMMessage]) -> tuple[str | None, list[dict]]:
        system = None
        convo: list[dict] = []
        for m in messages:
            if m.role == "system":
                system = (system + "\n\n" + m.content) if system else m.content
            else:
                convo.append({"role": m.role, "content": m.content})
        return system, convo

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20))
    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        # NOTE: current Claude models (opus 4.7+/sonnet 4.6+) reject `temperature`
        # and `top_p` ("`temperature` is deprecated for this model"). We rely on
        # the model's default sampling and never forward those knobs. `temperature`
        # is kept in the signature only for cross-provider parity (Ollama uses it).
        system, convo = self._split_system(messages)
        if json_mode:
            system = (system or "") + "\n\nRespond with a single valid JSON object only."
        resp = await self._client.messages.create(
            model=self.text_model,
            system=system or "",
            messages=convo,
            max_tokens=max_tokens,
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20))
    async def vision(
        self,
        prompt: str,
        images_png: list[bytes],
        *,
        system: str | None = None,
        detail: str = "auto",  # Claude has no detail knob; accepted for parity
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        blocks: list[dict] = []
        for png in images_png:
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(png).decode("ascii"),
                    },
                }
            )
        blocks.append({"type": "text", "text": prompt})
        # See chat(): `temperature` is not forwarded — current Claude models reject it.
        resp = await self._client.messages.create(
            model=self.vision_model,
            system=system or "",
            messages=[{"role": "user", "content": blocks}],
            max_tokens=max_tokens,
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    async def aclose(self) -> None:
        await self._client.close()
