"""Ollama provider — self-hosted, no API key. Uses the OpenAI-compatible REST API.

Vision requires a multimodal model (e.g. llama3.2-vision, qwen2.5-vl).
"""
from __future__ import annotations

import base64

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMMessage, LLMProvider


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: str, text_model: str, vision_model: str) -> None:
        self._base = base_url.rstrip("/")
        self.text_model = text_model
        self.vision_model = vision_model
        # trust_env=False: Ollama is a local/self-hosted service — never route it
        # through an ambient HTTP(S)/SOCKS proxy from the environment.
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0), trust_env=False
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=15))
    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        payload: dict = {
            "model": self.text_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        r = await self._client.post(f"{self._base}/api/chat", json=payload)
        r.raise_for_status()
        return r.json()["message"]["content"]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=15))
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
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append(
            {
                "role": "user",
                "content": prompt,
                "images": [base64.b64encode(p).decode("ascii") for p in images_png],
            }
        )
        r = await self._client.post(
            f"{self._base}/api/chat",
            json={
                "model": self.vision_model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    async def aclose(self) -> None:
        await self._client.aclose()
