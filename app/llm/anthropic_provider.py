"""Anthropic (Claude) provider. Strong handwriting/vision performance."""
from __future__ import annotations

import base64
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMMessage, LLMProvider

# Anthropic's GA server-side web-search tool. The API runs the search loop
# itself and returns the final message with text + auto-generated citations
# (no manual tool round-trips needed). Pin the latest tool version we ship with.
WEB_SEARCH_TOOL_TYPE = "web_search_20260209"
# Web search + tool turns can be slow; give them headroom over the default.
_WEB_SEARCH_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class WebCitation:
    """One web-search citation lifted verbatim from a Claude response."""
    url: str
    title: str
    cited_text: str


@dataclass(frozen=True)
class GroundedAnswer:
    """A web-grounded answer: the model's text plus the citations backing it."""
    text: str
    citations: list[WebCitation]


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

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=10))
    async def search_with_citations(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_uses: int = 5,
        max_tokens: int = 1024,
    ) -> GroundedAnswer:
        """Answer `prompt` using Claude's server-side web search and return the
        text together with the real citations Claude attached to it.

        The presence of this method is how the pipeline detects that a provider
        can ground peer-comparison questions; non-Anthropic providers don't have
        it and are skipped. Returns an empty :class:`GroundedAnswer` if the model
        produced no citations (caller treats that as "nothing found").
        """
        resp = await self._client.messages.create(
            model=self.text_model,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            tools=[{
                "type": WEB_SEARCH_TOOL_TYPE,
                "name": "web_search",
                "max_uses": max_uses,
            }],
            timeout=_WEB_SEARCH_TIMEOUT_S,
        )
        return self._collect_grounding(resp)

    @staticmethod
    def _collect_grounding(resp) -> GroundedAnswer:
        """Pull text + web-search citations out of a tool-enabled response."""
        text_parts: list[str] = []
        citations: list[WebCitation] = []
        seen: set[str] = set()
        for block in resp.content:
            if getattr(block, "type", None) != "text":
                continue
            text_parts.append(block.text)
            for c in (getattr(block, "citations", None) or []):
                if getattr(c, "type", None) != "web_search_result_location":
                    continue
                url = getattr(c, "url", "") or ""
                key = f"{url}|{getattr(c, 'cited_text', '')[:80]}"
                if not url or key in seen:
                    continue
                seen.add(key)
                citations.append(WebCitation(
                    url=url,
                    title=getattr(c, "title", None) or "",
                    cited_text=getattr(c, "cited_text", "") or "",
                ))
        return GroundedAnswer(text="".join(text_parts).strip(), citations=citations)

    async def aclose(self) -> None:
        await self._client.close()
