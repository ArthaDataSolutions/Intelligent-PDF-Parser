"""Interactive, grounded Q&A over a parsed document.

Where :mod:`app.qa.generator` *pre-generates* a batch of analyst questions,
this module answers a user's *own* questions about a specific parsed document.

Grounding strategy: the full parsed markdown (with ``<!-- page N -->`` markers)
is supplied as context and the model is instructed to answer ONLY from it and
to cite the page numbers it used. Quarterly reports comfortably fit the context
window, so no vector store is needed — the document is stuffed directly. If the
text exceeds ``max_chars`` it is truncated with a visible marker.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from ..llm.base import LLMMessage, LLMProvider

_SYSTEM = """You are a careful document analyst. You answer questions about ONE
specific document whose full text is provided to you.

Hard rules:
- Answer ONLY using facts present in the provided document text. If the answer
  is not in the document, say exactly: "That is not stated in this document."
  Never invent, estimate, or use outside knowledge.
- Cite the page number(s) you relied on. Page boundaries are marked in the text
  with `<!-- page N -->`.
- If a figure came from transcribed handwriting (marked `[handwritten: ...]`),
  note that it should be verified against the official source.
- Be concise and precise; quote exact figures and units from the document.

Return STRICT JSON only, no prose outside it:
{"answer": string, "source_pages": [int]}"""

_CONTEXT_TEMPLATE = """Document text (page markers `<!-- page N -->` indicate page numbers):

\"\"\"
{doc}
\"\"\""""

_TRUNCATION_NOTE = "\n\n<!-- document truncated for length -->"


@dataclass(frozen=True)
class ChatAnswer:
    answer: str
    source_pages: list[int]


class DocChat:
    """Answers user questions grounded in a single parsed document."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    @staticmethod
    def _extract_json(raw: str) -> dict:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            raw = raw[4:] if raw.lower().startswith("json") else raw
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start == -1 or end <= start:
            # Model ignored the JSON contract — treat the whole reply as prose.
            return {"answer": raw.strip(), "source_pages": []}
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            return {"answer": raw.strip(), "source_pages": []}

    @staticmethod
    def _coerce_pages(value: object) -> list[int]:
        if not isinstance(value, list):
            return []
        pages: list[int] = []
        for v in value:
            try:
                pages.append(int(v))
            except (TypeError, ValueError):
                continue
        return pages

    async def answer(
        self,
        markdown: str,
        question: str,
        *,
        history: list[LLMMessage] | None = None,
        max_chars: int = 60_000,
        max_tokens: int = 1024,
    ) -> ChatAnswer:
        doc = markdown[:max_chars]
        if len(markdown) > max_chars:
            doc += _TRUNCATION_NOTE

        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=_SYSTEM),
            LLMMessage(role="user", content=_CONTEXT_TEMPLATE.format(doc=doc)),
        ]
        if history:
            messages.extend(history)
        messages.append(LLMMessage(role="user", content=question))

        raw = await self._provider.chat(
            messages, temperature=0.1, max_tokens=max_tokens, json_mode=True
        )
        data = self._extract_json(raw)
        answer = str(data.get("answer") or "").strip() or (
            "That is not stated in this document."
        )
        return ChatAnswer(answer=answer, source_pages=self._coerce_pages(
            data.get("source_pages")
        ))
