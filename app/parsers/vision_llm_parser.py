"""Vision-LLM backend — the handwriting reader.

Renders each page to an image and asks the configured vision provider
(OpenAI / Anthropic / Azure / Ollama) to transcribe it faithfully to markdown.
This is where the Snowflake "handwritten notes are inaccurate" problem is
actually solved: a frontier multimodal model reading the page image directly,
with explicit instructions to flag uncertainty rather than guess.
"""
from __future__ import annotations

import asyncio
import json
import time

from ..config import Settings, get_settings
from ..llm.base import LLMProvider
from ..llm.factory import build_vision_provider
from ..logging_conf import get_logger
from ..schemas import ContentKind, PageResult
from ..utils.pdf import render_page_png
from .base import ParserAdapter

log = get_logger("parser.vision")

_META_MARKER = "===META==="

_SYSTEM = (
    "You are a meticulous document transcription engine for financial reports. "
    "You read printed text, tables, AND handwritten notes/annotations from page "
    "images and reproduce them as clean GitHub-flavored Markdown."
)

_PROMPT = f"""Transcribe this page to Markdown. Rules:
- Reproduce ALL content: printed text, tables (as Markdown tables), and any
  handwritten notes or margin annotations.
- Wrap transcribed handwriting in `[handwritten: ...]` so it is traceable.
- If a word/figure is illegible, write `[illegible]` — never guess at numbers.
- Preserve numeric values, units, and signs exactly. Do not "correct" figures.
- Do not add commentary or content that is not on the page.

After the Markdown, append the marker `{_META_MARKER}` on its own line, then a
single line of strict JSON (no code fence):
{{"has_handwriting": bool, "confidence": 0.0-1.0, "illegible_count": int}}
where confidence is your honest self-assessment of transcription fidelity."""


def _parse_meta(raw: str) -> tuple[str, dict]:
    if _META_MARKER in raw:
        md, _, meta_str = raw.rpartition(_META_MARKER)
    else:
        md, meta_str = raw, ""
    meta: dict = {}
    meta_str = meta_str.strip()
    if meta_str:
        try:
            start = meta_str.index("{")
            end = meta_str.rindex("}") + 1
            meta = json.loads(meta_str[start:end])
        except (ValueError, json.JSONDecodeError):
            meta = {}
    return md.strip(), meta


class VisionLLMParser(ParserAdapter):
    name = "vision_llm"
    handwriting_capable = True

    def __init__(
        self,
        provider: LLMProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._provider = provider  # injectable for tests; built lazily otherwise

    def available(self) -> bool:
        if self._provider is not None:
            return True
        try:
            build_vision_provider(self._settings)
            return True
        except Exception:
            return False

    def _get_provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = build_vision_provider(self._settings)
        return self._provider

    async def _parse_one(self, doc_bytes: bytes, idx: int) -> PageResult:
        render_t0 = time.perf_counter()
        png = await asyncio.to_thread(
            render_page_png, doc_bytes, idx, self._settings.vision_render_dpi
        )
        render_ms = round((time.perf_counter() - render_t0) * 1000, 1)
        provider = self._get_provider()
        log.debug(
            "vision_page_rendered",
            page=idx + 1,
            dpi=self._settings.vision_render_dpi,
            image_bytes=len(png),
            render_ms=render_ms,
            detail=self._settings.vision_image_detail,
            provider=provider.name,
            model=getattr(provider, "vision_model", ""),
        )
        raw = await provider.vision(
            _PROMPT,
            [png],
            system=_SYSTEM,
            detail=self._settings.vision_image_detail,
            temperature=0.0,
        )
        md, meta = _parse_meta(raw)
        has_hw = bool(meta.get("has_handwriting", False))
        confidence = float(meta.get("confidence", 0.7))
        illegible = int(meta.get("illegible_count", 0))
        log.debug(
            "vision_page_meta",
            page=idx + 1,
            has_handwriting=has_hw,
            confidence=round(confidence, 3),
            illegible_count=illegible,
            markdown_chars=len(md),
            meta_found=bool(meta),
        )
        notes = []
        if illegible:
            notes.append(f"{illegible} illegible token(s) flagged by model")
        return PageResult(
            page_number=idx + 1,
            markdown=md,
            backend=self.name,
            kind=ContentKind.HANDWRITTEN if has_hw else ContentKind.PRINTED,
            confidence=max(0.0, min(confidence, 1.0)),
            has_handwriting=has_hw,
            notes=notes,
        )

    async def parse_pages(
        self, doc_bytes: bytes, page_indices: list[int]
    ) -> list[PageResult]:
        # Bounded concurrency to avoid hammering provider rate limits.
        sem = asyncio.Semaphore(self._settings.vision_max_concurrency)
        total = len(page_indices)
        done = [0]
        log.info(
            "vision_transcribe_start",
            pages=total,
            concurrency=self._settings.vision_max_concurrency,
            dpi=self._settings.vision_render_dpi,
            detail=self._settings.vision_image_detail,
        )

        async def _guarded(idx: int) -> PageResult:
            async with sem:
                pr = await self._parse_one(doc_bytes, idx)
                done[0] += 1
                log.info(
                    "page_transcribed", page=idx + 1, done=done[0], total=total,
                    confidence=round(pr.confidence, 2),
                    handwriting=pr.has_handwriting,
                )
                return pr

        return await asyncio.gather(*(_guarded(i) for i in page_indices))
