"""Docling backend — IBM open-source, self-hosted, strong on tables/layout.

Weak on handwriting (per 2026 benchmarks), so the router uses it for clean
printed pages and escalates handwritten pages to the vision-LLM backend.
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from ..schemas import ContentKind, PageResult
from .base import ParserAdapter


class DoclingParser(ParserAdapter):
    name = "docling"
    handwriting_capable = False

    def __init__(self) -> None:
        self._converter = None

    def available(self) -> bool:
        try:
            import docling  # noqa: F401

            return True
        except Exception:
            return False

    def _get_converter(self):
        if self._converter is None:
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter()
        return self._converter

    def _convert_sync(self, doc_bytes: bytes) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "in.pdf"
            p.write_bytes(doc_bytes)
            result = self._get_converter().convert(str(p))
            return result.document.export_to_markdown()

    async def parse_pages(
        self, doc_bytes: bytes, page_indices: list[int]
    ) -> list[PageResult]:
        # Docling converts whole documents; we run once and attribute the markdown
        # to the first requested page, leaving the rest empty so the router can
        # still escalate specific pages if needed. For per-page fidelity on mixed
        # docs the router prefers page-scoped backends (vision_llm).
        markdown = await asyncio.to_thread(self._convert_sync, doc_bytes)
        results: list[PageResult] = []
        for idx in page_indices:
            results.append(
                PageResult(
                    page_number=idx + 1,
                    markdown=markdown if idx == page_indices[0] else "",
                    backend=self.name,
                    kind=ContentKind.PRINTED,
                    confidence=0.8 if markdown.strip() else 0.0,
                    has_handwriting=False,
                )
            )
        return results
