"""LlamaParse backend — API-only, cleanest markdown for complex printed layouts.

Good cost/quality default ($0.003/page class). Handwriting support is partial,
so the router treats it like Docling: great for printed pages, not the primary
handwriting reader.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from ..config import get_settings
from ..schemas import ContentKind, PageResult
from .base import ParserAdapter


class LlamaParseParser(ParserAdapter):
    name = "llamaparse"
    handwriting_capable = False

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or get_settings().llama_cloud_api_key

    def available(self) -> bool:
        if not self._api_key:
            return False
        try:
            import llama_cloud_services  # noqa: F401

            return True
        except Exception:
            return False

    async def parse_pages(
        self, doc_bytes: bytes, page_indices: list[int]
    ) -> list[PageResult]:
        from llama_cloud_services import LlamaParse

        parser = LlamaParse(api_key=self._api_key, result_type="markdown")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "in.pdf"
            p.write_bytes(doc_bytes)
            # aload_data returns one document per page for PDFs
            docs = await parser.aload_data(str(p))

        by_page: dict[int, str] = {}
        for i, d in enumerate(docs):
            pg = int((d.metadata or {}).get("page_number", i + 1))
            by_page[pg - 1] = getattr(d, "text", "") or ""

        results: list[PageResult] = []
        for idx in page_indices:
            md = by_page.get(idx, "")
            results.append(
                PageResult(
                    page_number=idx + 1,
                    markdown=md,
                    backend=self.name,
                    kind=ContentKind.PRINTED,
                    confidence=0.78 if md.strip() else 0.0,
                    has_handwriting=False,
                )
            )
        return results
