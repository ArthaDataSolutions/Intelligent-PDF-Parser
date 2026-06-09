"""Parser router — the brain of the parsing stage.

Strategy (`PARSER_BACKEND=auto`):
  1. Analyze every page for routing signals (text coverage, image coverage).
  2. Pages that look handwritten/scanned go straight to the vision-LLM backend.
  3. Remaining (clean printed) pages go to the best available cheap backend
     (Docling self-hosted, else LlamaParse).
  4. Any page whose cheap-backend confidence falls below the threshold is
     re-parsed by the vision-LLM backend (fallback escalation).
  5. `FORCE_VISION_LLM=true` sends every page through the vision-LLM backend.

If a specific backend is named (`docling|llamaparse|vision_llm`) the router uses
only that one, with no escalation.
"""
from __future__ import annotations

import time

from ..config import Settings, get_settings
from ..logging_conf import get_logger
from ..schemas import ContentKind, PageResult, ParseResult
from ..utils.pdf import PageSignal, analyze_pages, extract_native_text, page_count
from .base import ParserAdapter
from .docling_parser import DoclingParser
from .llamaparse_parser import LlamaParseParser
from .vision_llm_parser import VisionLLMParser

log = get_logger("parser.router")


class ParserRouter:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        docling: ParserAdapter | None = None,
        llamaparse: ParserAdapter | None = None,
        vision_llm: ParserAdapter | None = None,
    ) -> None:
        self.s = settings or get_settings()
        self.docling = docling or DoclingParser()
        self.llamaparse = llamaparse or LlamaParseParser()
        self.vision_llm = vision_llm or VisionLLMParser(settings=self.s)

    # ---- backend selection helpers ---------------------------------------
    def _cheap_printed_backend(self) -> ParserAdapter | None:
        if self.docling.available():
            return self.docling
        if self.llamaparse.available():
            return self.llamaparse
        return None

    def _named_backend(self, name: str) -> ParserAdapter:
        return {
            "docling": self.docling,
            "llamaparse": self.llamaparse,
            "vision_llm": self.vision_llm,
        }[name]

    # ---- main entry -------------------------------------------------------
    async def parse(self, doc_bytes: bytes, filename: str) -> ParseResult:
        t0 = time.perf_counter()
        n = page_count(doc_bytes)
        all_indices = list(range(n))
        warnings: list[str] = []
        log.debug(
            "parse_plan",
            filename=filename,
            pages=n,
            parser_backend=self.s.parser_backend,
            force_vision_llm=self.s.force_vision_llm,
            confidence_threshold=self.s.parser_confidence_threshold,
            scan_char_threshold=self.s.handwriting_scan_char_threshold,
            image_area_threshold=self.s.handwriting_image_area_threshold,
            text_area_threshold=self.s.handwriting_text_area_threshold,
            vision_image_detail=self.s.vision_image_detail,
            vision_render_dpi=self.s.vision_render_dpi,
        )

        if self.s.parser_backend != "auto":
            backend = self._named_backend(self.s.parser_backend)
            if not backend.available():
                raise RuntimeError(
                    f"Requested backend '{self.s.parser_backend}' is not available "
                    "(missing dependency or API key)."
                )
            log.info("routing", total=n, backend=self.s.parser_backend)
            log.debug(
                "route_pages",
                target_backend=self.s.parser_backend,
                pages=[i + 1 for i in all_indices],
                reason="named_backend_forced",
            )
            pages = await backend.parse_pages(doc_bytes, all_indices)
            return self._finalize(filename, pages, warnings, t0)

        # ---- auto routing ----
        signals = analyze_pages(
            doc_bytes,
            scan_char_threshold=self.s.handwriting_scan_char_threshold,
            image_area_threshold=self.s.handwriting_image_area_threshold,
            text_area_threshold=self.s.handwriting_text_area_threshold,
        )
        vision_ok = self.vision_llm.available()
        cheap = self._cheap_printed_backend()

        if self.s.force_vision_llm:
            if not vision_ok:
                raise RuntimeError("FORCE_VISION_LLM set but no vision provider configured.")
            log.info("routing", total=n, mode="force_vision")
            log.debug(
                "route_pages",
                target_backend="vision_llm",
                pages=[i + 1 for i in all_indices],
                reason="force_vision_llm",
            )
            pages = await self.vision_llm.parse_pages(doc_bytes, all_indices)
            return self._finalize(filename, pages, warnings, t0)

        handwritten_idx = [s.page_number - 1 for s in signals if s.likely_handwritten]
        printed_idx = [s.page_number - 1 for s in signals if not s.likely_handwritten]
        for sig in signals:
            log.debug(
                "page_analyzed",
                page=sig.page_number,
                char_count=sig.char_count,
                text_area_ratio=sig.text_area_ratio,
                image_area_ratio=sig.image_area_ratio,
                likely_scanned=sig.likely_scanned,
                likely_handwritten=sig.likely_handwritten,
                route_hint="vision_llm" if sig.likely_handwritten else "cheap_printed",
                reason=self._signal_reason(sig),
            )
        log.info(
            "routing", total=n, handwritten_pages=len(handwritten_idx),
            printed_pages=len(printed_idx), vision_available=vision_ok,
            cheap_backend=(cheap.name if cheap else None),
            scan_char_threshold=self.s.handwriting_scan_char_threshold,
            image_area_threshold=self.s.handwriting_image_area_threshold,
            text_area_threshold=self.s.handwriting_text_area_threshold,
        )

        results: dict[int, PageResult] = {}

        # 1) handwritten/scanned pages -> vision LLM
        if handwritten_idx:
            if vision_ok:
                log.debug(
                    "route_pages",
                    target_backend="vision_llm",
                    pages=[i + 1 for i in handwritten_idx],
                    reason="likely_handwritten_or_scanned",
                )
                for pr in await self.vision_llm.parse_pages(doc_bytes, handwritten_idx):
                    results[pr.page_number - 1] = pr
            else:
                warnings.append(
                    f"{len(handwritten_idx)} page(s) look handwritten but no vision "
                    "provider is configured; used native text extraction (low quality)."
                )
                for idx in handwritten_idx:
                    log.debug(
                        "route_pages",
                        target_backend="native",
                        pages=[idx + 1],
                        reason="likely_handwritten_but_vision_unavailable",
                    )
                    results[idx] = self._native_fallback(doc_bytes, idx)

        # 2) printed pages -> cheap backend (or vision if none / native fallback)
        if printed_idx:
            if cheap is not None:
                log.debug(
                    "route_pages",
                    target_backend=cheap.name,
                    pages=[i + 1 for i in printed_idx],
                    reason="printed_text_layer_available",
                )
                for pr in await cheap.parse_pages(doc_bytes, printed_idx):
                    results[pr.page_number - 1] = pr
            elif vision_ok:
                log.debug(
                    "route_pages",
                    target_backend="vision_llm",
                    pages=[i + 1 for i in printed_idx],
                    reason="no_cheap_backend_available",
                )
                for pr in await self.vision_llm.parse_pages(doc_bytes, printed_idx):
                    results[pr.page_number - 1] = pr
            else:
                warnings.append("No parser backend configured; used native text extraction.")
                for idx in printed_idx:
                    log.debug(
                        "route_pages",
                        target_backend="native",
                        pages=[idx + 1],
                        reason="no_parser_backend_available",
                    )
                    results[idx] = self._native_fallback(doc_bytes, idx)

        # 3) confidence-based escalation to vision LLM
        if vision_ok:
            low = [
                idx
                for idx, pr in results.items()
                if pr.backend in {"docling", "llamaparse", "native"}
                and pr.confidence < self.s.parser_confidence_threshold
            ]
            if low:
                log.debug(
                    "low_confidence_pages",
                    pages=[
                        {
                            "page": idx + 1,
                            "backend": results[idx].backend,
                            "confidence": round(results[idx].confidence, 3),
                        }
                        for idx in low
                    ],
                    threshold=self.s.parser_confidence_threshold,
                )
                log.info("escalating_low_confidence_pages", count=len(low))
                for pr in await self.vision_llm.parse_pages(doc_bytes, low):
                    pr.notes.append("escalated from low-confidence cheap backend")
                    results[pr.page_number - 1] = pr

        pages = [results[i] for i in sorted(results)]
        return self._finalize(filename, pages, warnings, t0)

    # ---- helpers ----------------------------------------------------------
    def _native_fallback(self, doc_bytes: bytes, idx: int) -> PageResult:
        text = extract_native_text(doc_bytes, idx)
        log.debug(
            "native_text_extracted",
            page=idx + 1,
            char_count=len(text.strip()),
            confidence=0.3 if text.strip() else 0.0,
        )
        return PageResult(
            page_number=idx + 1,
            markdown=text,
            backend="native",
            kind=ContentKind.PRINTED,
            confidence=0.3 if text.strip() else 0.0,
        )

    def _finalize(
        self,
        filename: str,
        pages: list[PageResult],
        warnings: list[str],
        t0: float,
    ) -> ParseResult:
        summary: dict[str, int] = {}
        for p in pages:
            summary[p.backend] = summary.get(p.backend, 0) + 1
        log.info(
            "parse_complete",
            filename=filename,
            pages=len(pages),
            backends=summary,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        return ParseResult(
            source_filename=filename,
            page_count=len(pages),
            backend_summary=summary,
            pages=pages,
            warnings=warnings,
        )

    def _signal_reason(self, sig: PageSignal) -> str:
        if sig.likely_scanned:
            return (
                "char_count below handwriting_scan_char_threshold "
                f"({sig.char_count} < {self.s.handwriting_scan_char_threshold})"
            )
        if sig.likely_handwritten:
            return (
                "image_area_ratio above handwriting_image_area_threshold and "
                "text_area_ratio below handwriting_text_area_threshold "
                f"({sig.image_area_ratio} > {self.s.handwriting_image_area_threshold}, "
                f"{sig.text_area_ratio} < {self.s.handwriting_text_area_threshold})"
            )
        return "extractable text is sufficient for the printed-page path"
