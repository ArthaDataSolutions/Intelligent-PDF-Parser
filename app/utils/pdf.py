"""PDF helpers: page rendering and a cheap handwriting/low-text heuristic.

The heuristic is intentionally simple and dependency-light. It is NOT the final
word on whether a page has handwriting — it is a *router signal*. Pages it flags
(little/no extractable text, or low text-coverage relative to ink on the page)
get escalated to the vision-LLM backend, which is where real handwriting reading
happens.
"""
from __future__ import annotations

from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass
class PageSignal:
    page_number: int
    char_count: int
    text_area_ratio: float   # area covered by extractable text spans / page area
    image_area_ratio: float  # area covered by raster images / page area
    likely_handwritten: bool
    likely_scanned: bool


def render_page_png(doc_bytes: bytes, page_index: int, dpi: int = 200) -> bytes:
    """Render a single page to PNG bytes at the given DPI."""
    with fitz.open(stream=doc_bytes, filetype="pdf") as doc:
        page = doc[page_index]
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")


def page_count(doc_bytes: bytes) -> int:
    with fitz.open(stream=doc_bytes, filetype="pdf") as doc:
        return doc.page_count


def extract_native_text(doc_bytes: bytes, page_index: int) -> str:
    with fitz.open(stream=doc_bytes, filetype="pdf") as doc:
        return doc[page_index].get_text("text")


def analyze_pages(doc_bytes: bytes) -> list[PageSignal]:
    """Produce routing signals for every page."""
    signals: list[PageSignal] = []
    with fitz.open(stream=doc_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            rect = page.rect
            page_area = max(rect.width * rect.height, 1.0)
            text = page.get_text("text")
            char_count = len(text.strip())

            # text coverage (block_type 0 == text)
            text_area = 0.0
            for blk in page.get_text("blocks"):
                if len(blk) >= 7 and blk[6] != 0:
                    continue
                x0, y0, x1, y1 = blk[:4]
                text_area += max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
            text_ratio = min(text_area / page_area, 1.0)

            # image coverage
            img_area = 0.0
            for img in page.get_images(full=True):
                for r in page.get_image_rects(img[0]):
                    img_area += r.width * r.height
            img_ratio = min(img_area / page_area, 1.0)

            # An empty/near-empty text layer means the page is a scan — its real
            # content (often handwriting) lives in pixels the text extractor can't
            # see. A meaningful embedded raster alongside sparse text is the
            # classic "printed report + handwritten annotation" case. Both should
            # be read by the vision-LLM backend.
            likely_scanned = char_count < 20
            likely_handwritten = likely_scanned or (
                img_ratio > 0.15 and text_ratio < 0.35
            )

            signals.append(
                PageSignal(
                    page_number=i + 1,
                    char_count=char_count,
                    text_area_ratio=round(text_ratio, 4),
                    image_area_ratio=round(img_ratio, 4),
                    likely_handwritten=likely_handwritten,
                    likely_scanned=likely_scanned,
                )
            )
    return signals
