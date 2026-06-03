"""Shared test fixtures: fake LLM provider, fake parser adapters, tiny PDFs.

All tests run fully offline — no network, no real API keys.
"""
from __future__ import annotations

import io

import fitz
import pytest
from PIL import Image

from app.llm.base import LLMMessage, LLMProvider
from app.logging_conf import configure_logging
from app.parsers.base import ParserAdapter
from app.schemas import ContentKind, PageResult

# Install the structlog config (incl. the per-run DB log sink) once for the
# whole test session, independent of whether app.main has been imported. The
# real service does this at startup; this keeps tests order-independent.
configure_logging("INFO")


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Point every test's run store at a throwaway SQLite file.

    Clears the settings cache so the new ``DB_PATH`` is picked up, both before
    and after, keeping the real ./data store untouched during tests.
    """
    from app.config import get_settings

    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class FakeLLM(LLMProvider):
    """Records calls and returns canned responses."""

    def __init__(self, chat_reply: str = "{}", vision_reply: str = "") -> None:
        self.name = "fake"
        self._chat_reply = chat_reply
        self._vision_reply = vision_reply
        self.chat_calls: list[list[LLMMessage]] = []
        self.vision_calls = 0

    async def chat(self, messages, *, temperature=0.2, max_tokens=4096, json_mode=False):
        self.chat_calls.append(messages)
        return self._chat_reply

    async def vision(self, prompt, images_png, *, system=None, detail="auto",
                     temperature=0.0, max_tokens=4096):
        self.vision_calls += 1
        return self._vision_reply


class FakeAdapter(ParserAdapter):
    def __init__(self, name, *, available=True, confidence=0.9,
                 handwriting_capable=False):
        self.name = name
        self.handwriting_capable = handwriting_capable
        self._available = available
        self._confidence = confidence
        self.parsed_indices: list[int] = []

    def available(self) -> bool:
        return self._available

    async def parse_pages(self, doc_bytes, page_indices):
        self.parsed_indices.extend(page_indices)
        kind = ContentKind.HANDWRITTEN if self.handwriting_capable else ContentKind.PRINTED
        return [
            PageResult(
                page_number=i + 1,
                markdown=f"{self.name} page {i + 1}",
                backend=self.name,
                kind=kind,
                confidence=self._confidence,
                has_handwriting=self.handwriting_capable,
            )
            for i in page_indices
        ]


@pytest.fixture
def text_pdf() -> bytes:
    """A 2-page printed text PDF (rich text layer)."""
    doc = fitz.open()
    for n in range(2):
        page = doc.new_page()
        for line in range(20):
            page.insert_text((50, 60 + line * 18), f"Printed line {line} on page {n}.")
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def scanned_pdf() -> bytes:
    """A 1-page image-only PDF (empty text layer) — should route to vision."""
    img = Image.new("RGB", (1000, 1300), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_image(fitz.Rect(20, 20, 575, 740), stream=buf.getvalue())
    data = doc.tobytes()
    doc.close()
    return data
