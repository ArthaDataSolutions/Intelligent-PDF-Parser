from app.config import Settings
from app.parsers.vision_llm_parser import VisionLLMParser, _parse_meta
from app.schemas import ContentKind
from tests.conftest import FakeLLM


def test_parse_meta_splits_markdown_and_json():
    raw = (
        "# Heading\n[handwritten: pay $5]\n===META===\n"
        '{"has_handwriting": true, "confidence": 0.91, "illegible_count": 1}'
    )
    md, meta = _parse_meta(raw)
    assert "Heading" in md and "===META===" not in md
    assert meta["has_handwriting"] is True
    assert meta["confidence"] == 0.91


def test_parse_meta_missing_marker():
    md, meta = _parse_meta("just markdown, no meta")
    assert md == "just markdown, no meta"
    assert meta == {}


async def test_vision_parser_uses_provider(scanned_pdf):
    reply = (
        "Revenue **$1,284.6M**\n[handwritten: refinance Q1?]\n===META===\n"
        '{"has_handwriting": true, "confidence": 0.88, "illegible_count": 0}'
    )
    fake = FakeLLM(vision_reply=reply)
    p = VisionLLMParser(provider=fake, settings=Settings())
    assert p.available() is True
    results = await p.parse_pages(scanned_pdf, [0])
    assert fake.vision_calls == 1
    pr = results[0]
    assert pr.backend == "vision_llm"
    assert pr.has_handwriting is True
    assert pr.kind == ContentKind.HANDWRITTEN
    assert "1,284.6M" in pr.markdown
    assert 0.0 <= pr.confidence <= 1.0
