import pytest

from app.config import Settings
from app.parsers.router import ParserRouter
from tests.conftest import FakeAdapter


def _router(settings, **adapters):
    return ParserRouter(settings, **adapters)


async def test_named_backend_used_directly(text_pdf):
    s = Settings(parser_backend="docling")
    doc = FakeAdapter("docling")
    r = _router(s, docling=doc, llamaparse=FakeAdapter("llamaparse", available=False),
                vision_llm=FakeAdapter("vision_llm", available=False, handwriting_capable=True))
    res = await r.parse(text_pdf, "x.pdf")
    assert res.backend_summary == {"docling": 2}
    assert doc.parsed_indices == [0, 1]


async def test_named_backend_unavailable_raises(text_pdf):
    s = Settings(parser_backend="llamaparse")
    r = _router(s, llamaparse=FakeAdapter("llamaparse", available=False))
    with pytest.raises(RuntimeError):
        await r.parse(text_pdf, "x.pdf")


async def test_auto_routes_scanned_to_vision(scanned_pdf):
    s = Settings(parser_backend="auto")
    vision = FakeAdapter("vision_llm", handwriting_capable=True, confidence=0.95)
    cheap = FakeAdapter("docling", confidence=0.9)
    r = _router(s, docling=cheap, llamaparse=FakeAdapter("llamaparse", available=False),
                vision_llm=vision)
    res = await r.parse(scanned_pdf, "scan.pdf")
    assert res.backend_summary == {"vision_llm": 1}
    assert cheap.parsed_indices == []


async def test_auto_routes_printed_to_cheap(text_pdf):
    s = Settings(parser_backend="auto")
    cheap = FakeAdapter("docling", confidence=0.9)
    vision = FakeAdapter("vision_llm", handwriting_capable=True)
    r = _router(s, docling=cheap, llamaparse=FakeAdapter("llamaparse", available=False),
                vision_llm=vision)
    res = await r.parse(text_pdf, "x.pdf")
    assert res.backend_summary == {"docling": 2}
    assert vision.parsed_indices == []


async def test_low_confidence_escalates_to_vision(text_pdf):
    s = Settings(parser_backend="auto", parser_confidence_threshold=0.8)
    cheap = FakeAdapter("docling", confidence=0.4)   # below threshold
    vision = FakeAdapter("vision_llm", handwriting_capable=True, confidence=0.95)
    r = _router(s, docling=cheap, llamaparse=FakeAdapter("llamaparse", available=False),
                vision_llm=vision)
    res = await r.parse(text_pdf, "x.pdf")
    # both printed pages get escalated -> end up attributed to vision_llm
    assert res.backend_summary == {"vision_llm": 2}
    assert sorted(vision.parsed_indices) == [0, 1]


async def test_force_vision_llm(text_pdf):
    s = Settings(parser_backend="auto", force_vision_llm=True)
    vision = FakeAdapter("vision_llm", handwriting_capable=True, confidence=0.9)
    cheap = FakeAdapter("docling")
    r = _router(s, docling=cheap, vision_llm=vision)
    res = await r.parse(text_pdf, "x.pdf")
    assert res.backend_summary == {"vision_llm": 2}
    assert cheap.parsed_indices == []


async def test_no_backend_falls_back_to_native(text_pdf):
    s = Settings(parser_backend="auto")
    r = _router(
        s,
        docling=FakeAdapter("docling", available=False),
        llamaparse=FakeAdapter("llamaparse", available=False),
        vision_llm=FakeAdapter("vision_llm", available=False),
    )
    res = await r.parse(text_pdf, "x.pdf")
    assert "native" in res.backend_summary
    assert res.warnings  # warns that no backend was configured
