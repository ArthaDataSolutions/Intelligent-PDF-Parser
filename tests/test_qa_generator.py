import json

from app.qa.generator import QAGenerator
from app.schemas import PageResult, ParseResult
from tests.conftest import FakeLLM


def _parse_result() -> ParseResult:
    return ParseResult(
        source_filename="q3.pdf",
        page_count=1,
        pages=[PageResult(page_number=1, markdown="Revenue $1,284.6M", backend="vision_llm")],
    )


async def test_generate_parses_items_and_filters_bad_ones():
    payload = {
        "company": "Northwind Industries",
        "period": "Q3 FY2026",
        "items": [
            {
                "question": "What drove the revenue beat?",
                "answer": "Revenue was $1,284.6M, up 11.4% YoY.",
                "category": "segments",
                "confidence": "high",
                "source_pages": [1],
                "caveat": None,
            },
            {"question": "broken item missing answer"},  # invalid -> dropped
        ],
    }
    gen = QAGenerator(FakeLLM(chat_reply=json.dumps(payload)))
    doc = await gen.generate(_parse_result(), num_questions=2)
    assert doc.company == "Northwind Industries"
    assert doc.period == "Q3 FY2026"
    assert len(doc.items) == 1
    assert doc.items[0].confidence == "high"


async def test_generate_handles_fenced_json():
    payload = {"company": None, "period": None, "items": []}
    fenced = "```json\n" + json.dumps(payload) + "\n```"
    gen = QAGenerator(FakeLLM(chat_reply=fenced))
    doc = await gen.generate(_parse_result())
    assert doc.items == []


def test_to_markdown_renders_sections_and_disclaimer():
    from app.schemas import QADocument, QAItem

    doc = QADocument(
        company="Acme",
        period="Q1",
        generated_from="q1.pdf",
        items=[
            QAItem(
                question="Guidance?",
                answer="Raised to high end.",
                category="guidance",
                source_pages=[2],
                caveat="From handwritten note; verify.",
            )
        ],
    )
    md = QAGenerator.to_markdown(doc)
    assert "# Analyst Q&A" in md
    assert "Guidance" in md
    assert "source page(s): 2" in md
    assert "verify" in md
    assert "verified against the official filing" in md
