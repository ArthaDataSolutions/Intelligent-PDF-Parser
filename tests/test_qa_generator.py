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
        "company": "Northwind Pharma",
        "period": "Q3 FY2026",
        "items": [
            {
                "question": "What drove the revenue beat?",
                "answer": "Revenue was $1,284.6M, up 11.4% YoY.",
                "asker": "analyst",
                "reasoning": "Top-line beat on p.1 invites a question on durability.",
                "peer_context": "Peers like Merck grew low-double-digits this quarter.",
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
    assert doc.company == "Northwind Pharma"
    assert doc.period == "Q3 FY2026"
    assert len(doc.items) == 1
    item = doc.items[0]
    assert item.confidence == "high"
    assert item.asker == "analyst"
    assert item.reasoning
    assert item.peer_context


async def test_item_defaults_when_new_fields_absent():
    """Older-shaped items (no asker/reasoning) still validate with defaults."""
    payload = {
        "company": None,
        "period": None,
        "items": [
            {"question": "Guidance?", "answer": "Raised.", "source_pages": [2]}
        ],
    }
    gen = QAGenerator(FakeLLM(chat_reply=json.dumps(payload)))
    doc = await gen.generate(_parse_result())
    assert len(doc.items) == 1
    assert doc.items[0].asker == "analyst"
    assert doc.items[0].reasoning == ""
    assert doc.items[0].peer_context is None


async def test_generate_passes_peer_hint_to_provider():
    fake = FakeLLM(chat_reply=json.dumps({"company": None, "period": None, "items": []}))
    gen = QAGenerator(fake)
    await gen.generate(_parse_result(), peers=["Pfizer", "Merck"])
    user_msg = fake.chat_calls[0][-1].content
    assert "Pfizer" in user_msg and "Merck" in user_msg


async def test_generate_salvages_truncated_json():
    """A response cut off by the token limit keeps its complete items."""
    good = {
        "question": "What drove the revenue beat?",
        "answer": "Revenue was $1,284.6M.",
        "asker": "analyst",
        "reasoning": "Top-line beat on p.1.",
        "peer_context": None,
        "category": "segments",
        "confidence": "high",
        "source_pages": [1],
        "caveat": None,
    }
    # Two complete items, then a third object truncated mid-string (no closing
    # brace / bracket) — exactly what an output-token cutoff produces.
    head = '{"company": "Northwind Pharma", "period": "Q3 FY2026", "items": ['
    truncated = head + json.dumps(good) + ", " + json.dumps(good) + ', {"question": "What about gui'
    gen = QAGenerator(FakeLLM(chat_reply=truncated))
    doc = await gen.generate(_parse_result(), num_questions=20)
    assert doc.company == "Northwind Pharma"
    assert doc.period == "Q3 FY2026"
    assert len(doc.items) == 2  # the two whole items survive; the partial is dropped


async def test_generate_scales_token_budget_with_question_count():
    fake = FakeLLM(chat_reply=json.dumps({"company": None, "period": None, "items": []}))
    gen = QAGenerator(fake)
    await gen.generate(_parse_result(), num_questions=20)
    # Budget must scale past the old hardcoded 4096 so 20 items don't truncate.
    assert fake.last_max_tokens >= 20 * 600


async def test_generate_handles_fenced_json():
    payload = {"company": None, "period": None, "items": []}
    fenced = "```json\n" + json.dumps(payload) + "\n```"
    gen = QAGenerator(FakeLLM(chat_reply=fenced))
    doc = await gen.generate(_parse_result())
    assert doc.items == []


def test_to_markdown_renders_sections_and_disclaimer():
    from app.schemas import QADocument, QAItem

    doc = QADocument(
        company="Acme Pharma",
        period="Q1",
        generated_from="q1.pdf",
        items=[
            QAItem(
                question="Guidance?",
                answer="Raised to high end.",
                asker="journalist",
                reasoning="Raised guidance on p.2 raises questions on pricing.",
                peer_context="Lilly also raised on obesity demand.",
                category="guidance",
                source_pages=[2],
                caveat="From handwritten note; verify.",
            )
        ],
    )
    md = QAGenerator.to_markdown(doc)
    assert "# Pharma Analyst & Journalist Q&A Brief" in md
    assert "Guidance" in md
    assert "Q (Journalist):" in md
    assert "Why it's asked:" in md
    assert "Peer context:" in md
    assert "asked by: journalist" in md
    assert "source page(s): 2" in md
    assert "verify" in md
    assert "official filing" in md
