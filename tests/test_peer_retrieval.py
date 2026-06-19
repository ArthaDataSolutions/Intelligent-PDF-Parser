"""Tests for web-grounded peer citation enrichment (app.qa.peer_retrieval)."""
from __future__ import annotations

from app.llm.anthropic_provider import GroundedAnswer, WebCitation
from app.qa.peer_retrieval import PeerCitationEnricher
from app.schemas import QADocument, QAItem
from tests.conftest import FakeLLM, FakeSearchLLM


def _doc() -> QADocument:
    return QADocument(
        company="Hikma",
        period="FY2023",
        generated_from="hikma.pdf",
        items=[
            QAItem(
                question="How do your generics margins compare to larger peers?",
                answer="Core gross margin was 49%.",
                asker="analyst",
                peer_context="Analysts benchmark generics margins against Teva and Pfizer.",
                source_pages=[1],
            ),
            QAItem(
                question="What drove the dividend increase?",
                answer="Board raised the dividend on stronger cash flow.",
                asker="journalist",
                peer_context=None,  # NOT a peer question — must stay untouched
                source_pages=[3],
            ),
        ],
    )


def _grounded() -> GroundedAnswer:
    return GroundedAnswer(
        text="Pfizer fielded a similar margin question on its Q3 2025 call and "
             "pointed to cost programs.",
        citations=[
            WebCitation(
                url="https://example.com/pfizer-q3-2025-call",
                title="Pfizer Q3 2025 Earnings Call Transcript",
                cited_text="We expect margin recovery from our cost realignment.",
            )
        ],
    )


def test_supported_detects_web_search_capability():
    assert PeerCitationEnricher.supported(FakeSearchLLM()) is True
    assert PeerCitationEnricher.supported(FakeLLM()) is False


async def test_enrich_grounds_only_peer_items():
    provider = FakeSearchLLM(grounded=_grounded())
    doc = _doc()
    out = await PeerCitationEnricher(provider).enrich(doc, peers=["Teva", "Pfizer"])

    peer_item, plain_item = out.items[0], out.items[1]
    # Peer item gets a grounded answer + real citation.
    assert peer_item.peer_answer and "Pfizer" in peer_item.peer_answer
    assert len(peer_item.peer_citations) == 1
    cite = peer_item.peer_citations[0]
    assert cite.url == "https://example.com/pfizer-q3-2025-call"
    assert cite.peer == "Pfizer"  # attributed from the title
    assert cite.venue == "earnings call"  # "Earnings Call" in the title
    assert cite.quote
    # Non-peer item is untouched, and only the one peer question was searched.
    assert plain_item.peer_answer is None
    assert plain_item.peer_citations == []
    assert len(provider.search_calls) == 1


async def test_enrich_grounded_or_nothing_when_no_citations():
    """A search that returns no citations must not produce a peer_answer."""
    provider = FakeSearchLLM(grounded=GroundedAnswer(text="Some ungrounded musing", citations=[]))
    out = await PeerCitationEnricher(provider).enrich(_doc(), peers=["Pfizer"])
    assert out.items[0].peer_answer is None
    assert out.items[0].peer_citations == []


async def test_enrich_is_non_fatal_on_provider_error():
    provider = FakeSearchLLM(raises=True)
    doc = _doc()
    out = await PeerCitationEnricher(provider).enrich(doc, peers=["Pfizer"])
    # Item survives unchanged; no exception propagates.
    assert out.items[0].peer_answer is None
    assert out.items[0].peer_citations == []


async def test_enrich_no_peer_items_returns_same_doc():
    provider = FakeSearchLLM(grounded=_grounded())
    doc = QADocument(items=[QAItem(question="q", answer="a", peer_context=None)])
    out = await PeerCitationEnricher(provider).enrich(doc)
    assert out is doc
    assert provider.search_calls == []


async def test_enrich_does_not_mutate_input_document():
    provider = FakeSearchLLM(grounded=_grounded())
    doc = _doc()
    await PeerCitationEnricher(provider).enrich(doc, peers=["Pfizer"])
    # Original document's peer item must remain ungrounded (immutability).
    assert doc.items[0].peer_answer is None
    assert doc.items[0].peer_citations == []


def test_qaitem_enforces_grounded_or_nothing_invariant():
    from app.schemas import PeerCitation

    # peer_answer without any citation is dropped (structural invariant).
    bare = QAItem(question="q", answer="a", peer_answer="unsourced peer claim")
    assert bare.peer_answer is None
    assert bare.peer_citations == []

    # peer_answer backed by a real citation is kept.
    grounded = QAItem(
        question="q", answer="a", peer_answer="grounded claim",
        peer_citations=[PeerCitation(url="https://x", title="X")],
    )
    assert grounded.peer_answer == "grounded claim"

    # The invariant also holds on deserialization (e.g. reading qa_json from DB).
    revived = QAItem.model_validate_json(
        '{"question":"q","answer":"a","peer_answer":"unsourced","peer_citations":[]}'
    )
    assert revived.peer_answer is None


def test_attribute_venue_from_title_or_url():
    from app.qa.peer_retrieval import _attribute_venue

    assert _attribute_venue("Pfizer Q3 2025 Earnings Call", "") == "earnings call"
    assert _attribute_venue("Merck Investor Day 2025", "") == "investor day"
    assert _attribute_venue("Form 10-K", "https://sec.gov/...10-k.htm") == "annual report"
    assert _attribute_venue("Novartis Press Release", "") == "press release"
    assert _attribute_venue("Some Random Blog Post", "https://blog.example.com") is None


def test_collect_grounding_drops_preamble_dedups_and_filters():
    from types import SimpleNamespace as NS

    from app.llm.anthropic_provider import AnthropicProvider

    resp = NS(content=[
        # Pre-search planning narration — must NOT leak into the answer.
        NS(type="text", text="I'll search for Pfizer's revenue.", citations=None),
        NS(type="server_tool_use"),
        NS(type="web_search_tool_result"),
        # The real answer comes after the final search, carrying the citations.
        NS(type="text", text="Pfizer reported $58.5B.", citations=[
            NS(type="web_search_result_location", url="https://a", title="A", cited_text="x"),
            NS(type="web_search_result_location", url="https://a", title="A", cited_text="x"),
            NS(type="char_location", url="", title="", cited_text="ignored"),
            NS(type="web_search_result_location", url="", title="no-url", cited_text="z"),
        ]),
    ])
    ga = AnthropicProvider._collect_grounding(resp)
    assert ga.text == "Pfizer reported $58.5B."  # planning preamble dropped
    assert len(ga.citations) == 1  # duped, non-web-search, and url-less dropped
    assert ga.citations[0].url == "https://a"


def test_to_markdown_renders_peer_answer_and_citation_link():
    from app.qa.generator import QAGenerator
    from app.schemas import PeerCitation

    doc = QADocument(
        company="Hikma", period="FY2023", generated_from="hikma.pdf",
        items=[QAItem(
            question="Margins vs peers?",
            answer="49% core gross margin.",
            asker="analyst",
            peer_context="Analysts benchmark vs Pfizer.",
            peer_answer="Pfizer cited cost programs on its Q3 2025 call.",
            peer_citations=[PeerCitation(
                url="https://example.com/pfizer",
                title="Pfizer Q3 2025 Call",
                peer="Pfizer",
                quote="margin recovery from cost realignment",
            )],
            source_pages=[1],
        )],
    )
    md = QAGenerator.to_markdown(doc)
    assert "How peers answered it:" in md
    assert "[Pfizer — Pfizer Q3 2025 Call](https://example.com/pfizer)" in md
    assert "margin recovery from cost realignment" in md
