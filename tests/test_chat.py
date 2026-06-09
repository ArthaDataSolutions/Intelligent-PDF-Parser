"""Unit tests for grounded interactive Q&A (DocChat)."""
from __future__ import annotations

import pytest

from app.llm.base import LLMMessage
from app.qa.chat import DocChat
from tests.conftest import FakeLLM

_DOC = "<!-- page 1 -->\nRevenue was $5.0M.\n<!-- page 2 -->\nNet loss was $1M."


@pytest.mark.asyncio
async def test_answer_parses_json_and_pages():
    llm = FakeLLM(chat_reply='{"answer": "Revenue was $5.0M.", "source_pages": [1]}')
    ans = await DocChat(llm).answer(_DOC, "What was revenue?")
    assert ans.answer == "Revenue was $5.0M."
    assert ans.source_pages == [1]


@pytest.mark.asyncio
async def test_answer_passes_document_and_history():
    llm = FakeLLM(chat_reply='{"answer": "ok", "source_pages": []}')
    history = [LLMMessage(role="user", content="earlier")]
    await DocChat(llm).answer(_DOC, "Follow up?", history=history)
    sent = llm.chat_calls[-1]
    # system + doc context + history + question
    assert sent[0].role == "system"
    assert "Revenue was $5.0M." in sent[1].content
    assert sent[-1].content == "Follow up?"
    assert any(m.content == "earlier" for m in sent)


@pytest.mark.asyncio
async def test_answer_tolerates_non_json_reply():
    llm = FakeLLM(chat_reply="Just plain prose, no JSON here.")
    ans = await DocChat(llm).answer(_DOC, "Anything?")
    assert ans.answer == "Just plain prose, no JSON here."
    assert ans.source_pages == []


@pytest.mark.asyncio
async def test_answer_coerces_bad_page_values():
    llm = FakeLLM(chat_reply='{"answer": "x", "source_pages": [2, "bad", null]}')
    ans = await DocChat(llm).answer(_DOC, "Pages?")
    assert ans.source_pages == [2]
