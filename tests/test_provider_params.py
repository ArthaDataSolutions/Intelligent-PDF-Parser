"""Regression tests: hosted providers must not send params the current
frontier models reject.

- Claude 4.x (opus 4.7+/sonnet 4.6+) rejects `temperature` and `top_p`.
- GPT-5.x rejects `temperature`/`top_p` and requires `max_completion_tokens`
  instead of `max_tokens`.

These tests stub the SDK client's create() and assert on the outgoing kwargs,
so they run fully offline.
"""
from __future__ import annotations

import types

from app.llm.anthropic_provider import AnthropicProvider
from app.llm.base import LLMMessage
from app.llm.openai_provider import OpenAIProvider


def _anthropic_resp():
    return types.SimpleNamespace(
        content=[types.SimpleNamespace(type="text", text="ok")]
    )


def _openai_resp():
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]
    )


def _capture(into: dict, resp):
    async def _create(**kwargs):
        into.update(kwargs)
        return resp
    return _create


async def test_anthropic_chat_omits_temperature_and_top_p():
    p = AnthropicProvider(api_key="x", text_model="claude-opus-4-7", vision_model="claude-opus-4-7")
    sent: dict = {}
    p._client.messages.create = _capture(sent, _anthropic_resp())

    await p.chat([LLMMessage(role="user", content="hi")], temperature=0.0, max_tokens=32)

    assert "temperature" not in sent
    assert "top_p" not in sent
    assert sent["max_tokens"] == 32


async def test_anthropic_vision_omits_temperature():
    p = AnthropicProvider(api_key="x", text_model="m", vision_model="m")
    sent: dict = {}
    p._client.messages.create = _capture(sent, _anthropic_resp())

    await p.vision("prompt", [b"\x89PNG\r\n\x1a\n"], temperature=0.0, max_tokens=64)

    assert "temperature" not in sent
    assert sent["max_tokens"] == 64


async def test_openai_chat_uses_max_completion_tokens_no_temperature():
    p = OpenAIProvider(api_key="x", text_model="gpt-5.5", vision_model="gpt-5.5")
    sent: dict = {}
    p._client.chat.completions.create = _capture(sent, _openai_resp())

    await p.chat([LLMMessage(role="user", content="hi")], temperature=0.5, max_tokens=128)

    assert "temperature" not in sent
    assert "max_tokens" not in sent
    assert sent["max_completion_tokens"] == 128


async def test_openai_vision_uses_max_completion_tokens_no_temperature():
    p = OpenAIProvider(api_key="x", text_model="gpt-5.5", vision_model="gpt-5.5")
    sent: dict = {}
    p._client.chat.completions.create = _capture(sent, _openai_resp())

    await p.vision("prompt", [b"\x89PNG\r\n\x1a\n"], temperature=0.0, max_tokens=200)

    assert "temperature" not in sent
    assert "max_tokens" not in sent
    assert sent["max_completion_tokens"] == 200
