"""Web-grounded peer citations for analyst peer-comparison questions.

The Q&A generator (``app.qa.generator``) deliberately keeps peer context
*qualitative* — it is forbidden from inventing competitor figures or sources.
This module adds the missing real-world grounding as a separate, best-effort
second pass: for each peer-comparison item (one with ``peer_context``), it asks
Claude — using Anthropic's server-side **web-search tool** — whether a
comparable pharma company actually faced a similar question and how they
answered it, and attaches the *real* citations the search returned.

Design rules:
  * Best-effort and non-fatal. Any per-item failure leaves that item unchanged;
    enrichment never fails the run (the generator output is still valuable).
  * Only items with ``peer_context`` are searched, bounding cost.
  * Grounded-or-nothing. If a search yields no citations, the item keeps no
    ``peer_answer`` — we never surface ungrounded text as if it were sourced.
  * Provider-agnostic by capability: a provider participates only if it exposes
    ``search_with_citations`` (today, just Anthropic). Others are skipped.
  * Immutable: returns a new ``QADocument``; inputs are never mutated.
"""
from __future__ import annotations

import asyncio

from ..logging_conf import get_logger
from ..schemas import PeerCitation, QADocument, QAItem

log = get_logger("qa.peer_retrieval")

# Fallback roster used for attribution and prompt hinting when the run supplies
# no explicit peers. Mirrors the roster named in the generator's system prompt.
DEFAULT_PEER_ROSTER: tuple[str, ...] = (
    "Pfizer", "Merck", "Novartis", "AstraZeneca", "Eli Lilly",
    "Bristol Myers Squibb", "AbbVie", "Amgen", "GSK", "Sanofi",
    "Novo Nordisk", "Roche",
)

_SYSTEM = (
    "You are a pharma equity-research assistant. Answer ONLY from the web-search "
    "results you retrieve. Never fabricate sources, companies, quotes, dates, or "
    "numbers. If the search returns nothing relevant, say so plainly and stop. "
    "Prefer primary sources (earnings-call transcripts, investor-day decks, "
    "10-K/10-Q/20-F filings) and reputable financial press."
)

_NOTHING_FOUND = "No comparable peer disclosure found."


def _peer_hint(peers: list[str] | None) -> str:
    names = [p.strip() for p in (peers or []) if p.strip()]
    if names:
        return ", ".join(names)
    return (
        "large-cap pharma peers (e.g. "
        + ", ".join(DEFAULT_PEER_ROSTER)
        + ")"
    )


def _build_prompt(question: str, peer_hint: str) -> str:
    return (
        "A pharmaceutical company's CFO faces this analyst peer-comparison "
        f'question after reporting results:\n\n"{question}"\n\n'
        f"Search the web for whether a comparable pharma company — {peer_hint} — "
        "has faced a similar question recently (an earnings call, investor day, "
        "analyst Q&A, regulatory filing, or reputable news report), and, if so, "
        "how that company answered it.\n\n"
        "Write 2-4 sentences grounded in real sources. Name the company and the "
        'venue (e.g. "Pfizer, Q3 2025 earnings call") when the source shows it. '
        f'If you find no relevant real source, reply exactly "{_NOTHING_FOUND}" '
        "and nothing else — do not speculate or invent."
    )


def _attribute_peer(title: str, url: str, peers: list[str]) -> str | None:
    """Best-effort: which peer company a citation is about, from its title/url."""
    haystack = f"{title} {url}".lower()
    for name in peers:
        if name and name.lower() in haystack:
            return name
    return None


class PeerCitationEnricher:
    """Adds web-grounded ``peer_answer`` + ``peer_citations`` to peer questions."""

    def __init__(
        self,
        provider,
        *,
        max_uses: int = 5,
        max_concurrency: int = 3,
        max_tokens: int = 1024,
    ) -> None:
        self._provider = provider
        self._max_uses = max_uses
        self._max_tokens = max_tokens
        self._sem = asyncio.Semaphore(max(1, max_concurrency))

    @staticmethod
    def supported(provider) -> bool:
        """True when the provider can run web search with citations."""
        return callable(getattr(provider, "search_with_citations", None))

    async def enrich(
        self, doc: QADocument, *, peers: list[str] | None = None
    ) -> QADocument:
        targets = [i for i, it in enumerate(doc.items) if it.peer_context]
        if not targets:
            return doc

        roster = [p.strip() for p in (peers or []) if p.strip()] or list(
            DEFAULT_PEER_ROSTER
        )
        hint = _peer_hint(peers)
        log.info("peer_retrieval_started", peer_items=len(targets))

        results = await asyncio.gather(
            *(self._enrich_one(doc.items[i], hint, roster) for i in targets),
            return_exceptions=True,
        )

        new_items = list(doc.items)
        grounded = 0
        for idx, res in zip(targets, results):
            if isinstance(res, QAItem):
                new_items[idx] = res
                if res.peer_citations:
                    grounded += 1
            elif isinstance(res, Exception):
                log.warning("peer_retrieval_item_failed", error=str(res))
        log.info("peer_retrieval_done", grounded=grounded, peer_items=len(targets))
        return doc.model_copy(update={"items": new_items})

    async def _enrich_one(
        self, item: QAItem, hint: str, roster: list[str]
    ) -> QAItem:
        async with self._sem:
            grounded = await self._provider.search_with_citations(
                _build_prompt(item.question, hint),
                system=_SYSTEM,
                max_uses=self._max_uses,
                max_tokens=self._max_tokens,
            )
        # Grounded-or-nothing: no real citations means no usable peer answer.
        if not grounded.citations:
            return item
        citations = [
            PeerCitation(
                url=c.url,
                title=c.title,
                quote=c.cited_text,
                peer=_attribute_peer(c.title, c.url, roster),
            )
            for c in grounded.citations
        ]
        text = grounded.text.strip()
        answer = None if text == _NOTHING_FOUND or not text else text
        return item.model_copy(
            update={"peer_answer": answer, "peer_citations": citations}
        )
