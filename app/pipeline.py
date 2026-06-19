"""End-to-end orchestration: PDF bytes -> ParseResult -> QADocument."""
from __future__ import annotations

import time

from .config import Settings, get_settings
from .llm.factory import build_llm_provider
from .parsers.router import ParserRouter
from .qa.generator import QAGenerator
from .qa.peer_retrieval import PeerCitationEnricher
from .schemas import ProcessResponse, QADocument


class Pipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self.router = ParserRouter(self.s)

    async def parse_only(self, doc_bytes: bytes, filename: str) -> ProcessResponse:
        t0 = time.perf_counter()
        parse = await self.router.parse(doc_bytes, filename)
        return ProcessResponse(
            parse=parse,
            qa=None,
            timings_ms={"parse": round((time.perf_counter() - t0) * 1000, 1)},
            meta={"parser_backend": self.s.parser_backend},
        )

    async def process(
        self,
        doc_bytes: bytes,
        filename: str,
        *,
        num_questions: int = 12,
        peers: list[str] | None = None,
    ) -> ProcessResponse:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        parse = await self.router.parse(doc_bytes, filename)
        timings["parse"] = round((time.perf_counter() - t0) * 1000, 1)

        t1 = time.perf_counter()
        provider = build_llm_provider(self.s)
        web_search_active = self.s.peer_web_search and PeerCitationEnricher.supported(
            provider
        )
        qa: QADocument | None
        peer_grounded = False
        try:
            qa = await QAGenerator(provider).generate(
                parse, num_questions=num_questions, peers=peers
            )
            timings["qa"] = round((time.perf_counter() - t1) * 1000, 1)

            # Second, best-effort pass: ground peer-comparison questions with
            # live web search so they carry real citations. Only runs when
            # enabled and the active provider supports web search (Anthropic).
            if qa and web_search_active:
                t2 = time.perf_counter()
                qa = await PeerCitationEnricher(
                    provider, max_uses=self.s.peer_web_search_max_uses
                ).enrich(qa, peers=peers)
                timings["peer_retrieval"] = round(
                    (time.perf_counter() - t2) * 1000, 1
                )
                peer_grounded = any(it.peer_citations for it in qa.items)
        finally:
            await provider.aclose()

        meta = {
            "parser_backend": self.s.parser_backend,
            "llm_provider": self.s.llm_provider,
            "vision_provider": self.s.effective_vision_provider,
            "peer_web_search": web_search_active,
            "peer_grounded": peer_grounded,
        }
        if peers:
            meta["peers"] = peers
        return ProcessResponse(parse=parse, qa=qa, timings_ms=timings, meta=meta)
