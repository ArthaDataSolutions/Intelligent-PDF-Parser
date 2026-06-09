"""End-to-end orchestration: PDF bytes -> ParseResult -> QADocument."""
from __future__ import annotations

import time

from .config import Settings, get_settings
from .llm.factory import build_llm_provider
from .parsers.router import ParserRouter
from .qa.generator import QAGenerator
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
        qa: QADocument | None
        try:
            qa = await QAGenerator(provider).generate(
                parse, num_questions=num_questions, peers=peers
            )
        finally:
            await provider.aclose()
        timings["qa"] = round((time.perf_counter() - t1) * 1000, 1)

        meta = {
            "parser_backend": self.s.parser_backend,
            "llm_provider": self.s.llm_provider,
            "vision_provider": self.s.effective_vision_provider,
        }
        if peers:
            meta["peers"] = peers
        return ProcessResponse(parse=parse, qa=qa, timings_ms=timings, meta=meta)
