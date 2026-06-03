"""Analyst Q&A generator.

Use case: a CFO publishes a quarterly report; WSJ/NYT analysts will ask pointed
questions. This module reads the parsed report and produces a grounded Q&A
document the finance team can use to prepare.

Anti-hallucination design (best practice, June 2026):
  * The model answers ONLY from the supplied parsed text — instructed to write
    "Not disclosed in this document" rather than invent figures.
  * Each answer cites the source page(s) it drew from.
  * Output is constrained to a strict JSON schema and validated with Pydantic.
  * Figures lifted from handwritten content are flagged with a caveat, since
    those carry transcription risk.
"""
from __future__ import annotations

import json

from ..llm.base import LLMMessage, LLMProvider
from ..logging_conf import get_logger
from ..schemas import ParseResult, QADocument, QAItem

log = get_logger("qa.generator")

_SYSTEM = """You are a financial-disclosure analyst assistant. You prepare a CFO
for tough questions from journalists and sell-side analysts (WSJ, NYT, etc.)
after a quarterly report.

Hard rules:
- Answer ONLY using facts present in the provided document text. If the document
  does not contain the answer, say "Not disclosed in this document." Never invent
  or estimate numbers.
- Every answer must cite the page number(s) it relied on.
- Prefer the toughest, most material questions an analyst would actually ask:
  margins, guidance, segment performance, cash flow, debt, one-offs, YoY/QoQ
  deltas, risks, and anything that looks unusual or hand-annotated.
- If a figure came from a handwritten note (marked with [handwritten: ...] in the
  source), include a caveat that it should be verified against the official filing.
- Be concise and precise. Use exact figures and units from the document."""

_INSTRUCTION = """Document text (page markers like `<!-- page N -->` indicate page
numbers; `[handwritten: ...]` marks transcribed handwriting):

\"\"\"
{doc}
\"\"\"

Generate {n} analyst-style question/answer pairs. Return STRICT JSON only, shape:
{{
  "company": string|null,
  "period": string|null,
  "items": [
    {{
      "question": string,
      "answer": string,
      "category": "guidance"|"margins"|"segments"|"cash_flow"|"balance_sheet"|"risks"|"general",
      "confidence": "low"|"medium"|"high",
      "source_pages": [int],
      "caveat": string|null
    }}
  ]
}}
No prose outside the JSON object."""


class QAGenerator:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    @staticmethod
    def _extract_json(raw: str) -> dict:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            raw = raw[4:] if raw.lower().startswith("json") else raw
        start, end = raw.find("{"), raw.rfind("}") + 1
        return json.loads(raw[start:end])

    async def generate(
        self,
        parse: ParseResult,
        *,
        num_questions: int = 12,
        max_chars: int = 60_000,
    ) -> QADocument:
        doc_md = parse.markdown[:max_chars]
        log.info("qa_generating", num_questions=num_questions, doc_chars=len(doc_md))
        log.debug(
            "qa_request",
            provider=self._provider.name,
            model=getattr(self._provider, "text_model", ""),
            requested_questions=num_questions,
            max_chars=max_chars,
            truncated=len(parse.markdown) > max_chars,
        )
        messages = [
            LLMMessage(role="system", content=_SYSTEM),
            LLMMessage(
                role="user",
                content=_INSTRUCTION.format(doc=doc_md, n=num_questions),
            ),
        ]
        raw = await self._provider.chat(
            messages, temperature=0.2, max_tokens=4096, json_mode=True
        )
        log.debug("qa_response_received", chars=len(raw))
        data = self._extract_json(raw)

        items: list[QAItem] = []
        for it in data.get("items", []):
            try:
                items.append(QAItem(**it))
            except Exception:
                # tolerate a single malformed item rather than failing the batch
                continue

        log.info("qa_generated", items=len(items))
        return QADocument(
            company=data.get("company"),
            period=data.get("period"),
            generated_from=parse.source_filename,
            items=items,
        )

    @staticmethod
    def to_markdown(doc: QADocument) -> str:
        lines = [f"# {doc.title}"]
        sub = " — ".join(x for x in (doc.company, doc.period) if x)
        if sub:
            lines.append(f"*{sub}*")
        lines.append(f"\n_Source: {doc.generated_from}_\n")
        by_cat: dict[str, list[QAItem]] = {}
        for it in doc.items:
            by_cat.setdefault(it.category, []).append(it)
        for cat, group in by_cat.items():
            lines.append(f"\n## {cat.replace('_', ' ').title()}\n")
            for it in group:
                pages = ", ".join(map(str, it.source_pages)) or "—"
                lines.append(f"**Q: {it.question}**\n")
                lines.append(f"{it.answer}\n")
                meta = f"<sub>confidence: {it.confidence} · source page(s): {pages}</sub>"
                lines.append(meta)
                if it.caveat:
                    lines.append(f"\n> ⚠️ {it.caveat}")
                lines.append("")
        lines.append(f"\n---\n_{doc.disclaimer}_")
        return "\n".join(lines)
