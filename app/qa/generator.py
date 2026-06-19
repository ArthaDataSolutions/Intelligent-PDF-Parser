"""Pharma analyst & journalist Q&A generator.

Scope: this generator is specialised for **pharmaceutical companies' financial
reports** (quarterly/annual results, earnings releases). A pharma CFO/IR team
publishes results; the people who then ask pointed questions are:

  * **Journalists** (WSJ, NYT, FT, Reuters, STAT, Endpoints) — public-interest
    angles: drug pricing and access, litigation, safety, layoffs, exec pay.
  * **Analysts** (sell-side / buy-side) — financial-model angles: guidance
    bridges, margins, pipeline value, and — importantly — **peer comparisons**
    against similar pharma companies (Pfizer, Merck, Novartis, AstraZeneca,
    Lilly, BMS, AbbVie, Amgen, GSK, Sanofi, Novo Nordisk, Roche, etc.).

Each generated item therefore records *who* would ask it (`asker`) and *why*
it is being asked / where it comes from (`reasoning`), and — when it is a
benchmarking question — how peers frame the same issue (`peer_context`).

Freshness: there is no stored question bank. Every question is generated fresh
from the most recent period disclosed in *this* document, so the output is
always tied to the latest filing rather than recycled boilerplate.

Anti-hallucination design (best practice, June 2026):
  * Answers use ONLY facts present in the supplied parsed text — the model is
    instructed to write "Not disclosed in this document" rather than invent
    figures, and to never fabricate specific competitor numbers.
  * Every answer cites the source page(s) it drew from; `reasoning` references
    those pages so the CFO can see the provenance.
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

_SYSTEM = """You are a pharma equity-research and financial-disclosure assistant.
You prepare a PHARMACEUTICAL company's CFO/IR team for the tough questions they
will face after publishing financial results. The document you are given is a
pharma company's financial report. If it is clearly NOT a pharma/biotech/
life-sciences financial report, set company/period to null and return an empty
items list rather than inventing pharma questions.

Two kinds of people ask the questions; tag every item with the right one:
- "journalist": WSJ / NYT / FT / Reuters / STAT / Endpoints reporters. They
  push on public-interest angles — drug pricing and patient access (IRA Medicare
  price negotiation, list-vs-net price, rebates, 340B), litigation, product
  safety/recalls, layoffs/restructuring, and executive compensation.
- "analyst": sell-side / buy-side analysts. They push on the financial model and
  routinely BENCHMARK AGAINST SIMILAR PHARMA COMPANIES (e.g. Pfizer, Merck,
  Novartis, AstraZeneca, Eli Lilly, Bristol Myers Squibb, AbbVie, Amgen, GSK,
  Sanofi, Novo Nordisk, Roche). Many analyst questions should be framed as
  peer/competitor comparisons.

Prefer the toughest, most MATERIAL pharma-specific questions, drawn from this
filing: top product/franchise revenue and growth, loss of exclusivity / patent
cliff and biosimilar-or-generic erosion, pipeline and clinical/regulatory
catalysts (Phase II/III readouts, FDA/EMA approvals, PDUFA dates), R&D spend and
productivity, gross/operating margins, guidance (revenue/EPS) and the bridge to
it, segment/geographic mix, FX, cash flow and capital allocation
(buybacks/dividends/BD&L, milestones and royalties), and litigation/regulatory
risk.

Hard rules:
- Answer ONLY using facts present in the provided document text. If the document
  does not contain the answer, write "Not disclosed in this document." Never
  invent or estimate numbers, and NEVER fabricate specific competitor figures —
  peer context must stay qualitative (name the peer and the theme, not made-up
  numbers).
- Base questions on the MOST RECENT period disclosed in this document. Do not
  recycle generic, undated questions.
- Every answer must cite the page number(s) it relied on (`source_pages`).
- Every item must include `reasoning`: one or two sentences on WHY this question
  is being asked and WHERE it comes from (what line/metric in the filing, citing
  the page; and/or the relevant sector dynamic). Citations matter — ground it.
- For peer/benchmarking questions set `peer_context` to how comparable pharma
  companies frame the same issue; otherwise set it to null.
- If a figure came from a handwritten note (marked [handwritten: ...] in the
  source), add a caveat that it should be verified against the official filing.
- Be concise and precise. Use exact figures and units from the document."""

_INSTRUCTION = """Document text (page markers like `<!-- page N -->` indicate page
numbers; `[handwritten: ...]` marks transcribed handwriting):

\"\"\"
{doc}
\"\"\"
{peers}
Generate {n} question/answer pairs a pharma CFO would be drilled on, mixing
journalist and analyst askers, and including several analyst peer-comparison
questions. Return STRICT JSON only, shape:
{{
  "company": string|null,
  "period": string|null,
  "items": [
    {{
      "question": string,
      "answer": string,
      "asker": "journalist"|"analyst",
      "reasoning": string,
      "peer_context": string|null,
      "category": "pipeline"|"rd_productivity"|"regulatory"|"patent_exclusivity"|"competition"|"pricing_access"|"guidance"|"margins"|"segments"|"cash_flow"|"balance_sheet"|"risks"|"general",
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
        snippet = raw[start:end]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            # The model most likely hit its output-token limit and the JSON was
            # cut off mid-object. Rather than failing the whole run, salvage the
            # complete items that did make it through.
            log.warning("qa_json_truncated", chars=len(raw))
            return QAGenerator._salvage_truncated(raw[start:])

    @staticmethod
    def _scan_string_field(text: str, key: str) -> str | None:
        """Pull a top-level "key": "value"|null pair out of (possibly broken) JSON."""
        marker = f'"{key}"'
        i = text.find(marker)
        if i == -1:
            return None
        i = text.find(":", i + len(marker))
        if i == -1:
            return None
        rest = text[i + 1:].lstrip()
        if rest.startswith("null"):
            return None
        if not rest.startswith('"'):
            return None
        # Read the string body, honouring escaped quotes.
        out, esc = [], False
        for ch in rest[1:]:
            if esc:
                out.append(ch)
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                break
            else:
                out.append(ch)
        return "".join(out)

    @staticmethod
    def _iter_json_objects(text: str):
        """Yield each complete top-level {...} object in `text`, string-aware.

        Stops at the first incomplete object (an unbalanced trailing brace from a
        truncated response), so callers get every whole item and nothing partial.
        """
        depth, in_str, esc, start = 0, False, False, -1
        for idx, ch in enumerate(text):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                if depth == 0:
                    start = idx
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    yield text[start: idx + 1]
                    start = -1
            elif ch == "]" and depth == 0:
                break  # end of the items array

    @staticmethod
    def _salvage_truncated(text: str) -> dict:
        """Recover company/period and any complete item objects from cut-off JSON."""
        items: list[dict] = []
        marker = text.find('"items"')
        if marker != -1:
            arr = text.find("[", marker)
            if arr != -1:
                for obj in QAGenerator._iter_json_objects(text[arr + 1:]):
                    try:
                        items.append(json.loads(obj))
                    except json.JSONDecodeError:
                        break
        return {
            "company": QAGenerator._scan_string_field(text, "company"),
            "period": QAGenerator._scan_string_field(text, "period"),
            "items": items,
        }

    async def generate(
        self,
        parse: ParseResult,
        *,
        num_questions: int = 12,
        max_chars: int = 60_000,
        peers: list[str] | None = None,
    ) -> QADocument:
        doc_md = parse.markdown[:max_chars]
        # Each rich Q&A item (question + grounded answer + reasoning + peer
        # context + metadata) runs ~450 output tokens. Size the budget to the
        # requested count so large batches aren't truncated mid-JSON — the bug
        # that previously failed whole runs. No upper clamp: let the provider's
        # own model limit be the ceiling.
        out_tokens = max(4096, num_questions * 600)
        peer_hint = ""
        if peers:
            named = ", ".join(p.strip() for p in peers if p.strip())
            if named:
                peer_hint = (
                    f"\nWhen framing analyst peer-comparison questions, benchmark "
                    f"against these comparable companies: {named}.\n"
                )
        log.info(
            "qa_generating",
            num_questions=num_questions,
            doc_chars=len(doc_md),
            peers=peers or [],
        )
        log.debug(
            "qa_request",
            provider=self._provider.name,
            model=getattr(self._provider, "text_model", ""),
            requested_questions=num_questions,
            max_chars=max_chars,
            max_tokens=out_tokens,
            truncated=len(parse.markdown) > max_chars,
        )
        messages = [
            LLMMessage(role="system", content=_SYSTEM),
            LLMMessage(
                role="user",
                content=_INSTRUCTION.format(
                    doc=doc_md, n=num_questions, peers=peer_hint
                ),
            ),
        ]
        raw = await self._provider.chat(
            messages, temperature=0.2, max_tokens=out_tokens, json_mode=True
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
                lines.append(f"**Q ({it.asker.title()}): {it.question}**\n")
                lines.append(f"{it.answer}\n")
                if it.reasoning:
                    lines.append(f"> 💡 *Why it's asked:* {it.reasoning}")
                if it.peer_context:
                    lines.append(f"> 🔍 *Peer context:* {it.peer_context}")
                if it.peer_answer:
                    lines.append(f"> 🌐 *How peers answered it:* {it.peer_answer}")
                for c in it.peer_citations:
                    label = c.title or c.url
                    attrib = ", ".join(x for x in (c.peer, c.venue) if x)
                    who = f"{attrib} — " if attrib else ""
                    cite = f"> 📎 [{who}{label}]({c.url})"
                    if c.quote:
                        cite += f" — “{c.quote.strip()}”"
                    lines.append(cite)
                meta = (
                    f"<sub>asked by: {it.asker} · confidence: {it.confidence} · "
                    f"source page(s): {pages}</sub>"
                )
                lines.append(meta)
                if it.caveat:
                    lines.append(f"\n> ⚠️ {it.caveat}")
                lines.append("")
        lines.append(f"\n---\n_{doc.disclaimer}_")
        return "\n".join(lines)
