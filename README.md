# Intelligent PDF Parser

A production-grade service that turns financial-report PDFs — including ones with
**handwritten notes and annotations** — into clean structured Markdown, then uses
an LLM to generate an **analyst- and journalist-style Q&A document** a CFO can use
to prepare for tough questions after results.

The Q&A generator is **specialised for pharmaceutical companies' financial
reports**. It models the two audiences that actually ask the questions —
**journalists** (WSJ/NYT/FT/Reuters/STAT/Endpoints; pricing & access, litigation,
safety, exec pay) and **sell-/buy-side analysts** (guidance, margins, pipeline,
and **peer comparisons** against similar pharma names) — and for every question it
records *who* would ask it, *why* it is being asked (with citations back to the
source page), and, for benchmarking questions, how comparable companies frame the
same issue. There is no stored question bank: every brief is generated fresh from
the latest filing you upload, so the questions always reflect the most recent
period.

It is built around two ideas that reflect the state of document AI as of mid-2026:

1. **No single parser wins on every page.** LLM/vision parsers read messy layouts
   and handwriting best; rule-based and layout-aware parsers (Docling, LlamaParse)
   are far cheaper and excellent on clean printed text. So this service runs a
   **router**: cheap backend for printed pages, a **vision-LLM backend for
   handwritten / scanned / low-confidence pages**, with automatic escalation.
2. **Provider independence.** Parsing-assist and Q&A generation run through a thin
   provider abstraction, so you can point it at **OpenAI, Anthropic (Claude),
   Azure OpenAI, or a self-hosted Ollama** model by changing one env var.

## Why this design (June 2026 landscape)

Recent benchmarks converge on a few points. Vision-LLM parsers produce the
cleanest output on visually complex documents and are the only category that reads
handwriting reliably; Gemini-class and GPT-5.5-class models lead on raw accuracy
while LlamaParse hits the cost/quality sweet spot (~$0.003/page) for clean printed
docs. **Docling (IBM, open-source) is strong on tables/layout and fully
self-hostable but does not handle handwriting.** Frontier multimodal models
(GPT-5.5, Claude Opus 4.x) can now read dense scans and handwritten forms in a single
pass, and vendor guidance is explicit that for handwriting / tiny text / low-quality
scans you should send the page image at full ("original") detail. These frontier
models are deterministic by default and no longer expose a `temperature`/`top_p`
knob (the API rejects them), so the provider layer omits those parameters. Those
findings are baked directly into the router and the vision prompt. Sources are
listed at the bottom of this file.

This is also the answer to the **Snowflake handwriting problem** (notes read
inaccurately today): instead of OCR on a text layer that doesn't exist, the
handwritten pages are rendered to images and read by a frontier vision model that
is instructed to flag uncertainty (`[illegible]`) rather than guess at numbers.

## Architecture

```
PDF bytes
   │
   ▼
PyMuPDF page analysis ──► per-page signals (text coverage, image coverage,
   │                       empty-text-layer ⇒ scanned/handwritten)
   ▼
ParserRouter (PARSER_BACKEND=auto)
   ├─ printed pages      ─► Docling  (self-host)  or  LlamaParse (API)
   ├─ handwritten/scanned ─► Vision-LLM backend  (OpenAI/Anthropic/Azure/Ollama)
   └─ low-confidence pages ─► escalated to Vision-LLM backend
   ▼
ParseResult (per-page Markdown + confidence + handwriting flags)
   ▼
QAGenerator ─► grounded analyst Q&A (cited to source pages, JSON-validated)
   ▼
ProcessResponse  /  Markdown document
```

Key modules:

| Path | Responsibility |
|------|----------------|
| `app/parsers/router.py` | Page routing, fallback, confidence escalation |
| `app/parsers/vision_llm_parser.py` | Renders pages to images; reads handwriting |
| `app/parsers/docling_parser.py`, `llamaparse_parser.py` | Cheap printed-text backends |
| `app/llm/` | Provider abstraction + OpenAI/Anthropic/Azure/Ollama adapters |
| `app/qa/generator.py` | Pharma analyst & journalist Q&A (asker + reasoning + peer-comparison, anti-hallucination prompt + schema) |
| `app/utils/pdf.py` | PyMuPDF rendering + handwriting/scan heuristic |
| `app/main.py` | FastAPI service |

## Quickstart (Docker)

```bash
cp .env.example .env          # then add your API key(s)
docker compose up --build     # API on http://localhost:8000
```

Fully self-hosted (no external API), using Ollama for both vision and Q&A:

```bash
# set LLM_PROVIDER=ollama and VISION_PROVIDER=ollama in .env
docker compose --profile local-llm up --build
docker compose exec ollama ollama pull llama3.2-vision
docker compose exec ollama ollama pull llama3.3
```

## Quickstart (local, no Docker)

```bash
pip install -r requirements.txt
make samples                  # generate handwritten test PDFs into ./samples
uvicorn app.main:app --reload # http://localhost:8000/docs
```

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness |
| `GET`  | `/config` | Which backends/providers are usable with the current env |
| `POST` | `/parse` | PDF → structured Markdown (no LLM Q&A) |
| `POST` | `/process` | PDF → parse **+** analyst Q&A (JSON) |
| `POST` | `/process/markdown` | PDF → parse + Q&A, returns the Q&A as a Markdown file |
| `POST` | `/runs` | Start a **background** run; returns `{run_id}` (the UI's primary path) |
| `GET`  | `/runs` | List past runs (paginated) |
| `GET`  | `/runs/{id}` | Full run detail: parse, Q&A, timings, status, error |
| `GET`  | `/runs/{id}/logs` | Per-run structured logs (incremental via `?after=`) |
| `DELETE` | `/runs/{id}` | Delete a run and its logs/chat |
| `GET`  | `/runs/{id}/chat` | Interactive Q&A history for a run |
| `POST` | `/runs/{id}/ask` | Ask a **grounded** question about the parsed document |
| `POST` | `/agent/test` | Live connectivity check for the configured Q&A agent (returns ok/latency, no secrets) |
| `GET` / `PUT` | `/settings` | Read / persist operational settings **and provider API keys** (keys are write-only) |

Every call to the legacy `/parse` and `/process` endpoints is also recorded in
run history, so nothing is lost. The single-page console at `/` covers all of
this: **New Run** (mode, question count, and optional **peer companies**),
**Run History**, a per-run detail view (parsed output, analyst Q&A, an
interactive *Ask the document* chat with page citations, and a live **Logs**
console), and a **Settings** editor.

**Everything an operator needs is configurable from the UI Settings page** — log
level, default question count, default peer companies, provider/parser selection,
models, and the **provider API keys** themselves. Keys are write-only (stored
server-side, never sent back to the browser), and a **Test agent** button issues
a minimal live completion so you can confirm the agent works before running a job.

```bash
# Parse + generate 10 analyst Q&A pairs, get a Markdown brief back
curl -s -F "file=@samples/sample_q3_report_with_handwriting.pdf" \
        -F "num_questions=10" \
        -F "peers=Pfizer,Merck,Novartis" \
        http://localhost:8000/process/markdown
```

CLI equivalent:

```bash
python scripts/cli.py parse samples/sample_handwritten_board_notes.pdf
python scripts/cli.py qa    samples/sample_q3_report_with_handwriting.pdf -n 10 -o qa.md

# Bias the analyst peer-comparison questions toward named comparable companies:
python scripts/cli.py qa    samples/sample_q3_report_with_handwriting.pdf \
        -n 12 --peers "Pfizer,Merck,Novartis"
```

## Configuration

Everything is env-driven (see `.env.example`). The most important knobs:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `anthropic` | Provider for Q&A + handwriting reasoning (`openai`/`anthropic`/`azure`/`ollama`) |
| `VISION_PROVIDER` | = `LLM_PROVIDER` | Provider for page-image understanding |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Claude model for Q&A (text reasoning) |
| `ANTHROPIC_VISION_MODEL` | `claude-opus-4-8` | Claude model for handwriting/page-image reading |
| `OPENAI_MODEL` / `OPENAI_VISION_MODEL` | `gpt-5.5` | OpenAI model for Q&A / vision |
| `PARSER_BACKEND` | `auto` | `auto` router, or force `docling`/`llamaparse`/`vision_llm` |
| `PARSER_CONFIDENCE_THRESHOLD` | `0.62` | Cheap-backend pages below this are re-read by the vision LLM |
| `HANDWRITING_SCAN_CHAR_THRESHOLD` | `20` | Pages with fewer extractable text chars are treated as scan/handwriting candidates |
| `HANDWRITING_IMAGE_AREA_THRESHOLD` | `0.15` | Image coverage needed to route sparse-text pages to vision in `auto` |
| `HANDWRITING_TEXT_AREA_THRESHOLD` | `0.35` | Text coverage below this can route image-heavy pages to vision in `auto` |
| `FORCE_VISION_LLM` | `false` | Send every page through the vision LLM (max accuracy/cost) |
| `VISION_IMAGE_DETAIL` | `original` | `original` recommended for handwriting/low-quality scans |
| `VISION_RENDER_DPI` | `200` | Page render resolution for the vision backend |
| `VISION_MAX_CONCURRENCY` | `4` | Maximum simultaneous vision page requests per run |
| `DB_PATH` | `./data/app.db` | SQLite file for run history, logs, and Q&A chat |
| `DEFAULT_NUM_QUESTIONS` | `12` | Analyst-question count when a run omits it |

Operational knobs (everything except secrets) are also editable at runtime from
the **Settings** page and persisted to the DB — changes apply immediately with
no restart. API keys and endpoints stay env-only and are never written to or
returned from the store.

The service **degrades gracefully**: with no keys and no Docling installed, `/parse`
still returns native-extracted text and a warning. `GET /config` tells you exactly
what is active.

## Sample data

`make samples` generates two synthetic fixtures (license-clean, reproducible):

- `sample_q3_report_with_handwriting.pdf` — printed quarterly results **plus** CFO
  handwritten margin notes (the mixed printed+handwritten case).
- `sample_handwritten_board_notes.pdf` — a near-fully handwritten page (the
  scanned-notes case that defeats text-layer OCR).

Both are deliberately built so the router classifies them as handwritten and sends
them to the vision-LLM backend.

## Testing

```bash
make test    # 22 unit tests, fully offline (providers + backends mocked)
```

Tests cover the router's decision logic (printed→cheap, handwritten→vision,
low-confidence escalation, forced-vision, named-backend, native fallback), the
handwriting heuristic on real rendered PDFs, the vision parser's meta-parsing, the
provider factory's config validation, and Q&A generation/Markdown rendering.

## Production notes

- Stateless API; scale horizontally behind a load balancer. Vision calls are the
  latency/cost driver — `auto` mode minimizes them by only escalating pages that
  need it.
- Vision parsing uses bounded concurrency (semaphore) to respect provider rate
  limits; provider calls retry with exponential backoff (`tenacity`).
- Structured JSON logs (`structlog`) for ingestion into your log stack.
- Q&A answers are grounded: the model is instructed to answer only from the
  document, cite source pages, and caveat figures lifted from handwriting. Output
  is validated against a Pydantic schema; malformed items are dropped, not trusted.
- Secrets can be supplied via `.env` (git-ignored; `.env.example` documents every
  key) **or** set from the UI Settings page. UI-set keys are stored as write-only
  overrides in the SQLite store (`DB_PATH`) and are never returned to the client —
  secure that file and its host like any other credential store, or keep keys in
  `.env`/your secret manager if you prefer they never touch disk via the app.

## Sources

- [LlamaIndex — Best Document Parsing Software (legacy OCR → agentic AI)](https://www.llamaindex.ai/insights/best-document-parsing-software)
- [Firecrawl — Best PDF Parsers for AI and RAG Workflows in 2026](https://www.firecrawl.dev/blog/best-pdf-parsers)
- [Reducto — Docling vs LlamaParse vs Unstructured vs Reducto comparison](https://llms.reducto.ai/document-parser-comparison)
- [OpenAI Cookbook — Getting the Most out of GPT-5.5 for Vision & Document Understanding](https://developers.openai.com/cookbook/examples/multimodal/document_and_multimodal_understanding_tips)
- [LlamaIndex — Best Vision Language Models & Agentic OCR Tools](https://www.llamaindex.ai/insights/best-vision-language-models)
- [Applied AI — The State of PDF Parsing (800+ docs, 7 frontier LLMs)](https://www.applied-ai.com/briefings/pdf-parsing-benchmark/)
