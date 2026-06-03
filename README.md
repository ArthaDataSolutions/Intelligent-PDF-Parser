# Intelligent PDF Parser

A production-grade service that turns financial-report PDFs — including ones with
**handwritten notes and annotations** — into clean structured Markdown, then uses
an LLM to generate an **analyst-style Q&A document** a CFO can use to prepare for
tough questions from journalists and sell-side analysts (WSJ, NYT, etc.).

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
handwriting reliably; Gemini-class and GPT-5.4-class models lead on raw accuracy
while LlamaParse hits the cost/quality sweet spot (~$0.003/page) for clean printed
docs. **Docling (IBM, open-source) is strong on tables/layout and fully
self-hostable but does not handle handwriting.** Frontier multimodal models
(GPT-5.4, Claude) can now read dense scans and handwritten forms in a single pass,
and vendor guidance is explicit that for handwriting / tiny text / low-quality
scans you should send the page image at full ("original") detail and keep
transcription temperature at 0. Those findings are baked directly into the router
and the vision prompt. Sources are listed at the bottom of this file.

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
| `app/qa/generator.py` | Analyst Q&A generation (anti-hallucination prompt + schema) |
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

```bash
# Parse + generate 10 analyst Q&A pairs, get a Markdown brief back
curl -s -F "file=@samples/sample_q3_report_with_handwriting.pdf" \
        -F "num_questions=10" \
        http://localhost:8000/process/markdown
```

CLI equivalent:

```bash
python scripts/cli.py parse samples/sample_handwritten_board_notes.pdf
python scripts/cli.py qa    samples/sample_q3_report_with_handwriting.pdf -n 10 -o qa.md
```

## Configuration

Everything is env-driven (see `.env.example`). The most important knobs:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `anthropic` | Provider for Q&A + handwriting reasoning (`openai`/`anthropic`/`azure`/`ollama`) |
| `VISION_PROVIDER` | = `LLM_PROVIDER` | Provider for page-image understanding |
| `PARSER_BACKEND` | `auto` | `auto` router, or force `docling`/`llamaparse`/`vision_llm` |
| `PARSER_CONFIDENCE_THRESHOLD` | `0.62` | Cheap-backend pages below this are re-read by the vision LLM |
| `FORCE_VISION_LLM` | `false` | Send every page through the vision LLM (max accuracy/cost) |
| `VISION_IMAGE_DETAIL` | `original` | `original` recommended for handwriting/low-quality scans |
| `VISION_RENDER_DPI` | `200` | Page render resolution for the vision backend |

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
- Secrets live only in `.env` (git-ignored). `.env.example` documents every key.

## Sources

- [LlamaIndex — Best Document Parsing Software (legacy OCR → agentic AI)](https://www.llamaindex.ai/insights/best-document-parsing-software)
- [Firecrawl — Best PDF Parsers for AI and RAG Workflows in 2026](https://www.firecrawl.dev/blog/best-pdf-parsers)
- [Reducto — Docling vs LlamaParse vs Unstructured vs Reducto comparison](https://llms.reducto.ai/document-parser-comparison)
- [OpenAI Cookbook — Getting the Most out of GPT-5.4 for Vision & Document Understanding](https://developers.openai.com/cookbook/examples/multimodal/document_and_multimodal_understanding_tips)
- [LlamaIndex — Best Vision Language Models & Agentic OCR Tools](https://www.llamaindex.ai/insights/best-vision-language-models)
- [Applied AI — The State of PDF Parsing (800+ docs, 7 frontier LLMs)](https://www.applied-ai.com/briefings/pdf-parsing-benchmark/)
