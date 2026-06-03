"""FastAPI service.

Endpoints:
  GET  /                 — test UI (single-page console)
  GET  /health           — liveness + active config summary
  GET  /config           — which backends/providers are available right now
  GET  /samples          — list bundled sample PDFs
  POST /parse            — PDF -> structured markdown (no LLM Q&A)
  POST /parse/sample     — parse a bundled sample by name (no LLM Q&A)
  POST /process          — PDF -> parse + analyst Q&A (JSON)
  POST /process/sample   — parse + Q&A for a bundled sample by name
  POST /process/markdown — PDF -> parse + Q&A, returns the Q&A as a Markdown file
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from .config import get_settings
from .llm.factory import build_llm_provider, build_vision_provider
from .logging_conf import configure_logging, get_logger
from .parsers.docling_parser import DoclingParser
from .parsers.llamaparse_parser import LlamaParseParser
from .pipeline import Pipeline
from .qa.generator import QAGenerator
from .samples import list_samples, read_sample
from .schemas import ProcessResponse

STATIC_DIR = Path(__file__).resolve().parent / "static"

settings = get_settings()
configure_logging(settings.log_level)
log = get_logger("api")

app = FastAPI(
    title="Intelligent PDF Parser",
    version="0.1.0",
    description="Multi-backend, multi-LLM PDF understanding with handwriting support "
    "and analyst Q&A generation.",
)


def _check_size(raw: bytes) -> None:
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit")


async def _read_pdf(file: UploadFile) -> bytes:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are accepted")
    raw = await file.read()
    _check_size(raw)
    if not raw.startswith(b"%PDF"):
        raise HTTPException(400, "File does not look like a valid PDF")
    return raw


@app.get("/", include_in_schema=False)
async def ui() -> FileResponse:
    """Serve the single-page test console."""
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(404, "UI not built")
    return FileResponse(index)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": app.version, "env": settings.app_env}


@app.get("/config")
async def config() -> dict:
    """Report which backends/providers are usable with the current env."""
    def _provider_ok(builder) -> bool:
        try:
            builder(settings)
            return True
        except Exception:
            return False

    return {
        "parser_backend": settings.parser_backend,
        "force_vision_llm": settings.force_vision_llm,
        "confidence_threshold": settings.parser_confidence_threshold,
        "backends_available": {
            "docling": DoclingParser().available(),
            "llamaparse": LlamaParseParser().available(),
            "vision_llm": _provider_ok(build_vision_provider),
        },
        "llm_provider": settings.llm_provider,
        "llm_provider_available": _provider_ok(build_llm_provider),
        "vision_provider": settings.effective_vision_provider,
    }


@app.get("/samples")
async def samples() -> dict:
    """List bundled sample PDFs available to parse without uploading."""
    return {"samples": list_samples()}


def _load_sample(name: str) -> bytes:
    try:
        raw = read_sample(name)
    except FileNotFoundError:
        raise HTTPException(404, f"Sample not found: {name}") from None
    _check_size(raw)
    return raw


@app.post("/parse", response_model=ProcessResponse)
async def parse(file: UploadFile = File(...)) -> ProcessResponse:
    raw = await _read_pdf(file)
    return await Pipeline(settings).parse_only(raw, file.filename or "upload.pdf")


@app.post("/parse/sample", response_model=ProcessResponse)
async def parse_sample(name: str = Form(...)) -> ProcessResponse:
    raw = _load_sample(name)
    return await Pipeline(settings).parse_only(raw, name)


@app.post("/process/sample", response_model=ProcessResponse)
async def process_sample(
    name: str = Form(...),
    num_questions: int = Form(12),
) -> ProcessResponse:
    raw = _load_sample(name)
    try:
        return await Pipeline(settings).process(raw, name, num_questions=num_questions)
    except Exception as exc:
        log.error("process_sample_failed", error=str(exc))
        raise HTTPException(502, f"Processing failed: {exc}") from exc


@app.post("/process", response_model=ProcessResponse)
async def process(
    file: UploadFile = File(...),
    num_questions: int = Form(12),
) -> ProcessResponse:
    raw = await _read_pdf(file)
    try:
        return await Pipeline(settings).process(
            raw, file.filename or "upload.pdf", num_questions=num_questions
        )
    except Exception as exc:  # surface config/provider errors cleanly
        log.error("process_failed", error=str(exc))
        raise HTTPException(502, f"Processing failed: {exc}") from exc


@app.post("/process/markdown", response_class=PlainTextResponse)
async def process_markdown(
    file: UploadFile = File(...),
    num_questions: int = Form(12),
) -> str:
    raw = await _read_pdf(file)
    resp = await Pipeline(settings).process(
        raw, file.filename or "upload.pdf", num_questions=num_questions
    )
    if resp.qa is None:
        raise HTTPException(502, "Q&A generation produced no output")
    return QAGenerator.to_markdown(resp.qa)
