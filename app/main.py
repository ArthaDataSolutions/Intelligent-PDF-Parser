"""FastAPI service.

Legacy synchronous endpoints (unchanged response shapes; now also recorded in
run history):
  GET  /                 — single-page app (history, run detail, settings, chat)
  GET  /health           — liveness + version
  GET  /config           — active backends/providers (no secrets)
  GET  /samples          — bundled sample PDFs
  POST /parse            — PDF -> structured markdown (no LLM Q&A)
  POST /parse/sample     — parse a bundled sample by name
  POST /process          — PDF -> parse + analyst Q&A (JSON)
  POST /process/sample   — parse + Q&A for a bundled sample
  POST /process/markdown — parse + Q&A as a Markdown file

Run history / logs / interactive Q&A (the production surface the UI uses):
  POST   /runs               — start a background run, returns {run_id}
  GET    /runs               — list past runs (paginated)
  GET    /runs/{id}          — full run detail (parse, qa, timings, status)
  GET    /runs/{id}/logs     — per-run log lines (incremental via ?after=)
  DELETE /runs/{id}          — delete a run and its logs/chat
  GET    /runs/{id}/chat     — interactive Q&A history for a run
  POST   /runs/{id}/ask      — ask a grounded question about the parsed document
  GET    /settings           — editable settings + current values
  PUT    /settings           — persist setting overrides (no secrets)
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from .config import (
    OVERRIDE_WHITELIST,
    SECRET_OVERRIDE_KEYS,
    Settings,
    get_active_settings,
    get_settings,
)
from .llm.base import LLMMessage
from .llm.factory import build_llm_provider, build_vision_provider
from .logging_conf import configure_logging, get_logger
from .parsers.docling_parser import DoclingParser
from .parsers.llamaparse_parser import LlamaParseParser
from .qa.chat import DocChat
from .qa.generator import QAGenerator
from .runner import MODE_PARSE, MODE_PROCESS, RunService
from .samples import list_samples, read_sample
from .schemas import ProcessResponse
from .storage import get_repository
from .storage.repository import STATUS_COMPLETED

STATIC_DIR = Path(__file__).resolve().parent / "static"

settings = get_settings()
configure_logging(settings.log_level)
log = get_logger("api")

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    configure_logging(get_active_settings().log_level)
    yield


app = FastAPI(
    title="Intelligent PDF Parser",
    version="0.2.0",
    description="Multi-backend, multi-LLM PDF understanding with handwriting "
    "support, analyst Q&A, run history, and interactive document chat.",
    lifespan=_lifespan,
)


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _repo():
    # db_path is intentionally not overridable from the UI, so the env value is
    # the single source of truth for where the store lives.
    return get_repository(get_settings().db_path)


def _service() -> RunService:
    return RunService(_repo())


# --------------------------------------------------------------------------- #
# Upload validation
# --------------------------------------------------------------------------- #
def _check_size(raw: bytes) -> None:
    limit = get_active_settings().max_upload_mb
    if len(raw) > limit * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {limit} MB limit")


async def _read_pdf(file: UploadFile) -> bytes:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are accepted")
    raw = await file.read()
    _check_size(raw)
    if not raw.startswith(b"%PDF"):
        raise HTTPException(400, "File does not look like a valid PDF")
    return raw


def _resolve_peers(raw: str | None) -> list[str]:
    """Comma-separated peers from the request, falling back to the saved default."""
    text = raw if raw is not None else get_active_settings().default_peers
    return [p.strip() for p in (text or "").split(",") if p.strip()]


def _load_sample(name: str) -> bytes:
    try:
        raw = read_sample(name)
    except FileNotFoundError:
        raise HTTPException(404, f"Sample not found: {name}") from None
    _check_size(raw)
    return raw


# --------------------------------------------------------------------------- #
# UI + meta
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
async def ui() -> FileResponse:
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(404, "UI not built")
    return FileResponse(index)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": app.version, "env": settings.app_env}


def _provider_models(s: Settings) -> dict[str, dict[str, str]]:
    return {
        "openai": {"text": s.openai_model, "vision": s.openai_vision_model},
        "anthropic": {"text": s.anthropic_model, "vision": s.anthropic_vision_model},
        "azure": {
            "text": s.azure_openai_deployment or "—",
            "vision": s.azure_openai_vision_deployment or s.azure_openai_deployment or "—",
        },
        "ollama": {"text": s.ollama_model, "vision": s.ollama_vision_model},
    }


def _key_status(s: Settings) -> dict[str, bool]:
    return {
        "openai": bool(s.openai_api_key),
        "anthropic": bool(s.anthropic_api_key),
        "azure": bool(s.azure_openai_api_key and s.azure_openai_endpoint),
        "ollama": True,
        "llama_cloud": bool(s.llama_cloud_api_key),
    }


def _backends_available(s: Settings) -> dict[str, bool]:
    def _ok(builder) -> bool:
        try:
            builder(s)
            return True
        except Exception:
            return False

    return {
        "docling": DoclingParser().available(),
        "llamaparse": LlamaParseParser().available(),
        "vision_llm": _ok(build_vision_provider),
    }


@app.get("/config")
async def config() -> dict:
    """Report every backend/provider setting active for the current env.

    Reflects persisted UI overrides. Secrets are never returned — only booleans.
    """
    s = get_active_settings()

    def _provider_ok(builder) -> bool:
        try:
            builder(s)
            return True
        except Exception:
            return False

    models = _provider_models(s)
    llm = s.llm_provider
    vis = s.effective_vision_provider
    backends = _backends_available(s)

    return {
        "app": {
            "env": s.app_env,
            "log_level": s.log_level,
            "max_upload_mb": s.max_upload_mb,
        },
        "parsing": {
            "parser_backend": s.parser_backend,
            "force_vision_llm": s.force_vision_llm,
            "confidence_threshold": s.parser_confidence_threshold,
            "handwriting_scan_char_threshold": s.handwriting_scan_char_threshold,
            "handwriting_image_area_threshold": s.handwriting_image_area_threshold,
            "handwriting_text_area_threshold": s.handwriting_text_area_threshold,
            "vision_image_detail": s.vision_image_detail,
            "vision_render_dpi": s.vision_render_dpi,
            "vision_max_concurrency": s.vision_max_concurrency,
        },
        "backends_available": backends,
        "providers": {
            "llm": {
                "name": llm,
                "model": models[llm]["text"],
                "available": _provider_ok(build_llm_provider),
            },
            "vision": {
                "name": vis,
                "model": models[vis]["vision"],
                "available": _provider_ok(build_vision_provider),
            },
        },
        "models": models,
        "keys_present": _key_status(s),
        # Back-compat flat fields (kept for existing clients/tests):
        "parser_backend": s.parser_backend,
        "force_vision_llm": s.force_vision_llm,
        "confidence_threshold": s.parser_confidence_threshold,
        "llm_provider": llm,
        "llm_provider_available": _provider_ok(build_llm_provider),
        "vision_provider": vis,
    }


@app.get("/samples")
async def samples() -> dict:
    return {"samples": list_samples()}


@app.post("/agent/test")
async def agent_test() -> dict:
    """Live connectivity check for the configured Q&A agent (LLM provider).

    Builds the active LLM provider and issues a minimal chat completion so the
    user can confirm — from the UI — that their provider selection, model, and
    API key actually work, with a round-trip latency. No secrets are returned.
    """
    s = get_active_settings()
    model = _provider_models(s)[s.llm_provider]["text"]
    base = {"provider": s.llm_provider, "model": model}
    try:
        provider = build_llm_provider(s)
    except Exception as exc:
        return {**base, "ok": False, "stage": "config", "error": str(exc)}

    t0 = time.perf_counter()
    try:
        reply = await provider.chat(
            [LLMMessage(role="user", content="Reply with the single word: OK")],
            temperature=0.0,
            max_tokens=16,
        )
        latency = round((time.perf_counter() - t0) * 1000, 1)
        log.info("agent_test_ok", provider=s.llm_provider, latency_ms=latency)
        return {
            **base,
            "ok": True,
            "latency_ms": latency,
            "reply": (reply or "").strip()[:200],
        }
    except Exception as exc:
        log.error("agent_test_failed", provider=s.llm_provider, error=str(exc))
        return {**base, "ok": False, "stage": "call", "error": str(exc)}
    finally:
        await provider.aclose()


# --------------------------------------------------------------------------- #
# Legacy synchronous endpoints (record history via run_inline)
# --------------------------------------------------------------------------- #
@app.post("/parse", response_model=ProcessResponse)
async def parse(file: UploadFile = File(...)) -> ProcessResponse:
    raw = await _read_pdf(file)
    _, resp = await _service().run_inline(
        raw, file.filename or "upload.pdf",
        mode=MODE_PARSE, source="upload", num_questions=0,
    )
    return resp


@app.post("/parse/sample", response_model=ProcessResponse)
async def parse_sample(name: str = Form(...)) -> ProcessResponse:
    raw = _load_sample(name)
    _, resp = await _service().run_inline(
        raw, name, mode=MODE_PARSE, source="sample", num_questions=0,
    )
    return resp


@app.post("/process", response_model=ProcessResponse)
async def process(
    file: UploadFile = File(...),
    num_questions: int = Form(12),
    peers: str | None = Form(None),
) -> ProcessResponse:
    raw = await _read_pdf(file)
    try:
        _, resp = await _service().run_inline(
            raw, file.filename or "upload.pdf",
            mode=MODE_PROCESS, source="upload", num_questions=num_questions,
            peers=_resolve_peers(peers),
        )
        return resp
    except Exception as exc:
        log.error("process_failed", error=str(exc))
        raise HTTPException(502, f"Processing failed: {exc}") from exc


@app.post("/process/sample", response_model=ProcessResponse)
async def process_sample(
    name: str = Form(...),
    num_questions: int = Form(12),
    peers: str | None = Form(None),
) -> ProcessResponse:
    raw = _load_sample(name)
    try:
        _, resp = await _service().run_inline(
            raw, name, mode=MODE_PROCESS, source="sample",
            num_questions=num_questions, peers=_resolve_peers(peers),
        )
        return resp
    except Exception as exc:
        log.error("process_sample_failed", error=str(exc))
        raise HTTPException(502, f"Processing failed: {exc}") from exc


@app.post("/process/markdown", response_class=PlainTextResponse)
async def process_markdown(
    file: UploadFile = File(...),
    num_questions: int = Form(12),
    peers: str | None = Form(None),
) -> str:
    raw = await _read_pdf(file)
    _, resp = await _service().run_inline(
        raw, file.filename or "upload.pdf",
        mode=MODE_PROCESS, source="upload", num_questions=num_questions,
        peers=_resolve_peers(peers),
    )
    if resp.qa is None:
        raise HTTPException(502, "Q&A generation produced no output")
    return QAGenerator.to_markdown(resp.qa)


# --------------------------------------------------------------------------- #
# Run history API
# --------------------------------------------------------------------------- #
@app.post("/runs", status_code=202)
async def start_run(
    source: str = Form("upload"),
    name: str | None = Form(None),
    file: UploadFile | None = File(None),
    mode: str = Form(MODE_PROCESS),
    num_questions: int | None = Form(None),
    peers: str | None = Form(None),
) -> dict:
    if mode not in (MODE_PARSE, MODE_PROCESS):
        raise HTTPException(400, "mode must be 'parse' or 'process'")
    s = get_active_settings()
    nq = num_questions or s.default_num_questions
    peer_list = _resolve_peers(peers)

    if source == "sample":
        if not name:
            raise HTTPException(400, "name is required for sample runs")
        raw = _load_sample(name)
        filename = name
    elif source == "upload":
        if file is None:
            raise HTTPException(400, "file is required for upload runs")
        raw = await _read_pdf(file)
        filename = file.filename or "upload.pdf"
    else:
        raise HTTPException(400, "source must be 'sample' or 'upload'")

    run_id = _service().start_background(
        raw, filename, mode=mode, source=source, num_questions=nq,
        peers=peer_list,
    )
    return {"run_id": run_id, "status": "queued"}


@app.get("/runs")
async def list_runs(limit: int = 50, offset: int = 0) -> dict:
    limit = max(1, min(limit, 200))
    repo = _repo()
    return {"runs": repo.list_runs(limit=limit, offset=offset),
            "total": repo.count_runs()}


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    run = _repo().get_run(run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    return run


@app.get("/runs/{run_id}/logs")
async def get_run_logs(run_id: str, after: int = 0) -> dict:
    repo = _repo()
    run = repo.get_run(run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    return {"status": run["status"], "logs": repo.get_logs(run_id, after_seq=after)}


@app.delete("/runs/{run_id}")
async def delete_run(run_id: str) -> dict:
    if not _repo().delete_run(run_id):
        raise HTTPException(404, "Run not found")
    return {"deleted": True}


@app.get("/runs/{run_id}/chat")
async def get_chat(run_id: str) -> dict:
    repo = _repo()
    if repo.get_run(run_id) is None:
        raise HTTPException(404, "Run not found")
    return {"messages": repo.get_chat(run_id)}


class AskRequest(BaseModel):
    question: str


@app.post("/runs/{run_id}/ask")
async def ask(run_id: str, body: AskRequest) -> dict:
    repo = _repo()
    run = repo.get_run(run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    if run["status"] != STATUS_COMPLETED:
        raise HTTPException(409, "Run is not complete yet")

    question = body.question.strip()
    if not question:
        raise HTTPException(400, "question must not be empty")

    markdown = repo.get_markdown(run_id)
    if not markdown:
        raise HTTPException(409, "No parsed text is available for this run")

    prior = repo.get_chat(run_id)
    history = [LLMMessage(role=m["role"], content=m["content"]) for m in prior][-8:]

    repo.add_chat_message(run_id, "user", question)

    s = get_active_settings()
    try:
        provider = build_llm_provider(s)
    except Exception as exc:
        raise HTTPException(502, f"LLM provider unavailable: {exc}") from exc
    try:
        answer = await DocChat(provider).answer(markdown, question, history=history)
    except Exception as exc:
        log.error("ask_failed", run_id=run_id, error=str(exc))
        raise HTTPException(502, f"Q&A failed: {exc}") from exc
    finally:
        await provider.aclose()

    return repo.add_chat_message(
        run_id, "assistant", answer.answer, answer.source_pages
    )


# --------------------------------------------------------------------------- #
# Settings API
# --------------------------------------------------------------------------- #
_PROVIDER_OPTIONS = ["openai", "anthropic", "azure", "ollama"]

_SETTINGS_SCHEMA: dict[str, dict[str, Any]] = {
    "log_level": {"type": "enum", "options": ["DEBUG", "INFO", "WARNING", "ERROR"]},
    "max_upload_mb": {"type": "int", "min": 1, "max": 500},
    "default_num_questions": {"type": "int", "min": 1, "max": 40},
    "default_peers": {"type": "str"},
    "peer_web_search": {"type": "bool"},
    "peer_web_search_max_uses": {"type": "int", "min": 1, "max": 20},
    "llm_provider": {"type": "enum", "options": _PROVIDER_OPTIONS},
    "vision_provider": {"type": "enum", "options": _PROVIDER_OPTIONS, "nullable": True},
    "parser_backend": {
        "type": "enum",
        "options": ["auto", "docling", "llamaparse", "vision_llm"],
    },
    "force_vision_llm": {"type": "bool"},
    "parser_confidence_threshold": {"type": "float", "min": 0.0, "max": 1.0},
    "handwriting_scan_char_threshold": {"type": "int", "min": 0, "max": 2000},
    "handwriting_image_area_threshold": {"type": "float", "min": 0.0, "max": 1.0},
    "handwriting_text_area_threshold": {"type": "float", "min": 0.0, "max": 1.0},
    "vision_image_detail": {"type": "enum", "options": ["auto", "original"]},
    "vision_render_dpi": {"type": "int", "min": 72, "max": 600},
    "vision_max_concurrency": {"type": "int", "min": 1, "max": 16},
    "openai_model": {"type": "str"},
    "openai_vision_model": {"type": "str"},
    "anthropic_model": {"type": "str"},
    "anthropic_vision_model": {"type": "str"},
    "ollama_base_url": {"type": "str"},
    "ollama_model": {"type": "str"},
    "ollama_vision_model": {"type": "str"},
    "azure_openai_deployment": {"type": "str", "nullable": True},
    "azure_openai_vision_deployment": {"type": "str", "nullable": True},
    "azure_openai_endpoint": {"type": "str", "nullable": True},
    "azure_openai_api_version": {"type": "str"},
    # Write-only credentials (rendered as password inputs; never returned).
    "openai_api_key": {"type": "secret"},
    "anthropic_api_key": {"type": "secret"},
    "azure_openai_api_key": {"type": "secret"},
    "llama_cloud_api_key": {"type": "secret"},
}


def _settings_view() -> dict:
    s = get_active_settings()
    overrides = _repo().get_overrides()
    values = {k: getattr(s, k) for k in OVERRIDE_WHITELIST}
    return {
        "values": values,
        "overridden": sorted(
            k for k in overrides if k in OVERRIDE_WHITELIST | SECRET_OVERRIDE_KEYS
        ),
        "schema": _SETTINGS_SCHEMA,
        # Secret credentials are write-only: never echo their values, only the
        # field names (so the UI can render inputs) and the present/absent flags.
        "secret_keys": sorted(SECRET_OVERRIDE_KEYS),
        "keys_present": _key_status(s),
        "backends_available": _backends_available(s),
    }


@app.get("/settings")
async def get_settings_view() -> dict:
    return _settings_view()


class SettingsUpdate(BaseModel):
    values: dict[str, Any]


@app.put("/settings")
async def update_settings(body: SettingsUpdate) -> dict:
    editable = OVERRIDE_WHITELIST | SECRET_OVERRIDE_KEYS
    invalid = [k for k in body.values if k not in editable]
    if invalid:
        raise HTTPException(400, f"Not editable: {', '.join(sorted(invalid))}")

    # Blank secret values mean "clear this override" (fall back to env), not
    # "store an empty key" — coerce them to None before validating/persisting.
    values = dict(body.values)
    for key in SECRET_OVERRIDE_KEYS:
        if key in values and not (values[key] or "").strip():
            values[key] = None

    # Validate the merged result before persisting anything.
    candidate = get_active_settings().model_dump()
    candidate.update(values)
    try:
        Settings(**candidate)
    except ValidationError as exc:
        raise HTTPException(422, f"Invalid settings: {exc}") from exc

    _repo().set_overrides(values)
    if "log_level" in values and values["log_level"]:
        configure_logging(str(values["log_level"]))
    # Never log secret values — only which keys changed.
    log.info("settings_updated", keys=sorted(values.keys()))
    return _settings_view()
