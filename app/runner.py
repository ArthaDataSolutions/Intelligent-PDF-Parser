"""Run orchestration with persistence and per-run logging.

A "run" is one parse (or parse + Q&A) of a document, recorded in the run
history with its inputs, outputs, timings, logs, and any error. Two entry
points share the same core:

  * :meth:`RunService.start_background` — fire-and-forget; returns a run id
    immediately and executes on the event loop. The UI polls for progress.
  * :meth:`RunService.run_inline` — awaits completion and returns the
    ``ProcessResponse`` (used by the legacy synchronous endpoints so their
    behaviour is unchanged while still recording history).
"""
from __future__ import annotations

import asyncio
import time

from .config import Settings, get_active_settings
from .logging_conf import get_logger, run_context
from .pipeline import Pipeline
from .schemas import ProcessResponse
from .storage import RunRepository, get_repository
from .storage.repository import STATUS_COMPLETED, STATUS_FAILED, STATUS_RUNNING

log = get_logger("runner")

MODE_PARSE = "parse"
MODE_PROCESS = "process"


class RunService:
    def __init__(self, repo: RunRepository | None = None) -> None:
        self._repo = repo or get_repository(get_active_settings().db_path)

    @property
    def repo(self) -> RunRepository:
        return self._repo

    def _create(
        self, *, mode: str, source: str, filename: str,
        num_questions: int | None, settings: Settings,
    ) -> str:
        return self._repo.create_run(
            mode=mode,
            source=source,
            filename=filename,
            num_questions=num_questions if mode == MODE_PROCESS else None,
            parser_backend=settings.parser_backend,
            llm_provider=settings.llm_provider,
            vision_provider=settings.effective_vision_provider,
        )

    async def _execute(
        self, run_id: str, settings: Settings, doc_bytes: bytes,
        filename: str, mode: str, num_questions: int,
    ) -> ProcessResponse:
        """Run the pipeline, persisting status/logs/results. Re-raises on error."""
        with run_context(run_id):
            t0 = time.perf_counter()
            self._repo.update_run(run_id, status=STATUS_RUNNING)
            log.info("run_started", mode=mode, filename=filename,
                     parser_backend=settings.parser_backend)
            try:
                pipeline = Pipeline(settings)
                if mode == MODE_PROCESS:
                    log.info("parsing_and_qa", num_questions=num_questions)
                    resp = await pipeline.process(
                        doc_bytes, filename, num_questions=num_questions
                    )
                else:
                    log.info("parsing")
                    resp = await pipeline.parse_only(doc_bytes, filename)

                duration = round((time.perf_counter() - t0) * 1000, 1)
                self._repo.update_run(
                    run_id,
                    status=STATUS_COMPLETED,
                    finished_at=_now_iso(),
                    duration_ms=duration,
                    num_pages=resp.parse.page_count,
                    markdown=resp.parse.markdown,
                    parse_json=resp.parse.model_dump_json(),
                    qa_json=resp.qa.model_dump_json() if resp.qa else None,
                    timings_json=_json(resp.timings_ms),
                    meta_json=_json(resp.meta),
                )
                log.info("run_completed", pages=resp.parse.page_count,
                         duration_ms=duration,
                         backends=resp.parse.backend_summary)
                return resp
            except Exception as exc:
                duration = round((time.perf_counter() - t0) * 1000, 1)
                self._repo.update_run(
                    run_id,
                    status=STATUS_FAILED,
                    finished_at=_now_iso(),
                    duration_ms=duration,
                    error=str(exc),
                )
                log.error("run_failed", error=str(exc))
                raise

    def start_background(
        self, doc_bytes: bytes, filename: str, *,
        mode: str, source: str, num_questions: int,
    ) -> str:
        settings = get_active_settings()
        run_id = self._create(
            mode=mode, source=source, filename=filename,
            num_questions=num_questions, settings=settings,
        )

        async def _runner() -> None:
            try:
                await self._execute(
                    run_id, settings, doc_bytes, filename, mode, num_questions
                )
            except Exception:  # already recorded on the run; don't crash the loop
                pass

        asyncio.create_task(_runner())
        return run_id

    async def run_inline(
        self, doc_bytes: bytes, filename: str, *,
        mode: str, source: str, num_questions: int,
    ) -> tuple[str, ProcessResponse]:
        settings = get_active_settings()
        run_id = self._create(
            mode=mode, source=source, filename=filename,
            num_questions=num_questions, settings=settings,
        )
        resp = await self._execute(
            run_id, settings, doc_bytes, filename, mode, num_questions
        )
        return run_id, resp


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _json(obj: object) -> str:
    import json

    return json.dumps(obj, default=str)
