"""Integration tests for the run service (inline path, offline)."""
from __future__ import annotations

import pytest

from app.config import get_active_settings
from app.runner import MODE_PARSE, RunService
from app.storage import get_repository


@pytest.mark.asyncio
async def test_run_inline_records_completed_run(text_pdf):
    svc = RunService()
    run_id, resp = await svc.run_inline(
        text_pdf, "report.pdf", mode=MODE_PARSE, source="upload", num_questions=0,
    )
    assert resp.parse.page_count == 2

    repo = get_repository(get_active_settings().db_path)
    run = repo.get_run(run_id)
    assert run["status"] == "completed"
    assert run["num_pages"] == 2
    assert run["mode"] == "parse"
    assert run["duration_ms"] is not None
    assert run["has_markdown"] is True
    # The parsed text is persisted for later grounded Q&A.
    assert repo.get_markdown(run_id)


@pytest.mark.asyncio
async def test_run_inline_captures_per_run_logs(text_pdf):
    svc = RunService()
    run_id, _ = await svc.run_inline(
        text_pdf, "report.pdf", mode=MODE_PARSE, source="upload", num_questions=0,
    )
    repo = get_repository(get_active_settings().db_path)
    events = [l["event"] for l in repo.get_logs(run_id)]
    assert "run_started" in events
    assert "run_completed" in events
