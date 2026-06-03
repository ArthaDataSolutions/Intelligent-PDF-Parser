"""Unit tests for the SQLite run repository."""
from __future__ import annotations

import pytest

from app.storage.repository import (
    STATUS_COMPLETED,
    STATUS_QUEUED,
    RunRepository,
)


@pytest.fixture
def repo(tmp_path) -> RunRepository:
    return RunRepository(str(tmp_path / "store.db"))


def _new_run(repo: RunRepository, filename: str = "r.pdf") -> str:
    return repo.create_run(
        mode="process", source="upload", filename=filename, num_questions=5,
        parser_backend="auto", llm_provider="anthropic", vision_provider="anthropic",
    )


def test_create_run_starts_queued(repo):
    run_id = _new_run(repo)
    run = repo.get_run(run_id)
    assert run["status"] == STATUS_QUEUED
    assert run["filename"] == "r.pdf"
    assert run["num_questions"] == 5
    assert run["has_markdown"] is False


def test_update_run_persists_fields_and_parses_json(repo):
    run_id = _new_run(repo)
    repo.update_run(
        run_id, status=STATUS_COMPLETED, num_pages=3,
        markdown="page text", parse_json='{"page_count": 3}',
        qa_json='{"items": []}', timings_json='{"parse": 12.0}',
    )
    run = repo.get_run(run_id, include_markdown=True)
    assert run["status"] == STATUS_COMPLETED
    assert run["num_pages"] == 3
    assert run["parse"] == {"page_count": 3}
    assert run["qa"] == {"items": []}
    assert run["timings_ms"] == {"parse": 12.0}
    assert run["markdown"] == "page text"
    assert run["has_markdown"] is True


def test_get_run_excludes_markdown_by_default(repo):
    run_id = _new_run(repo)
    repo.update_run(run_id, markdown="secret big text")
    run = repo.get_run(run_id)
    assert "markdown" not in run
    assert run["has_markdown"] is True
    assert repo.get_markdown(run_id) == "secret big text"


def test_list_runs_newest_first(repo):
    first = _new_run(repo, "a.pdf")
    second = _new_run(repo, "b.pdf")
    runs = repo.list_runs()
    assert [r["id"] for r in runs] == [second, first]
    assert repo.count_runs() == 2


def test_logs_are_sequential_and_incremental(repo):
    run_id = _new_run(repo)
    repo.append_log(run_id, "info", "started")
    repo.append_log(run_id, "info", "parsing", {"pages": 2})
    logs = repo.get_logs(run_id)
    assert [l["seq"] for l in logs] == [1, 2]
    assert logs[1]["data"] == {"pages": 2}
    # Incremental fetch returns only newer lines.
    assert repo.get_logs(run_id, after_seq=1) == logs[1:]


def test_chat_roundtrip(repo):
    run_id = _new_run(repo)
    repo.add_chat_message(run_id, "user", "What is revenue?")
    repo.add_chat_message(run_id, "assistant", "$5M", sources=[2, 3])
    msgs = repo.get_chat(run_id)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["sources"] == [2, 3]


def test_delete_cascades(repo):
    run_id = _new_run(repo)
    repo.append_log(run_id, "info", "x")
    repo.add_chat_message(run_id, "user", "hi")
    assert repo.delete_run(run_id) is True
    assert repo.get_run(run_id) is None
    assert repo.get_logs(run_id) == []
    assert repo.get_chat(run_id) == []
    assert repo.delete_run(run_id) is False


def test_overrides_set_merge_and_clear(repo):
    repo.set_overrides({"llm_provider": "openai", "vision_render_dpi": 300})
    assert repo.get_overrides() == {"llm_provider": "openai", "vision_render_dpi": 300}
    repo.set_overrides({"vision_render_dpi": None})  # None clears the key
    assert repo.get_overrides() == {"llm_provider": "openai"}
