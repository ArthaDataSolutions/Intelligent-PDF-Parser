"""API tests for run history, logs, settings, and interactive Q&A.

These drive the *inline* path (``/parse/sample`` → ``run_inline``) to produce
deterministic, offline, completed runs, then exercise the history/logs/chat/ask
surface against them. Docling handles the printed test PDF with no LLM call, so
no network or API key is involved.

The background ``POST /runs`` endpoint is verified for acceptance + recording
only; completion is not awaited here because Starlette's TestClient does not
keep the event loop running between synchronous requests (it works under real
uvicorn). End-to-end background completion is covered by ``test_runner.py``.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import samples
from app.main import app
from tests.conftest import FakeLLM


@pytest.fixture
def client(tmp_path, monkeypatch, text_pdf):
    d = tmp_path / "samples"
    d.mkdir()
    (d / "report.pdf").write_bytes(text_pdf)
    monkeypatch.setattr(samples, "SAMPLES_DIR", d.resolve())
    return TestClient(app)


def _make_completed_run(client: TestClient) -> str:
    """Create a completed run via the inline parse endpoint; return its id."""
    r = client.post("/parse/sample", data={"name": "report.pdf"})
    assert r.status_code == 200
    runs = client.get("/runs").json()["runs"]
    assert runs, "expected at least one recorded run"
    return runs[0]["id"]  # newest first


def test_inline_parse_is_recorded_as_completed_run(client):
    run_id = _make_completed_run(client)
    run = client.get(f"/runs/{run_id}").json()
    assert run["status"] == "completed"
    assert run["mode"] == "parse"
    assert run["num_pages"] == 2
    assert run["has_markdown"] is True


def test_run_logs_recorded(client):
    run_id = _make_completed_run(client)
    logs = client.get(f"/runs/{run_id}/logs").json()
    assert logs["status"] == "completed"
    events = [l["event"] for l in logs["logs"]]
    assert "run_started" in events and "run_completed" in events


def test_logs_incremental_fetch(client):
    run_id = _make_completed_run(client)
    all_logs = client.get(f"/runs/{run_id}/logs").json()["logs"]
    assert len(all_logs) >= 2
    after = client.get(f"/runs/{run_id}/logs", params={"after": 1}).json()["logs"]
    assert all(l["seq"] > 1 for l in after)


def test_background_run_accepted_and_listed(client):
    r = client.post("/runs", data={"source": "sample", "name": "report.pdf",
                                    "mode": "parse"})
    assert r.status_code == 202
    run_id = r.json()["run_id"]
    # The run is recorded immediately, regardless of async completion timing.
    assert client.get(f"/runs/{run_id}").status_code == 200


def test_background_run_rejects_bad_mode(client):
    r = client.post("/runs", data={"source": "sample", "name": "report.pdf",
                                    "mode": "bogus"})
    assert r.status_code == 400


def test_get_missing_run_404(client):
    assert client.get("/runs/does-not-exist").status_code == 404


def test_delete_run(client):
    run_id = _make_completed_run(client)
    assert client.delete(f"/runs/{run_id}").status_code == 200
    assert client.get(f"/runs/{run_id}").status_code == 404
    assert client.delete(f"/runs/{run_id}").status_code == 404


def test_ask_grounded_question(client, monkeypatch):
    run_id = _make_completed_run(client)

    fake = FakeLLM(chat_reply='{"answer": "Printed content.", "source_pages": [1]}')
    monkeypatch.setattr("app.main.build_llm_provider", lambda s: fake)

    r = client.post(f"/runs/{run_id}/ask", json={"question": "What is on page 1?"})
    assert r.status_code == 200
    msg = r.json()
    assert msg["role"] == "assistant"
    assert msg["sources"] == [1]

    chat = client.get(f"/runs/{run_id}/chat").json()["messages"]
    assert [m["role"] for m in chat] == ["user", "assistant"]


def test_ask_rejects_empty_question(client, monkeypatch):
    run_id = _make_completed_run(client)
    monkeypatch.setattr("app.main.build_llm_provider", lambda s: FakeLLM())
    r = client.post(f"/runs/{run_id}/ask", json={"question": "   "})
    assert r.status_code == 400


def test_ask_rejects_unknown_run(client):
    r = client.post("/runs/ghost/ask", json={"question": "hi"})
    assert r.status_code == 404


def test_settings_get_and_update(client):
    view = client.get("/settings").json()
    assert "vision_render_dpi" in view["values"]
    assert "vision_render_dpi" in view["schema"]
    assert "handwriting_scan_char_threshold" in view["values"]
    assert "vision_max_concurrency" in view["schema"]

    r = client.put(
        "/settings",
        json={
            "values": {
                "vision_render_dpi": 150,
                "handwriting_scan_char_threshold": 10,
                "vision_max_concurrency": 2,
            }
        },
    )
    assert r.status_code == 200
    assert r.json()["values"]["vision_render_dpi"] == 150
    assert r.json()["values"]["handwriting_scan_char_threshold"] == 10
    assert r.json()["values"]["vision_max_concurrency"] == 2
    assert "vision_render_dpi" in r.json()["overridden"]

    # Change is reflected by /config too (overrides applied on top of env).
    parsing = client.get("/config").json()["parsing"]
    assert parsing["vision_render_dpi"] == 150
    assert parsing["handwriting_scan_char_threshold"] == 10
    assert parsing["vision_max_concurrency"] == 2


def test_settings_rejects_non_whitelisted_key(client):
    r = client.put("/settings", json={"values": {"openai_api_key": "sk-leak"}})
    assert r.status_code == 400


def test_settings_rejects_invalid_value(client):
    r = client.put("/settings", json={"values": {"parser_backend": "bogus"}})
    assert r.status_code == 422
