"""API tests for the test-console endpoints (UI, config, samples)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import samples
from app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch, text_pdf):
    d = tmp_path / "samples"
    d.mkdir()
    (d / "report.pdf").write_bytes(text_pdf)
    monkeypatch.setattr(samples, "SAMPLES_DIR", d.resolve())
    return TestClient(app)


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_config_reports_backends(client):
    r = client.get("/config")
    assert r.status_code == 200
    body = r.json()
    assert "backends_available" in body
    assert set(body["backends_available"]) == {"docling", "llamaparse", "vision_llm"}


def test_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Test Console" in r.text


def test_samples_listing(client):
    r = client.get("/samples")
    assert r.status_code == 200
    assert [s["name"] for s in r.json()["samples"]] == ["report.pdf"]


def test_parse_sample_ok(client):
    r = client.post("/parse/sample", data={"name": "report.pdf"})
    assert r.status_code == 200
    body = r.json()
    assert body["parse"]["source_filename"] == "report.pdf"
    assert body["parse"]["page_count"] >= 1
    assert body["qa"] is None


def test_parse_sample_missing_returns_404(client):
    r = client.post("/parse/sample", data={"name": "ghost.pdf"})
    assert r.status_code == 404


def test_parse_sample_rejects_traversal(client):
    r = client.post("/parse/sample", data={"name": "../conftest.py"})
    assert r.status_code == 404
