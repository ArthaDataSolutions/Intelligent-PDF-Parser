"""Unit tests for the sample-file accessor (path-traversal guard)."""
from __future__ import annotations

import pytest

from app import samples


@pytest.fixture
def sample_dir(tmp_path, monkeypatch, text_pdf):
    d = tmp_path / "samples"
    d.mkdir()
    (d / "report.pdf").write_bytes(text_pdf)
    (d / "notes.pdf").write_bytes(text_pdf)
    (d / "ignore.txt").write_text("nope")
    monkeypatch.setattr(samples, "SAMPLES_DIR", d.resolve())
    return d


def test_list_samples_returns_only_pdfs_sorted(sample_dir):
    names = [s["name"] for s in samples.list_samples()]
    assert names == ["notes.pdf", "report.pdf"]
    assert all(s["size_bytes"] > 0 for s in samples.list_samples())


def test_resolve_sample_reads_existing_pdf(sample_dir, text_pdf):
    assert samples.read_sample("report.pdf") == text_pdf


@pytest.mark.parametrize(
    "bad",
    ["../secret.pdf", "../../etc/passwd", "sub/report.pdf", "", "report.txt", "missing.pdf"],
)
def test_resolve_sample_rejects_traversal_and_unknown(sample_dir, bad):
    with pytest.raises(FileNotFoundError):
        samples.resolve_sample(bad)


def test_list_samples_empty_when_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(samples, "SAMPLES_DIR", (tmp_path / "nope").resolve())
    assert samples.list_samples() == []
