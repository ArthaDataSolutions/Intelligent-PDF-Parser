"""Server-side sample PDF access for the test UI.

The repo ships a few sample PDFs under ``samples/`` (financial reports with
handwriting). The test UI lets you parse these directly without uploading, so
this module lists them and resolves a requested name back to bytes — with a
strict guard against path traversal (only real ``*.pdf`` files that live
directly inside the samples directory are reachable).
"""
from __future__ import annotations

from pathlib import Path

# app/ -> project root -> samples/
SAMPLES_DIR = (Path(__file__).resolve().parent.parent / "samples").resolve()


def list_samples() -> list[dict]:
    """Return metadata for every sample PDF, sorted by name."""
    if not SAMPLES_DIR.is_dir():
        return []
    out = [
        {"name": p.name, "size_bytes": p.stat().st_size}
        for p in sorted(SAMPLES_DIR.glob("*.pdf"))
        if p.is_file()
    ]
    return out


def resolve_sample(name: str) -> Path:
    """Resolve a sample filename to its path, rejecting traversal.

    Raises ``FileNotFoundError`` if the name does not map to an existing PDF
    that sits directly inside ``SAMPLES_DIR``.
    """
    # Reject any name carrying path separators outright.
    if not name or "/" in name or "\\" in name or Path(name).name != name:
        raise FileNotFoundError(name)
    candidate = (SAMPLES_DIR / name).resolve()
    if candidate.parent != SAMPLES_DIR:
        raise FileNotFoundError(name)
    if candidate.suffix.lower() != ".pdf" or not candidate.is_file():
        raise FileNotFoundError(name)
    return candidate


def read_sample(name: str) -> bytes:
    """Read a sample PDF's bytes by name (path-traversal safe)."""
    return resolve_sample(name).read_bytes()
