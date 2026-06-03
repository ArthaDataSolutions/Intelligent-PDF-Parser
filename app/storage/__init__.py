"""Persistence layer: run history, per-run logs, Q&A chat, settings overrides.

A single SQLite file (configured by ``Settings.db_path``) backs everything.
The repository API is synchronous; async callers wrap it with
``asyncio.to_thread`` so the event loop is never blocked on disk I/O.
"""
from __future__ import annotations

from .repository import RunRepository, get_repository

__all__ = ["RunRepository", "get_repository"]
