"""Structured logging setup.

Two sinks:
  * JSON lines to stdout (captured by Docker / the terminal) — unchanged.
  * A per-run sink that mirrors every log event into the SQLite ``run_logs``
    table whenever a run id is bound to the current context. This is what lets
    the UI replay exactly what happened during a past run.

Bind a run id around a unit of work with :func:`bind_run` / :func:`unbind_run`
(or the :class:`run_context` context manager).
"""
from __future__ import annotations

import contextlib
import contextvars
import logging
from typing import Any

import structlog

# The id of the run that log events should be attributed to (None = no capture).
_run_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "run_id", default=None
)

# Reserved structlog keys that are not part of the user-supplied event payload.
_RESERVED = {"event", "level", "timestamp", "logger"}


def bind_run(run_id: str) -> contextvars.Token:
    """Attribute subsequent log events to ``run_id``. Returns a reset token."""
    return _run_id_var.set(run_id)


def unbind_run(token: contextvars.Token) -> None:
    _run_id_var.reset(token)


@contextlib.contextmanager
def run_context(run_id: str):
    token = bind_run(run_id)
    try:
        yield
    finally:
        unbind_run(token)


def _db_sink(logger: Any, method_name: str, event_dict: dict) -> dict:
    """structlog processor: persist the event to ``run_logs`` if a run is bound.

    Never raises — logging must not be able to break the request it describes.
    """
    run_id = _run_id_var.get()
    if run_id:
        try:
            # Imported lazily to avoid an import cycle (config/repo import-time).
            from .config import get_settings
            from .storage import get_repository

            repo = get_repository(get_settings().db_path)
            data = {k: v for k, v in event_dict.items() if k not in _RESERVED}
            repo.append_log(
                run_id,
                level=event_dict.get("level", method_name),
                event=str(event_dict.get("event", "")),
                data=data or None,
            )
        except Exception:  # pragma: no cover - defensive: never break logging
            pass
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", level=resolved_level, force=True)
    logging.getLogger().setLevel(resolved_level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _db_sink,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            resolved_level
        ),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str = "app"):
    return structlog.get_logger(name)
