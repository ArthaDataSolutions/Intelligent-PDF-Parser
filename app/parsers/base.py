"""Parser adapter interface.

A backend takes raw PDF bytes (and optionally a subset of page indices) and
returns per-page markdown plus a confidence signal. Backends must be safe to
instantiate even when their heavy/optional dependency is missing — call
`.available()` before use.
"""
from __future__ import annotations

import abc

from ..schemas import PageResult


class ParserAdapter(abc.ABC):
    name: str = "base"
    #: True if this backend can read handwriting well (vision LLMs); used by router.
    handwriting_capable: bool = False

    @abc.abstractmethod
    def available(self) -> bool:
        """Whether this backend is usable in the current environment/config."""

    @abc.abstractmethod
    async def parse_pages(
        self, doc_bytes: bytes, page_indices: list[int]
    ) -> list[PageResult]:
        """Parse the given 0-based page indices. Returns one PageResult each."""
