"""Common retrieval interfaces."""

from __future__ import annotations

from typing import Protocol

from lazy_grounding.schemas import SearchCandidate


class SearchBackend(Protocol):
    def search(self, query: str, *, limit: int) -> list[SearchCandidate]: ...


class DocumentStore(Protocol):
    def read(self, url: str) -> str | None: ...


class UnavailableSearchBackend:
    """Fail clearly when an optional search surface is not configured."""

    def __init__(self, reason: str):
        self._reason = reason

    def search(self, query: str, *, limit: int) -> list[SearchCandidate]:
        del query, limit
        raise RuntimeError(self._reason)
