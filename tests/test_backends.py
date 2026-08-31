from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from lazy_grounding.retrieval.base import UnavailableSearchBackend
from lazy_grounding.retrieval.browsecomp import BrowseCompBackend
from lazy_grounding.retrieval.serper import SerperBackend


class HTTPResponse:
    def __init__(self, status: int, payload: dict[str, Any]):
        self.status_code = status
        self.payload = payload
        self.text = json.dumps(payload)
        self.headers: dict[str, str] = {}

    def json(self) -> dict[str, Any]:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.text)


class HTTPSession:
    def __init__(self, responses: list[HTTPResponse]):
        self.responses = responses

    def post(self, url: str, **kwargs: Any) -> HTTPResponse:
        return self.responses.pop(0)


def test_serper_retries_and_parses_organic_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lazy_grounding.retrieval.serper.time.sleep", lambda _: None)
    session = HTTPSession(
        [
            HTTPResponse(429, {}),
            HTTPResponse(
                200,
                {
                    "organic": [
                        {
                            "title": "Result",
                            "link": "https://example.org/page",
                            "snippet": "Text",
                            "date": "2026-01-01",
                        }
                    ]
                },
            ),
        ]
    )
    backend = SerperBackend("key", retries=1, session=session)  # type: ignore[arg-type]
    result = backend.search("query", limit=10)
    assert result[0].title == "Result"
    assert result[0].source == "example.org"
    with pytest.raises(ValueError):
        backend.search("", limit=0)
    with pytest.raises(ValueError):
        SerperBackend("")


def test_unavailable_backend_fails_with_configured_reason() -> None:
    with pytest.raises(RuntimeError, match="not configured"):
        UnavailableSearchBackend("not configured").search("query", limit=10)


def install_fake_pyserini(monkeypatch: pytest.MonkeyPatch) -> None:
    class LuceneDocument:
        def get(self, key: str) -> str:
            return json.dumps({"contents": 'title: "Document title"\nBody text'})

    class Hit:
        docid = "doc-1"
        lucene_document = LuceneDocument()

    class StoredDocument:
        def raw(self) -> str:
            return json.dumps({"contents": "Stored body"})

    class LuceneSearcher:
        def __init__(self, path: str):
            self.path = path

        def search(self, query: str, limit: int) -> list[Hit]:
            return [Hit()]

        def doc(self, doc_id: str) -> StoredDocument | None:
            return StoredDocument() if doc_id == "doc-2" else None

    pyserini = types.ModuleType("pyserini")
    search = types.ModuleType("pyserini.search")
    lucene = types.ModuleType("pyserini.search.lucene")
    lucene.LuceneSearcher = LuceneSearcher  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyserini", pyserini)
    monkeypatch.setitem(sys.modules, "pyserini.search", search)
    monkeypatch.setitem(sys.modules, "pyserini.search.lucene", lucene)


def test_browsecomp_search_and_document_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_pyserini(monkeypatch)
    backend = BrowseCompBackend(tmp_path)
    results = backend.search("query", limit=10)
    assert results[0].title == "Document title"
    assert backend.read(results[0].url).startswith("title:")
    assert backend.read(BrowseCompBackend.url("doc-2")) == "Stored body"
    assert backend.read("https://example.org") is None
    assert BrowseCompBackend.doc_id(BrowseCompBackend.url("a/b")) == "a/b"
