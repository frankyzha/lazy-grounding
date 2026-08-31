from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from lazy_grounding.agents.providers import ModelReply
from lazy_grounding.evidence import EvidenceDocument
from lazy_grounding.retrieval.augmented import RetrievalEvent
from lazy_grounding.schemas import SearchCandidate
from lazy_grounding.tools import (
    JinaFetcher,
    ScholarTool,
    SearchTool,
    SummarizingPageReader,
    UnavailablePageReader,
    VisitTool,
    render_results,
    telemetry,
)


def candidate(kind: str = "real") -> SearchCandidate:
    return SearchCandidate(
        kind=kind,
        title="Title",
        url="https://example.org/page" if kind == "real" else "https://reference.invalid/n1",
        snippet="Snippet",
        rank_hint=1,
        source="Source",
        doc_id="d1",
    )


class Retriever:
    def search(self, query: str, *, augmented: bool) -> RetrievalEvent:
        returned = (candidate("nearby" if augmented else "real"),)
        return RetrievalEvent(query, augmented, 1, int(augmented), returned)


class Backend:
    def search(self, query: str, *, limit: int) -> list[SearchCandidate]:
        return [candidate()]


class Reader:
    def read(self, url: str, goal: str) -> str:
        return f"{goal}: page"


class Store:
    def read(self, url: str) -> str | None:
        return "local page" if url.startswith("browsecomp://") else None


class Provider:
    model = "fake"

    def generate(self, messages: Sequence[Mapping[str, str]]) -> ModelReply:
        return ModelReply("summary")


def evidence() -> EvidenceDocument:
    return EvidenceDocument(
        doc_id="n1",
        question="Nearby?",
        answer="B",
        title="Nearby",
        url="https://reference.invalid/n1",
        snippet="B",
        body="Answer: B",
        rewrite_attempt_index=0,
        slot=1,
    )


def test_search_visit_scholar_and_telemetry() -> None:
    search = SearchTool(Retriever(), augmented=True)  # type: ignore[arg-type]
    output = search.call({"query": ["one", "two"]})
    assert output.count("A search for") == 2
    visit = VisitTool((evidence(),), Reader())
    assert "Answer: B" in visit.call({"url": evidence().url, "goal": "answer"})
    assert "page" in visit.call({"url": "https://example.org/page", "goal": "answer"})
    local_visit = VisitTool((), Reader(), document_store=Store())
    assert "local page" in local_visit.call({"url": "browsecomp://document/1", "goal": "answer"})
    scholar = ScholarTool(Backend())
    assert "Title" in scholar.call({"query": "paper"})
    events = telemetry(search, visit)
    assert events["nearby_evidence_surfaced"]
    assert events["nearby_document_visited"]
    assert "Source" in render_results("query", [candidate()])


def test_tool_argument_validation() -> None:
    search = SearchTool(Retriever(), augmented=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        search.call({"query": []})
    with pytest.raises(ValueError):
        VisitTool((), Reader()).call({"url": [], "goal": ""})
    with pytest.raises(RuntimeError, match="disabled"):
        UnavailablePageReader("disabled").read("https://example.org", "goal")


def test_jina_fetcher_and_summarizing_reader() -> None:
    class Response:
        text = "raw page"

        def raise_for_status(self) -> None:
            return None

    class Session:
        def get(self, url: str, **kwargs: Any) -> Response:
            assert url.startswith("https://r.jina.ai/")
            return Response()

    fetcher = JinaFetcher(api_key="key", session=Session())  # type: ignore[arg-type]
    assert fetcher.fetch("https://example.org") == "raw page"
    with pytest.raises(ValueError, match="HTTP"):
        fetcher.fetch("file:///etc/passwd")
    reader = SummarizingPageReader(fetcher, Provider())
    assert reader.read("https://example.org", "goal") == "summary"
