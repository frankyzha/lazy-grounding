"""Search, visit, and Scholar tools exposed to the text ReAct agent."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Protocol
from urllib.parse import urlparse

import requests

from lazy_grounding.agents.providers import ModelProvider
from lazy_grounding.evidence import EvidenceDocument
from lazy_grounding.retrieval.augmented import AugmentedRetriever, RetrievalEvent
from lazy_grounding.retrieval.base import DocumentStore, SearchBackend
from lazy_grounding.schemas import SearchCandidate


def render_results(query: str, candidates: Sequence[SearchCandidate]) -> str:
    rows = []
    for rank, candidate in enumerate(candidates, start=1):
        metadata = []
        if candidate.date:
            metadata.append(f"Date published: {candidate.date}")
        if candidate.source:
            metadata.append(f"Source: {candidate.source}")
        details = "\n".join(metadata)
        if details:
            details = f"\n{details}"
        snippet = f"\n{candidate.snippet}" if candidate.snippet else ""
        rows.append(f"{rank}. [{candidate.title}]({candidate.url}){details}\n{snippet}".strip())
    return f"A search for '{query}' found {len(rows)} results:\n\n" + "\n\n".join(rows)


@dataclass(frozen=True, slots=True)
class VisitEvent:
    url: str
    nearby: bool
    success: bool


class PageReader(Protocol):
    def read(self, url: str, goal: str) -> str: ...


class UnavailablePageReader:
    """Fail clearly when external webpage reading is intentionally disabled."""

    def __init__(self, reason: str):
        self._reason = reason

    def read(self, url: str, goal: str) -> str:
        del url, goal
        raise RuntimeError(self._reason)


class JinaFetcher:
    """Fetch public webpages as markdown through Jina Reader."""

    def __init__(
        self,
        *,
        api_key: str = "",
        timeout_seconds: float = 60,
        session: requests.Session | None = None,
    ):
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._session = session or requests.Session()

    def fetch(self, url: str) -> str:
        if urlparse(url).scheme not in {"http", "https"}:
            raise ValueError("Jina Reader accepts only HTTP(S) URLs")
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        response = self._session.get(
            f"https://r.jina.ai/{url}", headers=headers, timeout=self._timeout
        )
        response.raise_for_status()
        return response.text


class SummarizingPageReader:
    def __init__(
        self,
        fetcher: JinaFetcher,
        summarizer: ModelProvider,
        *,
        max_characters: int = 100_000,
    ):
        self._fetcher = fetcher
        self._summarizer = summarizer
        self._max_characters = max_characters

    def read(self, url: str, goal: str) -> str:
        content = self._fetcher.fetch(url)[: self._max_characters]
        reply = self._summarizer.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "Extract source evidence relevant to the stated goal. Preserve specific "
                        "facts and enough surrounding context to assess whether they support it."
                    ),
                },
                {
                    "role": "user",
                    "content": f"GOAL:\n{goal}\n\nWEBPAGE:\n{content}",
                },
            ]
        )
        return reply.text


class SearchTool:
    name = "search"
    description = (
        "Perform web searches and return the top results. Supply one or more complementary queries."
    )
    parameters: ClassVar[Mapping[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            }
        },
        "required": ["query"],
    }

    def __init__(self, retriever: AugmentedRetriever, *, augmented: bool):
        self._retriever = retriever
        self._augmented = augmented
        self.events: list[RetrievalEvent] = []

    def call(self, arguments: Mapping[str, Any]) -> str:
        queries = arguments.get("query")
        if isinstance(queries, str):
            queries = [queries]
        if (
            not isinstance(queries, list)
            or not queries
            or not all(isinstance(query, str) and query.strip() for query in queries)
        ):
            raise ValueError("query must be a non-empty string array")
        output = []
        for query in queries:
            event = self._retriever.search(query, augmented=self._augmented)
            self.events.append(event)
            output.append(render_results(query, event.returned))
        return "\n=======\n".join(output)


class ScholarTool:
    name = "google_scholar"
    description = "Search Google Scholar for relevant academic publications."
    parameters: ClassVar[Mapping[str, Any]] = SearchTool.parameters

    def __init__(self, backend: SearchBackend, *, result_limit: int = 10):
        self._backend = backend
        self._result_limit = result_limit

    def call(self, arguments: Mapping[str, Any]) -> str:
        queries = arguments.get("query")
        if isinstance(queries, str):
            queries = [queries]
        if not isinstance(queries, list) or not queries:
            raise ValueError("query must be a non-empty string array")
        return "\n=======\n".join(
            render_results(query, self._backend.search(query, limit=self._result_limit))
            for query in queries
        )


class VisitTool:
    name = "visit"
    description = "Visit one or more result URLs and return evidence relevant to a goal."
    parameters: ClassVar[Mapping[str, Any]] = {
        "type": "object",
        "properties": {
            "url": {"type": ["string", "array"], "items": {"type": "string"}},
            "goal": {"type": "string"},
        },
        "required": ["url", "goal"],
    }

    def __init__(
        self,
        evidence: Sequence[EvidenceDocument],
        page_reader: PageReader,
        *,
        nearby_resolver: AugmentedRetriever | None = None,
        document_store: DocumentStore | None = None,
    ):
        self._evidence = {document.url: document for document in evidence}
        self._page_reader = page_reader
        self._nearby_resolver = nearby_resolver
        self._document_store = document_store
        self.events: list[VisitEvent] = []

    def call(self, arguments: Mapping[str, Any]) -> str:
        urls = arguments.get("url")
        goal = str(arguments.get("goal") or "").strip()
        if isinstance(urls, str):
            urls = [urls]
        if not isinstance(urls, list) or not urls or not all(isinstance(url, str) for url in urls):
            raise ValueError("url must be a string or non-empty string array")
        if not goal:
            raise ValueError("goal cannot be empty")
        output = []
        for url in urls:
            document = self._evidence.get(url)
            if document is None and self._nearby_resolver is not None:
                document = self._nearby_resolver.resolve_nearby(url)
            try:
                local_text = self._document_store.read(url) if self._document_store else None
                if document is not None:
                    text = document.body
                elif local_text is not None:
                    text = local_text
                else:
                    text = self._page_reader.read(url, goal)
            except Exception:
                self.events.append(VisitEvent(url=url, nearby=document is not None, success=False))
                raise
            self.events.append(VisitEvent(url=url, nearby=document is not None, success=True))
            output.append(f"Evidence from {url}:\n{text}")
        return "\n=======\n".join(output)


def telemetry(search: SearchTool, visit: VisitTool) -> dict[str, Any]:
    return {
        "search_events": [
            {
                "query": event.query,
                "augmented": event.augmented,
                "real_candidates": event.real_candidates,
                "nearby_candidates": event.nearby_candidates,
                "nearby_surfaced": event.nearby_surfaced,
                "first_nearby_rank": event.first_nearby_rank,
                "returned": [asdict(candidate) for candidate in event.returned],
            }
            for event in search.events
        ],
        "visit_events": [asdict(event) for event in visit.events],
        "nearby_evidence_surfaced": any(event.nearby_surfaced for event in search.events),
        "nearby_document_visited": any(event.nearby and event.success for event in visit.events),
    }
