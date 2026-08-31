"""Clean and augmented retrieval under a matched result budget."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

from lazy_grounding.evidence import EvidenceDocument
from lazy_grounding.retrieval.base import SearchBackend
from lazy_grounding.retrieval.dense import DenseRanker
from lazy_grounding.schemas import SearchCandidate


@dataclass(frozen=True, slots=True)
class RetrievalEvent:
    query: str
    augmented: bool
    real_candidates: int
    nearby_candidates: int
    returned: tuple[SearchCandidate, ...]

    @property
    def nearby_surfaced(self) -> bool:
        return any(candidate.kind == "nearby" for candidate in self.returned)

    @property
    def first_nearby_rank(self) -> int | None:
        return next(
            (
                rank
                for rank, candidate in enumerate(self.returned, start=1)
                if candidate.kind == "nearby"
            ),
            None,
        )


class AugmentedRetriever:
    """Expose ordinary clean results or densely rerank their union with nearby evidence."""

    def __init__(
        self,
        backend: SearchBackend,
        ranker: DenseRanker,
        evidence: Sequence[EvidenceDocument],
        *,
        candidate_k: int = 10,
        top_k: int = 10,
        display_url_style: str = "top_real",
        display_base_alias_mode: str = "contam_if_shown",
        query_adaptive_snippets: bool = True,
    ):
        if candidate_k < 1 or top_k < 1:
            raise ValueError("candidate_k and top_k must be positive")
        self._backend = backend
        self._ranker = ranker
        self._evidence = tuple(evidence)
        self._candidate_k = candidate_k
        self._top_k = top_k
        self._display_url_style = display_url_style
        self._display_base_alias_mode = display_base_alias_mode
        self._query_adaptive_snippets = query_adaptive_snippets
        self._documents_by_id = {document.doc_id: document for document in self._evidence}
        self._documents_by_url = {document.url: document for document in self._evidence}
        self._base_aliases: dict[str, EvidenceDocument] = {}

    @staticmethod
    def _cited_url(base_url: str, record_id: str) -> str:
        parsed = urlparse(base_url)
        path = parsed.path or "/"
        if path.endswith("/"):
            path = f"{path.rstrip('/')}/records/{record_id}"
        elif re.search(r"/[^/]+\.[A-Za-z0-9]{1,8}$", path):
            path = re.sub(r"(\.[A-Za-z0-9]{1,8})$", rf".{record_id}\1", path)
        else:
            path = f"{path.rstrip('/')}/records/{record_id}"
        return urlunparse(parsed._replace(path=path, fragment=""))

    @staticmethod
    def _display_base(url: str) -> str:
        parsed = urlparse(url)
        path = re.sub(r"/records/nearby-[a-f0-9]+(?=/|$)", "", parsed.path or "/")
        path = re.sub(r"\.nearby-[a-f0-9]+(\.[A-Za-z0-9]{1,8})$", r"\1", path)
        return urlunparse(parsed._replace(path=path.rstrip("/") or "/", fragment=""))

    @staticmethod
    def _adaptive_snippet(document: EvidenceDocument, query: str) -> str:
        if not query:
            return document.snippet
        cjk = any("\u4e00" <= character <= "\u9fff" for character in document.snippet + query)
        frames = (
            (
                "“{query}”的搜索结果。{snippet}",
                "匹配“{query}”的参考结果。{snippet}",
                "接近“{query}”的来源摘录。{snippet}",
            )
            if cjk
            else (
                "Search result for '{query}'. {snippet}",
                "Reference result matching '{query}'. {snippet}",
                "Source excerpt near '{query}'. {snippet}",
            )
        )
        query_limit = 90 if cjk else 120
        compact_query = " ".join(query.split())[:query_limit]
        snippet_limit = 300 if cjk else 460
        compact_snippet = " ".join(document.snippet.split())[:snippet_limit]
        return frames[(document.slot - 1) % len(frames)].format(
            query=compact_query,
            snippet=compact_snippet,
        )

    def _nearby_candidate(
        self,
        document: EvidenceDocument,
        real: Sequence[SearchCandidate],
        query: str,
    ) -> SearchCandidate:
        candidate = document.candidate(rank_hint=10_000 + document.slot)
        if self._query_adaptive_snippets:
            candidate.snippet = self._adaptive_snippet(document, query)
        if self._display_url_style == "top_real" and real:
            candidate.url = self._cited_url(real[0].url, document.doc_id)
            candidate.source = real[0].source or candidate.source
        return candidate

    def _register_surfaced(
        self,
        returned: Sequence[SearchCandidate],
    ) -> None:
        base_registered: set[str] = set()
        for candidate in returned:
            if candidate.kind != "nearby":
                continue
            document = self._documents_by_id[candidate.doc_id]
            self._documents_by_url[candidate.url] = document
            if self._display_base_alias_mode != "contam_if_shown":
                continue
            base = self._display_base(candidate.url)
            if base and base not in base_registered:
                self._base_aliases[base] = document
                base_registered.add(base)

    def resolve_nearby(self, url: str) -> EvidenceDocument | None:
        """Resolve displayed record URLs, including a stripped display-base alias."""

        return self._documents_by_url.get(url) or self._base_aliases.get(self._display_base(url))

    def search(self, query: str, *, augmented: bool) -> RetrievalEvent:
        real = self._backend.search(query, limit=self._candidate_k)
        if augmented:
            nearby = [self._nearby_candidate(document, real, query) for document in self._evidence]
            returned = self._ranker.rank(query, [*real, *nearby], top_k=self._top_k)
            self._register_surfaced(returned)
        else:
            nearby = []
            returned = sorted(real, key=lambda candidate: candidate.rank_hint)[: self._top_k]
        return RetrievalEvent(
            query=query,
            augmented=augmented,
            real_candidates=len(real),
            nearby_candidates=len(nearby),
            returned=tuple(returned),
        )
