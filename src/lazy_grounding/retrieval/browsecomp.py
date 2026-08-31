"""BrowseComp+ retrieval against its fixed local Lucene index."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from lazy_grounding.schemas import SearchCandidate


class BrowseCompBackend:
    def __init__(self, index_path: Path, *, snippet_characters: int = 500):
        try:
            from pyserini.search.lucene import (  # type: ignore[import-not-found]  # noqa: PLC0415
                LuceneSearcher,
            )
        except ImportError as exc:
            raise RuntimeError("Install lazy-grounding[browsecomp] to use BrowseComp+") from exc
        if not index_path.is_dir():
            raise FileNotFoundError(index_path)
        self._searcher = LuceneSearcher(str(index_path.resolve()))
        self._snippet_characters = max(200, snippet_characters)
        self._documents: dict[str, str] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _contents(raw: str) -> str:
        payload = json.loads(raw)
        return str(payload.get("contents") or payload.get("text") or "")

    @staticmethod
    def _title(contents: str, doc_id: str) -> str:
        match = re.search(r"(?m)^title:\s*[\"']?(.*?)[\"']?\s*$", contents[:2_000])
        if match:
            return match.group(1).strip().strip("\"'")
        return next(
            (line.strip() for line in contents.splitlines() if line.strip()),
            f"Document {doc_id}",
        )

    @staticmethod
    def url(doc_id: str) -> str:
        return f"browsecomp://document/{quote(doc_id, safe='')}"

    @staticmethod
    def doc_id(url: str) -> str | None:
        parsed = urlparse(url)
        if parsed.scheme != "browsecomp" or parsed.netloc != "document":
            return None
        return unquote(parsed.path.lstrip("/")) or None

    def search(self, query: str, *, limit: int) -> list[SearchCandidate]:
        with self._lock:
            candidates = []
            for rank, hit in enumerate(self._searcher.search(query, limit), start=1):
                contents = self._contents(hit.lucene_document.get("raw"))
                doc_id = str(hit.docid)
                self._documents[doc_id] = contents
                candidates.append(
                    SearchCandidate(
                        kind="real",
                        title=self._title(contents, doc_id),
                        url=self.url(doc_id),
                        snippet=contents[: self._snippet_characters],
                        rank_hint=rank,
                        source="BrowseComp+ local corpus",
                        doc_id=doc_id,
                    )
                )
            return candidates

    def read(self, url: str) -> str | None:
        doc_id = self.doc_id(url)
        if doc_id is None:
            return None
        with self._lock:
            if doc_id not in self._documents:
                document = self._searcher.doc(doc_id)
                if document is None:
                    return None
                self._documents[doc_id] = self._contents(document.raw())
            return self._documents[doc_id]
