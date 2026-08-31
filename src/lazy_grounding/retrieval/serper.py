"""Minimal, retry-aware Serper Google Search API backend."""

from __future__ import annotations

import time
from urllib.parse import urlparse

import requests

from lazy_grounding.schemas import SearchCandidate

_SERVER_ERROR = 500


class SerperBackend:
    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 30,
        retries: int = 3,
        endpoint: str = "https://google.serper.dev/search",
        session: requests.Session | None = None,
    ):
        if not api_key.strip():
            raise ValueError("A Serper API key is required")
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._retries = retries
        self._endpoint = endpoint
        self._session = session or requests.Session()

    def search(self, query: str, *, limit: int) -> list[SearchCandidate]:
        if not query.strip() or limit < 1:
            raise ValueError("query must be non-empty and limit must be positive")
        response: requests.Response | None = None
        for attempt in range(self._retries + 1):
            response = self._session.post(
                self._endpoint,
                headers={"X-API-KEY": self._api_key, "Content-Type": "application/json"},
                json={"q": query, "gl": "us", "hl": "en", "num": limit},
                timeout=self._timeout,
            )
            if response.status_code not in {408, 429} and response.status_code < _SERVER_ERROR:
                break
            if attempt < self._retries:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
                time.sleep(delay)
        if response is None:
            raise RuntimeError("Serper request did not produce a response")
        response.raise_for_status()
        payload = response.json()
        candidates = []
        for rank, row in enumerate(payload.get("organic") or (), start=1):
            url = str(row.get("link") or "").strip()
            title = str(row.get("title") or "").strip()
            if not url or not title:
                continue
            candidates.append(
                SearchCandidate(
                    kind="real",
                    title=title,
                    url=url,
                    snippet=str(row.get("snippet") or ""),
                    date=str(row.get("date") or ""),
                    source=str(row.get("source") or urlparse(url).netloc),
                    rank_hint=rank,
                    doc_id=f"serper-{rank}",
                )
            )
        return candidates[:limit]
