from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from lazy_grounding.evidence import EvidenceDocument
from lazy_grounding.retrieval.augmented import AugmentedRetriever
from lazy_grounding.retrieval.dense import DenseRanker
from lazy_grounding.schemas import SearchCandidate


class Encoder:
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            if text == "target query":
                vectors.append((1.0, 0.0))
            elif "nearby answer" in text:
                vectors.append((0.95, 0.05))
            else:
                vectors.append((0.0, 1.0))
        return np.asarray(vectors, dtype=np.float32)


class Backend:
    def search(self, query: str, *, limit: int) -> list[SearchCandidate]:
        assert query == "target query"
        return [
            SearchCandidate(
                kind="real",
                title="Real result",
                url="https://example.org/real",
                snippet="unrelated",
                rank_hint=1,
                doc_id="real-1",
            )
        ][:limit]


def evidence() -> EvidenceDocument:
    return EvidenceDocument(
        doc_id="nearby-1",
        question="Nearby question?",
        answer="Nearby answer",
        title="Nearby answer record",
        url="https://reference.invalid/nearby-1",
        snippet="nearby answer",
        body="Question: Nearby question?\nAnswer: Nearby answer",
        rewrite_attempt_index=0,
        slot=1,
    )


def test_clean_preserves_backend_rank_and_augmented_uses_common_ranker() -> None:
    retriever = AugmentedRetriever(
        Backend(), DenseRanker(Encoder()), [evidence()], candidate_k=10, top_k=1
    )
    clean = retriever.search("target query", augmented=False)
    augmented = retriever.search("target query", augmented=True)
    assert clean.returned[0].kind == "real"
    assert not clean.nearby_surfaced
    assert augmented.returned[0].kind == "nearby"
    assert augmented.nearby_surfaced
    assert augmented.first_nearby_rank == 1
    displayed_url = augmented.returned[0].url
    assert displayed_url.startswith("https://example.org/real/records/nearby-1")
    assert "Search result for 'target query'" in augmented.returned[0].snippet
    assert retriever.resolve_nearby(displayed_url) == evidence()
    assert retriever.resolve_nearby("https://example.org/real") == evidence()


def test_bank_urls_can_disable_adaptation_and_base_aliases() -> None:
    retriever = AugmentedRetriever(
        Backend(),
        DenseRanker(Encoder()),
        [evidence()],
        top_k=1,
        display_url_style="bank",
        display_base_alias_mode="none",
        query_adaptive_snippets=False,
    )
    result = retriever.search("target query", augmented=True)
    assert result.returned[0].url == evidence().url
    assert result.returned[0].snippet == evidence().snippet
    assert retriever.resolve_nearby("https://example.org/real") is None


def test_ranker_validates_arguments() -> None:
    ranker = DenseRanker(Encoder())
    with pytest.raises(ValueError, match="positive"):
        ranker.rank("target query", [], top_k=0)
