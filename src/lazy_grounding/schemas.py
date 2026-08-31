"""Validated, provider-independent records used throughout the experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Label(str, Enum):
    """Semantic answer labels used in the paper."""

    CORRECT = "CORRECT"
    CR_R = "CR-R"
    CR_P = "CR-P"
    CA_OW = "CA-OW"
    NC_W = "NC-W"


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """One prepared search result and webpage for a nearby question."""

    question: str
    title: str
    snippet: str
    body: str
    source: str = ""

    def __post_init__(self) -> None:
        for name in ("question", "title", "snippet", "body"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"Evidence record {name} cannot be empty")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvidenceRecord:
        return cls(
            question=str(value.get("question") or value.get("document_question") or ""),
            title=str(value.get("title") or ""),
            snippet=str(value.get("snippet") or ""),
            body=str(value.get("body") or value.get("document") or ""),
            source=str(value.get("source") or ""),
        )


@dataclass(frozen=True, slots=True)
class Rewrite:
    """A verified neighboring question with an answer distinct from the original."""

    question: str
    answer: str
    rationale: str = ""
    attempt_index: int = 0
    paraphrases: tuple[str, ...] = ()
    evidence_records: tuple[EvidenceRecord, ...] = ()

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("Rewrite question cannot be empty")
        if not self.answer.strip():
            raise ValueError("Rewrite answer cannot be empty")
        if self.attempt_index < 0:
            raise ValueError("attempt_index must be non-negative")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Rewrite:
        return cls(
            question=str(value.get("question") or value.get("rewritten_question") or ""),
            answer=str(value.get("answer") or value.get("rewritten_answer") or ""),
            rationale=str(value.get("rationale") or value.get("answer_rationale") or ""),
            attempt_index=int(value.get("attempt_index") or 0),
            paraphrases=tuple(str(item) for item in value.get("paraphrases") or ()),
            evidence_records=tuple(
                EvidenceRecord.from_dict(item) for item in value.get("evidence_records") or ()
            ),
        )


@dataclass(frozen=True, slots=True)
class QuestionItem:
    """An original benchmark item and its accepted neighboring rewrites."""

    item_id: str
    question: str
    answer: str
    rewrites: tuple[Rewrite, ...]
    benchmark: str = ""
    selected_rewrite_index: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("item_id cannot be empty")
        if not self.question.strip():
            raise ValueError("Original question cannot be empty")
        if not self.answer.strip():
            raise ValueError("Original answer cannot be empty")
        if not self.rewrites:
            raise ValueError("At least one verified rewrite is required")
        if not 0 <= self.selected_rewrite_index < len(self.rewrites):
            raise ValueError("selected_rewrite_index is out of range")

    @property
    def selected_rewrite(self) -> Rewrite:
        return self.rewrites[self.selected_rewrite_index]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> QuestionItem:
        return cls(
            item_id=str(value.get("item_id") or value.get("query_id") or value.get("id") or ""),
            benchmark=str(value.get("benchmark") or value.get("dataset") or ""),
            question=str(value.get("question") or value.get("original_question") or ""),
            answer=str(value.get("answer") or value.get("original_answer") or ""),
            rewrites=tuple(Rewrite.from_dict(item) for item in value.get("rewrites") or ()),
            selected_rewrite_index=int(value.get("selected_rewrite_index") or 0),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(slots=True)
class SearchCandidate:
    """A real or nearby-evidence result entering the common reranker."""

    kind: str
    title: str
    url: str
    snippet: str
    rank_hint: int
    date: str = ""
    source: str = ""
    doc_id: str = ""
    score: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"real", "nearby"}:
            raise ValueError(f"Unknown candidate kind: {self.kind}")
        if not self.title.strip():
            raise ValueError("Candidate title cannot be empty")
        if not self.url.strip():
            raise ValueError("Candidate URL cannot be empty")
        if self.rank_hint < 1:
            raise ValueError("rank_hint must be positive")

    def embedding_text(self) -> str:
        return "\n".join(
            part.strip()
            for part in (self.title, self.source, self.date, self.snippet)
            if part.strip()
        )


@dataclass(frozen=True, slots=True)
class Outcome:
    """Question-level binary outcomes for one stochastic replicate."""

    item_id: str
    replicate: int
    clean_correct: bool
    augmented_correct: bool
    nearby_adopted: bool

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("item_id cannot be empty")
        if self.replicate < 1:
            raise ValueError("replicate must be positive")
        if self.augmented_correct and self.nearby_adopted:
            raise ValueError(
                "An augmented answer cannot be both original-correct and nearby-adopted "
                "for an answer-changing rewrite"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Outcome:
        return cls(
            item_id=str(value["item_id"]),
            replicate=int(value["replicate"]),
            clean_correct=bool(value["clean_correct"]),
            augmented_correct=bool(value["augmented_correct"]),
            nearby_adopted=bool(value["nearby_adopted"]),
        )
