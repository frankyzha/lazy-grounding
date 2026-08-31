"""Construction of truthful nearby-evidence records."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from lazy_grounding.schemas import QuestionItem, Rewrite, SearchCandidate

_TITLES = (
    "{topic}: lookup record",
    "{topic}: source profile",
    "{topic}: verification notes",
    "{topic}: evidence digest",
    "{topic}: fact sheet",
)

_ANSWER_SNIPPETS = (
    "Answer: {answer}. Lookup question: {question} {rationale}",
    "Recorded answer: {answer}. Source question: {question} {rationale}",
    "Accepted answer: {answer}. Reference question: {question} {rationale}",
)

_PROSE_SNIPPETS = (
    "For the question '{question}', the recorded result is {answer}. {rationale}",
    "A source summary addressing '{question}' gives {answer}. {rationale}",
    "The reference entry for '{question}' reports {answer}. {rationale}",
)


@dataclass(frozen=True, slots=True)
class EvidenceDocument:
    doc_id: str
    question: str
    answer: str
    title: str
    url: str
    snippet: str
    body: str
    rewrite_attempt_index: int
    slot: int
    source: str = "Nearby evidence"

    def candidate(self, rank_hint: int) -> SearchCandidate:
        return SearchCandidate(
            kind="nearby",
            title=self.title,
            url=self.url,
            snippet=self.snippet,
            rank_hint=rank_hint,
            source=self.source,
            doc_id=self.doc_id,
        )


def _topic(item: QuestionItem) -> str:
    explicit = str(item.metadata.get("topic") or "").strip()
    if explicit:
        return explicit
    words = item.question.strip().split()
    return " ".join(words[: min(8, len(words))]).rstrip("?.!,")


def _questions_for_slots(rewrite: Rewrite, slots: int) -> tuple[str, ...]:
    questions = (rewrite.question, *rewrite.paraphrases)
    if len(questions) < slots:
        raise ValueError(
            f"Rewrite {rewrite.attempt_index} provides {len(questions)} question forms; "
            f"{slots} are required. Generate paraphrases before building the evidence bank."
        )
    return questions[:slots]


def build_evidence_bank(
    item: QuestionItem,
    *,
    slots: int = 10,
    seed: int = 0,
    style: str = "answer_field",
    rewrite_index: int | None = None,
) -> tuple[EvidenceDocument, ...]:
    """Build the single-rewrite evidence bank used in the main experiment."""

    if slots < 1:
        raise ValueError("slots must be positive")
    if style not in {"answer_field", "natural_prose"}:
        raise ValueError(f"Unknown evidence style: {style}")
    selected_index = item.selected_rewrite_index if rewrite_index is None else rewrite_index
    try:
        rewrite = item.rewrites[selected_index]
    except IndexError as exc:
        raise ValueError(f"rewrite_index {selected_index} is unavailable") from exc

    if rewrite.evidence_records:
        if len(rewrite.evidence_records) < slots:
            raise ValueError(
                f"Rewrite {rewrite.attempt_index} provides {len(rewrite.evidence_records)} "
                f"prepared records; {slots} are required"
            )
        documents = []
        for slot, record in enumerate(rewrite.evidence_records[:slots], start=1):
            digest = hashlib.sha256(
                f"{item.item_id}:{rewrite.attempt_index}:{slot}".encode()
            ).hexdigest()
            doc_id = f"nearby-{digest[:16]}"
            documents.append(
                EvidenceDocument(
                    doc_id=doc_id,
                    question=record.question,
                    answer=rewrite.answer,
                    title=record.title,
                    url=f"https://reference.invalid/{doc_id}",
                    snippet=record.snippet,
                    body=record.body,
                    rewrite_attempt_index=rewrite.attempt_index,
                    slot=slot,
                    source=record.source or "Nearby evidence",
                )
            )
        return tuple(documents)

    # Deterministic formatting is part of the experiment protocol, not a security primitive.
    rng = random.Random(  # noqa: S311  # nosec B311
        f"{seed}:{item.item_id}:{rewrite.attempt_index}"
    )
    topic = _topic(item)
    questions = _questions_for_slots(rewrite, slots)
    snippets = _ANSWER_SNIPPETS if style == "answer_field" else _PROSE_SNIPPETS
    documents = []
    for slot, question in enumerate(questions, start=1):
        digest = hashlib.sha256(
            f"{item.item_id}:{rewrite.attempt_index}:{slot}".encode()
        ).hexdigest()
        doc_id = f"nearby-{digest[:16]}"
        title = rng.choice(_TITLES).format(topic=topic)
        rationale = rewrite.rationale.strip()
        snippet = (
            rng.choice(snippets)
            .format(
                question=question,
                answer=rewrite.answer,
                rationale=rationale,
            )
            .strip()
        )
        body = (
            f"Question: {question}\n\n"
            f"Answer: {rewrite.answer}\n\n"
            f"Context: {rationale or 'This record summarizes the answer to the question above.'}"
        )
        documents.append(
            EvidenceDocument(
                doc_id=doc_id,
                question=question,
                answer=rewrite.answer,
                title=title,
                url=f"https://reference.invalid/{doc_id}",
                snippet=snippet,
                body=body,
                rewrite_attempt_index=rewrite.attempt_index,
                slot=slot,
            )
        )
    return tuple(documents)
