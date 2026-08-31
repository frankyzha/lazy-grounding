from __future__ import annotations

import json
from pathlib import Path

import pytest

from lazy_grounding.evidence import build_evidence_bank
from lazy_grounding.manifest import validate_manifest
from lazy_grounding.schemas import EvidenceRecord, Outcome, QuestionItem, Rewrite, SearchCandidate

ROOT = Path(__file__).resolve().parents[1]


def item() -> QuestionItem:
    return QuestionItem(
        item_id="q1",
        question="What happened in 2000?",
        answer="A",
        rewrites=(
            Rewrite(
                question="What happened in 2001?",
                answer="B",
                rationale="B happened in 2001.",
                paraphrases=tuple(f"Paraphrase {index}?" for index in range(1, 10)),
            ),
        ),
        metadata={"topic": "example"},
    )


def test_bank_is_deterministic_and_contains_all_question_forms() -> None:
    first = build_evidence_bank(item(), slots=10, seed=17)
    second = build_evidence_bank(item(), slots=10, seed=17)
    assert first == second
    assert len(first) == 10
    assert first[0].question == "What happened in 2001?"
    assert {document.answer for document in first} == {"B"}
    assert all(document.candidate(10_000).kind == "nearby" for document in first)


def test_bank_rejects_missing_paraphrases() -> None:
    short = QuestionItem(
        item_id="q1",
        question="Original?",
        answer="A",
        rewrites=(Rewrite(question="Nearby?", answer="B"),),
    )
    with pytest.raises(ValueError, match="question forms"):
        build_evidence_bank(short)


def test_toy_manifest_is_valid() -> None:
    report = validate_manifest(ROOT / "examples" / "toy" / "manifest.jsonl")
    assert report.items == 1
    assert report.rewrites == 1
    assert len(report.sha256) == 64


def test_schema_validation_rejects_malformed_records() -> None:
    with pytest.raises(ValueError, match="title"):
        EvidenceRecord(question="Q?", title="", snippet="S", body="B")
    with pytest.raises(ValueError, match="empty"):
        Rewrite(question="", answer="B")
    with pytest.raises(ValueError, match="At least one"):
        QuestionItem(item_id="q", question="Q?", answer="A", rewrites=())
    with pytest.raises(ValueError, match="Unknown candidate"):
        SearchCandidate(kind="bad", title="T", url="u", snippet="", rank_hint=1)
    with pytest.raises(ValueError, match="both"):
        Outcome("q", 1, True, True, True)


def test_evidence_style_and_selection_validation() -> None:
    prose = build_evidence_bank(item(), slots=2, style="natural_prose")
    assert prose[0].answer == "B"
    assert "Answer:" not in prose[0].snippet
    with pytest.raises(ValueError, match="Unknown evidence style"):
        build_evidence_bank(item(), style="invalid")
    with pytest.raises(ValueError, match="unavailable"):
        build_evidence_bank(item(), rewrite_index=4)


def test_explicit_rewrite_zero_overrides_selected_rewrite() -> None:
    base = item()
    second = Rewrite(
        question="Second nearby question?",
        answer="C",
        paraphrases=tuple(f"Second paraphrase {index}?" for index in range(1, 10)),
        attempt_index=1,
    )
    selected = QuestionItem(
        item_id=base.item_id,
        question=base.question,
        answer=base.answer,
        rewrites=(*base.rewrites, second),
        selected_rewrite_index=1,
    )
    assert build_evidence_bank(selected, slots=1)[0].answer == "C"
    assert build_evidence_bank(selected, slots=1, rewrite_index=0)[0].answer == "B"


def test_prepared_evidence_is_used_verbatim() -> None:
    prepared = QuestionItem(
        item_id="q2",
        question="Original?",
        answer="A",
        rewrites=(
            Rewrite(
                question="Nearby?",
                answer="B",
                evidence_records=(
                    EvidenceRecord(
                        question="Nearby wording?",
                        title="Prepared title",
                        snippet="Prepared snippet",
                        body="Prepared body",
                        source="Prepared source",
                    ),
                ),
            ),
        ),
    )
    bank = build_evidence_bank(prepared, slots=1)
    assert bank[0].question == "Nearby wording?"
    assert bank[0].title == "Prepared title"
    assert bank[0].snippet == "Prepared snippet"
    assert bank[0].body == "Prepared body"
    assert bank[0].source == "Prepared source"
    with pytest.raises(ValueError, match="prepared records"):
        build_evidence_bank(prepared, slots=2)


def test_manifest_can_require_prepared_evidence(tmp_path: Path) -> None:
    report = validate_manifest(
        ROOT / "examples" / "toy" / "manifest.jsonl",
        require_prepared_evidence=True,
    )
    assert report.items == 1

    unprepared = tmp_path / "unprepared.jsonl"
    unprepared.write_text(
        json.dumps(
            {
                "item_id": "q",
                "question": "Original?",
                "answer": "A",
                "rewrites": [
                    {
                        "question": "Nearby?",
                        "answer": "B",
                        "paraphrases": [f"Nearby {index}?" for index in range(9)],
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="prepared evidence"):
        validate_manifest(unprepared, require_prepared_evidence=True)
    duplicate = tmp_path / "duplicate.jsonl"
    row = (ROOT / "examples" / "toy" / "manifest.jsonl").read_text()
    duplicate.write_text(row + row)
    with pytest.raises(ValueError, match="unique"):
        validate_manifest(duplicate)
