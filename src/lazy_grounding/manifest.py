"""Structural validation for locally prepared experiment manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lazy_grounding.io import read_jsonl, sha256_file
from lazy_grounding.normalization import answers_exactly_match
from lazy_grounding.schemas import QuestionItem


@dataclass(frozen=True, slots=True)
class ManifestReport:
    path: str
    sha256: str
    items: int
    rewrites: int


def validate_manifest(
    path: Path,
    *,
    required_question_forms: int = 10,
    require_prepared_evidence: bool = False,
) -> ManifestReport:
    rows = read_jsonl(path)
    items = [QuestionItem.from_dict(row) for row in rows]
    item_ids = [item.item_id for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Manifest item IDs must be unique")
    rewrite_count = 0
    for item in items:
        for rewrite in item.rewrites:
            rewrite_count += 1
            if answers_exactly_match(item.answer, rewrite.answer):
                raise ValueError(
                    f"{item.item_id}: rewrite answer does not change the original answer"
                )
            available_forms = max(
                1 + len(rewrite.paraphrases),
                len(rewrite.evidence_records),
            )
            if available_forms < required_question_forms:
                raise ValueError(
                    f"{item.item_id}: rewrite {rewrite.attempt_index} has "
                    f"{available_forms} question forms; "
                    f"{required_question_forms} required"
                )
        selected = item.selected_rewrite
        if require_prepared_evidence and len(selected.evidence_records) < required_question_forms:
            raise ValueError(
                f"{item.item_id}: selected rewrite requires {required_question_forms} "
                "prepared evidence records for the paper protocol"
            )
    return ManifestReport(
        path=str(path.resolve()),
        sha256=sha256_file(path),
        items=len(items),
        rewrites=rewrite_count,
    )
