from __future__ import annotations

import math

import pytest

from lazy_grounding.metrics import outcomes_to_array, summarize_outcomes
from lazy_grounding.schemas import Outcome


def outcomes() -> list[Outcome]:
    rows = []
    values = {
        "q1": ((1, 0, 1), (1, 1, 0), (1, 0, 1)),
        "q2": ((0, 0, 1), (0, 1, 0), (0, 0, 0)),
    }
    for item_id, replicates in values.items():
        for replicate, (clean, augmented, adopted) in enumerate(replicates, start=1):
            rows.append(
                Outcome(
                    item_id=item_id,
                    replicate=replicate,
                    clean_correct=bool(clean),
                    augmented_correct=bool(augmented),
                    nearby_adopted=bool(adopted),
                )
            )
    return rows


def test_summary_uses_trajectory_conditioning_for_raa_c_and_raa_f() -> None:
    summary = summarize_outcomes(outcomes(), bootstrap_samples=500, seed=7)
    estimates = {name: row.estimate for name, row in summary.metrics.items()}
    assert estimates == pytest.approx(
        {
            "clean": 0.5,
            "augmented": 1 / 3,
            "drop": 1 / 6,
            "raa": 0.5,
            "raa_c": 2 / 3,
            "raa_f": 1 / 3,
        }
    )
    assert len(summary.run_metrics) == 3
    assert summary.run_metrics[0]["drop"] == pytest.approx(0.5)
    assert summary.run_metrics[1]["drop"] == pytest.approx(-0.5)
    assert summary.metrics["clean"].run_sd == pytest.approx(0.0)


def test_bootstrap_is_reproducible_and_retains_question_clusters() -> None:
    first = summarize_outcomes(outcomes(), bootstrap_samples=1_000, seed=42)
    second = summarize_outcomes(outcomes(), bootstrap_samples=1_000, seed=42)
    assert first.to_dict() == second.to_dict()
    assert first.metrics["drop"].ci95 == pytest.approx((-1 / 3, 2 / 3))


def test_incomplete_grid_and_duplicate_rows_fail() -> None:
    rows = outcomes()
    with pytest.raises(ValueError, match="Incomplete"):
        outcomes_to_array(rows[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        outcomes_to_array([*rows, rows[0]])


def test_one_replicate_has_undefined_run_sd() -> None:
    rows = [row for row in outcomes() if row.replicate == 1]
    summary = summarize_outcomes(rows, bootstrap_samples=20)
    assert math.isnan(summary.metrics["clean"].run_sd)
