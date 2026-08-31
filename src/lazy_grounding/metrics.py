"""Paper metrics and paired question-cluster bootstrap intervals."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from lazy_grounding.schemas import Outcome

METRIC_NAMES = ("clean", "augmented", "drop", "raa", "raa_c", "raa_f")


@dataclass(frozen=True, slots=True)
class MetricEstimate:
    estimate: float
    run_sd: float
    ci95: tuple[float, float]


@dataclass(frozen=True, slots=True)
class MetricSummary:
    question_count: int
    replicate_count: int
    bootstrap_samples: int
    seed: int
    metrics: Mapping[str, MetricEstimate]
    run_metrics: tuple[Mapping[str, float], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "question_count": self.question_count,
            "replicate_count": self.replicate_count,
            "bootstrap_samples": self.bootstrap_samples,
            "seed": self.seed,
            "bootstrap_unit": "question",
            "replicates_retained_within_cluster": True,
            "interval": "percentile_95",
            "metrics": {name: asdict(value) for name, value in self.metrics.items()},
            "run_metrics": list(self.run_metrics),
        }


def _safe_ratio(
    numerator: NDArray[np.float64], denominator: NDArray[np.float64]
) -> NDArray[np.float64]:
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def _metrics(values: NDArray[np.float64], axes: tuple[int, ...]) -> NDArray[np.float64]:
    clean = values[..., 0]
    augmented = values[..., 1]
    adopted = values[..., 2]
    clean_total = clean.sum(axis=axes)
    failed = 1.0 - clean
    failed_total = failed.sum(axis=axes)
    return np.stack(
        (
            clean.mean(axis=axes),
            augmented.mean(axis=axes),
            (clean - augmented).mean(axis=axes),
            adopted.mean(axis=axes),
            _safe_ratio((adopted * clean).sum(axis=axes), clean_total),
            _safe_ratio((adopted * failed).sum(axis=axes), failed_total),
        ),
        axis=-1,
    )


def outcomes_to_array(
    outcomes: Iterable[Outcome],
) -> tuple[tuple[str, ...], tuple[int, ...], NDArray[np.float64]]:
    rows = list(outcomes)
    if not rows:
        raise ValueError("At least one outcome is required")
    item_ids = tuple(sorted({row.item_id for row in rows}))
    replicates = tuple(sorted({row.replicate for row in rows}))
    expected = {(item_id, replicate) for item_id in item_ids for replicate in replicates}
    indexed: dict[tuple[str, int], Outcome] = {}
    for row in rows:
        key = (row.item_id, row.replicate)
        if key in indexed:
            raise ValueError(f"Duplicate outcome: item={row.item_id}, replicate={row.replicate}")
        indexed[key] = row
    missing = expected - set(indexed)
    if missing:
        preview = ", ".join(f"{item}/r{rep}" for item, rep in sorted(missing)[:5])
        raise ValueError(f"Incomplete question-replicate grid ({len(missing)} missing): {preview}")
    values = np.asarray(
        [
            [
                (
                    indexed[(item_id, replicate)].clean_correct,
                    indexed[(item_id, replicate)].augmented_correct,
                    indexed[(item_id, replicate)].nearby_adopted,
                )
                for replicate in replicates
            ]
            for item_id in item_ids
        ],
        dtype=np.float64,
    )
    return item_ids, replicates, values


def summarize_outcomes(
    outcomes: Iterable[Outcome],
    *,
    bootstrap_samples: int = 100_000,
    seed: int = 20260808,
    chunk_size: int = 5_000,
) -> MetricSummary:
    """Estimate metrics and bootstrap questions while retaining paired runs and arms.

    Values are returned as proportions. Multiply by 100 for percentage points.
    The reported standard deviation is the sample SD across replicate-level metric
    values; the confidence interval quantifies question-sampling uncertainty.
    """

    if bootstrap_samples < 1 or chunk_size < 1:
        raise ValueError("bootstrap_samples and chunk_size must be positive")
    item_ids, replicates, values = outcomes_to_array(outcomes)
    point = _metrics(values, axes=(0, 1))
    per_run = _metrics(values, axes=(0,))
    run_sd = (
        np.std(per_run, axis=0, ddof=1)
        if len(replicates) > 1
        else np.full(len(METRIC_NAMES), np.nan)
    )

    rng = np.random.default_rng(seed)
    draws = np.empty((bootstrap_samples, len(METRIC_NAMES)), dtype=np.float64)
    for start in range(0, bootstrap_samples, chunk_size):
        stop = min(start + chunk_size, bootstrap_samples)
        indices = rng.integers(0, len(item_ids), size=(stop - start, len(item_ids)))
        sampled = values[indices]
        draws[start:stop] = _metrics(sampled, axes=(1, 2))
    intervals = np.nanpercentile(draws, (2.5, 97.5), axis=0)

    metrics = {
        name: MetricEstimate(
            estimate=float(point[index]),
            run_sd=float(run_sd[index]),
            ci95=(float(intervals[0, index]), float(intervals[1, index])),
        )
        for index, name in enumerate(METRIC_NAMES)
    }
    run_metrics = tuple(
        {name: float(per_run[replicate_index, index]) for index, name in enumerate(METRIC_NAMES)}
        for replicate_index in range(len(replicates))
    )
    return MetricSummary(
        question_count=len(item_ids),
        replicate_count=len(replicates),
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        metrics=metrics,
        run_metrics=run_metrics,
    )
