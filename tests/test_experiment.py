from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lazy_grounding.agents.providers import ModelReply
from lazy_grounding.config import AgentConfig, ExperimentConfig, RetrievalConfig, SummarizerConfig
from lazy_grounding.experiment import ExperimentRunner, _completed_run, score_run_directory
from lazy_grounding.io import atomic_write_json
from lazy_grounding.retrieval.dense import DenseRanker
from lazy_grounding.schemas import QuestionItem, Rewrite, SearchCandidate
from lazy_grounding.tools import UnavailablePageReader


class Encoder:
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


class Backend:
    def search(self, query: str, *, limit: int) -> list[SearchCandidate]:
        return [
            SearchCandidate(
                kind="real",
                title="Real",
                url="https://example.org",
                snippet="Real snippet",
                rank_hint=1,
                doc_id="real",
            )
        ]


class Provider:
    model = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages: Sequence[Mapping[str, str]]) -> ModelReply:
        self.calls += 1
        if self.calls == 1:
            return ModelReply(
                '<tool_call>{"name":"search","arguments":{"query":["nearby"]}}</tool_call>'
            )
        return ModelReply("<answer>B</answer>")


def config() -> ExperimentConfig:
    return ExperimentConfig(
        benchmark="toy",
        split="test",
        sample_size=1,
        sample_seed=1,
        run_seed=2,
        replicates=1,
        retrieval=RetrievalConfig(backend="serper", real_candidates=1, nearby_records=2, top_k=2),
        agent=AgentConfig(provider="openai", model="fake", max_llm_calls=3),
        summarizer=SummarizerConfig(provider="openai_chat", model="summary"),
    )


def item() -> QuestionItem:
    return QuestionItem(
        item_id="toy",
        question="Original?",
        answer="A",
        rewrites=(
            Rewrite(
                question="Nearby?",
                answer="B",
                paraphrases=("Nearby paraphrase?",),
            ),
        ),
    )


def test_runner_writes_paired_records_and_scoring_builds_outcome(tmp_path: Path) -> None:
    runner = ExperimentRunner(
        config(),
        ranker=DenseRanker(Encoder()),
        backend_factory=Backend,
        scholar_factory=Backend,
        provider_factory=Provider,
        page_reader_factory=lambda: UnavailablePageReader("unused"),
    )
    runner.run([item()], tmp_path, workers=2)
    assert len(list((tmp_path / "runs").glob("*.json"))) == 2

    def unused_judge(_: Any) -> dict[str, Any]:
        raise AssertionError("Exact answers should not call the semantic judge")

    outcomes = score_run_directory(tmp_path, output_dir=tmp_path / "scored", judge=unused_judge)  # type: ignore[arg-type]
    assert len(outcomes) == 1
    assert not outcomes[0].clean_correct
    assert not outcomes[0].augmented_correct
    assert outcomes[0].nearby_adopted
    assert (tmp_path / "scored" / "outcomes.jsonl").exists()


def test_runner_resume_skips_complete_records(tmp_path: Path) -> None:
    runner = ExperimentRunner(
        config(),
        ranker=DenseRanker(Encoder()),
        backend_factory=Backend,
        scholar_factory=Backend,
        provider_factory=Provider,
        page_reader_factory=lambda: UnavailablePageReader("unused"),
    )
    runner.run([item()], tmp_path)
    paths = sorted((tmp_path / "runs").glob("*.json"))
    mtimes = [path.stat().st_mtime_ns for path in paths]
    runner.run([item()], tmp_path, resume=True)
    assert [path.stat().st_mtime_ns for path in paths] == mtimes


def test_completed_run_rejects_malformed_or_mismatched_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    assert not _completed_run(path, "item", 1, "clean")
    path.write_text("not json")
    assert not _completed_run(path, "item", 1, "clean")
    atomic_write_json(path, [])
    assert not _completed_run(path, "item", 1, "clean")
    atomic_write_json(
        path,
        {"item_id": "item", "replicate": 1, "arm": "clean", "result": {"answer": "A"}},
    )
    assert _completed_run(path, "item", 1, "clean")
    assert not _completed_run(path, "other", 1, "clean")
