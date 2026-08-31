"""Resumable execution and scoring of paired clean/augmented trajectories."""

from __future__ import annotations

import hashlib
import os
import platform
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from lazy_grounding import __version__
from lazy_grounding.agents import ModelProvider, ReactAgent, provider_from_config
from lazy_grounding.agents.providers import OpenAIChatProvider
from lazy_grounding.config import AgentConfig, ExperimentConfig
from lazy_grounding.evidence import build_evidence_bank
from lazy_grounding.io import atomic_write_json, read_json, read_jsonl, write_jsonl
from lazy_grounding.manifest import validate_manifest
from lazy_grounding.retrieval.augmented import AugmentedRetriever
from lazy_grounding.retrieval.base import SearchBackend, UnavailableSearchBackend
from lazy_grounding.retrieval.browsecomp import BrowseCompBackend
from lazy_grounding.retrieval.dense import DenseRanker, HuggingFaceEncoder
from lazy_grounding.retrieval.serper import SerperBackend
from lazy_grounding.schemas import Label, Outcome, QuestionItem, Rewrite
from lazy_grounding.scoring import OpenAIAnswerJudge, ScoreRequest, score_answer
from lazy_grounding.tools import (
    JinaFetcher,
    ScholarTool,
    SearchTool,
    SummarizingPageReader,
    UnavailablePageReader,
    VisitTool,
    telemetry,
)

ProviderFactory = Callable[[], ModelProvider]
BackendFactory = Callable[[], SearchBackend]
PageReaderFactory = Callable[[], SummarizingPageReader | UnavailablePageReader]


def _run_path(output_dir: Path, item_id: str, replicate: int, arm: str) -> Path:
    digest = hashlib.sha256(item_id.encode()).hexdigest()[:12]
    return output_dir / "runs" / f"{digest}_r{replicate:02d}_{arm}.json"


def _completed_run(path: Path, item_id: str, replicate: int, arm: str) -> bool:
    if not path.is_file():
        return False
    try:
        record = read_json(path)
        if not isinstance(record, Mapping):
            return False
        result = record.get("result") or {}
        return (
            record.get("item_id") == item_id
            and int(record.get("replicate") or 0) == replicate
            and record.get("arm") == arm
            and bool(str(result.get("answer") or "").strip())
        )
    except (OSError, TypeError, ValueError):
        return False


def _load_items(manifest_path: Path) -> list[QuestionItem]:
    return [QuestionItem.from_dict(row) for row in read_jsonl(manifest_path)]


class ExperimentRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        ranker: DenseRanker,
        backend_factory: BackendFactory,
        scholar_factory: BackendFactory,
        provider_factory: ProviderFactory,
        page_reader_factory: PageReaderFactory,
    ):
        self._config = config
        self._ranker = ranker
        self._backend_factory = backend_factory
        self._scholar_factory = scholar_factory
        self._provider_factory = provider_factory
        self._page_reader_factory = page_reader_factory

    def run_one(self, item: QuestionItem, *, replicate: int, augmented: bool) -> dict[str, Any]:
        provider = self._provider_factory()
        backend = self._backend_factory()
        evidence = build_evidence_bank(
            item,
            slots=self._config.retrieval.nearby_records,
            seed=self._config.run_seed,
            style=str(self._config.metadata.get("evidence_style") or "answer_field"),
        )
        retriever = AugmentedRetriever(
            backend,
            self._ranker,
            evidence,
            candidate_k=self._config.retrieval.real_candidates,
            top_k=self._config.retrieval.top_k,
            display_url_style=self._config.retrieval.display_url_style,
            display_base_alias_mode=self._config.retrieval.display_base_alias_mode,
            query_adaptive_snippets=self._config.retrieval.query_adaptive_snippets,
        )
        search = SearchTool(retriever, augmented=augmented)
        visit = VisitTool(
            evidence if augmented else (),
            self._page_reader_factory(),
            nearby_resolver=retriever,
            document_store=backend if isinstance(backend, BrowseCompBackend) else None,
        )
        scholar = ScholarTool(self._scholar_factory())
        agent = ReactAgent(
            provider,
            (search, visit, scholar),
            self._config.agent,
            max_context_tokens=self._config.agent.max_context_tokens,
            initial_instruction=str(self._config.metadata.get("initial_instruction") or ""),
        )
        result = agent.run(item.question)
        return {
            "schema_version": 1,
            "item_id": item.item_id,
            "benchmark": item.benchmark or self._config.benchmark,
            "replicate": replicate,
            "arm": "augmented" if augmented else "clean",
            "original_question": item.question,
            "original_answer": item.answer,
            "injected_rewrites": [asdict(item.selected_rewrite)],
            "agent": asdict(self._config.agent),
            "retrieval": asdict(self._config.retrieval),
            "run_seed": self._config.run_seed,
            "result": result.to_dict(),
            "telemetry": telemetry(search, visit),
        }

    def run(
        self,
        items: Iterable[QuestionItem],
        output_dir: Path,
        *,
        workers: int = 1,
        resume: bool = True,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be positive")
        jobs = []
        for item in items:
            for replicate in range(1, self._config.replicates + 1):
                for augmented in (False, True):
                    arm = "augmented" if augmented else "clean"
                    path = _run_path(output_dir, item.item_id, replicate, arm)
                    if resume and _completed_run(path, item.item_id, replicate, arm):
                        continue
                    jobs.append((item, replicate, augmented, path))
        failures = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.run_one, item, replicate=replicate, augmented=augmented): path
                for item, replicate, augmented, path in jobs
            }
            for future in as_completed(futures):
                path = futures[future]
                try:
                    atomic_write_json(path, future.result())
                except Exception as error:  # noqa: BLE001
                    failures.append(f"{path.name}: {type(error).__name__}: {error}")
        if failures:
            preview = "\n".join(failures[:10])
            raise RuntimeError(f"{len(failures)} trajectories failed:\n{preview}")


def build_runner(config: ExperimentConfig) -> ExperimentRunner:
    load_dotenv()
    encoder = HuggingFaceEncoder(
        config.retrieval.embedding_model,
        revision=config.retrieval.embedding_revision,
        device=config.retrieval.embedding_device,
        max_length=config.retrieval.embedding_max_length,
    )
    ranker = DenseRanker(encoder)
    serper_key = os.environ.get("SERPER_API_KEY", "")
    if config.retrieval.backend == "serper":
        if not serper_key:
            raise ValueError("SERPER_API_KEY is required for web-search benchmarks")

        def backend_factory() -> SearchBackend:
            return SerperBackend(serper_key)
    else:
        local_backend = BrowseCompBackend(Path(config.retrieval.browsecomp_index_path))

        def backend_factory() -> SearchBackend:
            return local_backend

    def scholar_factory() -> SearchBackend:
        if serper_key:
            return SerperBackend(serper_key, endpoint="https://google.serper.dev/scholar")
        return UnavailableSearchBackend(
            "Google Scholar is unavailable because SERPER_API_KEY is not configured"
        )

    summary_api_key = os.environ.get("SUMMARY_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    summary_api_base = os.environ.get("SUMMARY_API_BASE") or os.environ.get(
        "OPENAI_API_BASE", "https://api.openai.com/v1"
    )
    if config.retrieval.backend == "serper" and not summary_api_key:
        raise ValueError("SUMMARY_API_KEY or OPENAI_API_KEY is required for webpage extraction")
    summary_agent_config = AgentConfig(
        provider="vllm",
        model=config.summarizer.model,
        max_llm_calls=1,
        max_output_tokens=config.summarizer.max_output_tokens,
        max_wall_time_seconds=config.summarizer.timeout_seconds,
        max_context_tokens=config.agent.max_context_tokens,
        temperature=0.0,
        top_p=1.0,
        presence_penalty=0.0,
    )

    if config.retrieval.backend == "serper":

        def page_reader_factory() -> SummarizingPageReader | UnavailablePageReader:
            summarizer = OpenAIChatProvider(
                summary_agent_config,
                api_base=summary_api_base,
                api_key=summary_api_key,
            )
            return SummarizingPageReader(
                JinaFetcher(
                    api_key=os.environ.get("JINA_API_KEY") or os.environ.get("JINA_API_KEYS", "")
                ),
                summarizer,
            )
    else:

        def page_reader_factory() -> SummarizingPageReader | UnavailablePageReader:
            return UnavailablePageReader(
                "External webpage visits are disabled for BrowseComp+ local-corpus runs"
            )

    return ExperimentRunner(
        config,
        ranker=ranker,
        backend_factory=backend_factory,
        scholar_factory=scholar_factory,
        provider_factory=lambda: provider_from_config(config.agent),
        page_reader_factory=page_reader_factory,
    )


def run_from_files(
    config: ExperimentConfig,
    manifest_path: Path,
    output_dir: Path,
    *,
    workers: int,
    resume: bool,
) -> None:
    report = validate_manifest(
        manifest_path,
        required_question_forms=config.retrieval.nearby_records,
        require_prepared_evidence=bool(config.metadata.get("require_prepared_evidence")),
    )
    items = _load_items(manifest_path)
    if len(items) != config.sample_size:
        raise ValueError(
            f"Manifest has {len(items)} items; paper config requires exactly {config.sample_size}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema_version": 1,
        "data": asdict(report),
        "config": asdict(config),
    }
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        existing_protocol = {
            "schema_version": existing.get("schema_version"),
            "data": existing.get("data"),
            "config": existing.get("config"),
        }
        if existing_protocol != protocol:
            raise ValueError("Existing output directory uses a different data or run config")
    else:
        atomic_write_json(
            manifest_path,
            {
                **protocol,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "software": {
                    "lazy_grounding": __version__,
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                },
            },
        )
    build_runner(config).run(items, output_dir, workers=workers, resume=resume)


def _record_rewrites(record: Mapping[str, Any]) -> tuple[Rewrite, ...]:
    values = record.get("injected_rewrites") or record.get("rewrites") or ()
    return tuple(Rewrite.from_dict(value) for value in values)


def score_run_directory(
    run_dir: Path,
    *,
    output_dir: Path,
    judge: OpenAIAnswerJudge,
) -> list[Outcome]:
    records = [read_json(path) for path in sorted((run_dir / "runs").glob("*.json"))]
    scored_rows = []
    pairs: dict[tuple[str, int], dict[str, Label]] = {}
    for record in records:
        request = ScoreRequest(
            original_question=str(record["original_question"]),
            original_answer=str(record["original_answer"]),
            rewrites=_record_rewrites(record),
            model_response=str((record.get("result") or {}).get("answer") or ""),
            nearby_evidence_surfaced=bool(
                (record.get("telemetry") or {}).get("nearby_evidence_surfaced")
            ),
        )
        result = score_answer(request, judge=judge)
        scored_rows.append(
            {
                "item_id": record["item_id"],
                "replicate": record["replicate"],
                "arm": record["arm"],
                "score": result.to_dict(),
            }
        )
        key = (str(record["item_id"]), int(record["replicate"]))
        pairs.setdefault(key, {})[str(record["arm"])] = result.label

    outcomes = []
    for (item_id, replicate), pair in sorted(pairs.items()):
        if set(pair) != {"clean", "augmented"}:
            raise ValueError(f"Incomplete scored pair: {item_id}/r{replicate}")
        outcomes.append(
            Outcome(
                item_id=item_id,
                replicate=replicate,
                clean_correct=pair["clean"] is Label.CORRECT,
                augmented_correct=pair["augmented"] is Label.CORRECT,
                nearby_adopted=pair["augmented"] is Label.CR_R,
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "scores.jsonl", scored_rows)
    write_jsonl(output_dir / "outcomes.jsonl", (asdict(outcome) for outcome in outcomes))
    return outcomes
