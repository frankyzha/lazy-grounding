"""Command-line interface for validation and deterministic analysis."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from lazy_grounding.config import load_config
from lazy_grounding.experiment import run_from_files, score_run_directory
from lazy_grounding.io import atomic_write_json, read_jsonl
from lazy_grounding.manifest import validate_manifest
from lazy_grounding.metrics import summarize_outcomes
from lazy_grounding.schemas import Outcome
from lazy_grounding.scoring import OpenAIAnswerJudge


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lazy-grounding", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate_config = commands.add_parser("validate-config", help="Validate a paper YAML config")
    validate_config.add_argument("path", type=Path)
    validate_config.add_argument("--model")
    validate_config.add_argument("--dataset")

    validate_data = commands.add_parser("validate-data", help="Validate a local JSONL manifest")
    validate_data.add_argument("path", type=Path)
    validate_data.add_argument("--question-forms", type=int, default=10)
    validate_data.add_argument("--require-prepared-evidence", action="store_true")

    summarize = commands.add_parser("summarize", help="Compute repeated-run metrics and intervals")
    summarize.add_argument("path", type=Path, help="JSONL file of question-level outcomes")
    summarize.add_argument("--bootstrap-samples", type=int, default=100_000)
    summarize.add_argument("--seed", type=int, default=20260808)
    summarize.add_argument("--output", type=Path)

    run = commands.add_parser("run", help="Run paired clean and augmented trajectories")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--model")
    run.add_argument("--dataset")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--no-resume", action="store_true")

    score = commands.add_parser("score", help="Score completed trajectories with one judge")
    score.add_argument("--runs", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--judge-model", required=True)
    score.add_argument("--api-base", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "validate-config":
        config = load_config(args.path, model_key=args.model, dataset_key=args.dataset)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "benchmark": config.benchmark,
                    "agent": config.agent.model,
                    "retrieval_backend": config.retrieval.backend,
                },
                indent=2,
            )
        )
        return
    if args.command == "validate-data":
        report = validate_manifest(
            args.path,
            required_question_forms=args.question_forms,
            require_prepared_evidence=args.require_prepared_evidence,
        )
        print(json.dumps(asdict(report), indent=2))
        return
    if args.command == "summarize":
        outcomes = [Outcome.from_dict(row) for row in read_jsonl(args.path)]
        summary = summarize_outcomes(
            outcomes,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        result = summary.to_dict()
        if args.output:
            atomic_write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return
    if args.command == "run":
        run_from_files(
            load_config(args.config, model_key=args.model, dataset_key=args.dataset),
            args.manifest,
            args.output,
            workers=args.workers,
            resume=not args.no_resume,
        )
        return
    if args.command == "score":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for semantic scoring")
        judge = OpenAIAnswerJudge(
            api_key=api_key,
            model=args.judge_model,
            api_base=args.api_base
            or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        )
        outcomes = score_run_directory(args.runs, output_dir=args.output, judge=judge)
        print(json.dumps({"status": "ok", "outcomes": len(outcomes)}, indent=2))
        return
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
