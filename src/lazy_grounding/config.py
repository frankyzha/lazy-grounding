"""Typed configuration loading with explicit environment-variable resolution."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_REFERENCE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
_MAX_TEMPERATURE = 2.0


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    backend: str
    embedding_model: str = "google/embeddinggemma-300m"
    embedding_revision: str = "57c266a740f537b4dc058e1b0cda161fd15afa75"
    embedding_device: str = "auto"
    embedding_max_length: int = 512
    real_candidates: int = 10
    nearby_records: int = 10
    top_k: int = 10
    browsecomp_index_path: str = ""
    display_url_style: str = "top_real"
    display_base_alias_mode: str = "contam_if_shown"
    query_adaptive_snippets: bool = True

    def __post_init__(self) -> None:
        if self.backend not in {"serper", "browsecomp_bm25"}:
            raise ValueError(f"Unsupported retrieval backend: {self.backend}")
        if self.display_url_style not in {"bank", "top_real"}:
            raise ValueError(f"Unsupported display_url_style: {self.display_url_style}")
        if self.display_base_alias_mode not in {"none", "contam_if_shown"}:
            raise ValueError(f"Unsupported display_base_alias_mode: {self.display_base_alias_mode}")
        if not self.embedding_model.strip() or not self.embedding_revision.strip():
            raise ValueError("embedding_model and embedding_revision must be explicit")
        for name in ("embedding_max_length", "real_candidates", "nearby_records", "top_k"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.backend == "browsecomp_bm25" and not self.browsecomp_index_path:
            raise ValueError("browsecomp_index_path is required for BrowseComp+")


@dataclass(frozen=True, slots=True)
class AgentConfig:
    provider: str
    model: str
    max_llm_calls: int = 120
    max_output_tokens: int = 10_000
    max_wall_time_seconds: int = 2_400
    max_context_tokens: int = 110 * 1_024
    temperature: float = 0.6
    top_p: float = 0.95
    presence_penalty: float = 1.1

    def __post_init__(self) -> None:
        if self.provider not in {"openai", "gemini", "vllm"}:
            raise ValueError(f"Unsupported provider: {self.provider}")
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if (
            self.max_llm_calls < 1
            or self.max_output_tokens < 1
            or self.max_wall_time_seconds < 1
            or self.max_context_tokens < 1
        ):
            raise ValueError("Agent budgets must be positive")
        if not 0 <= self.temperature <= _MAX_TEMPERATURE:
            raise ValueError("temperature must be in [0, 2]")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class SummarizerConfig:
    provider: str
    model: str
    max_output_tokens: int = 4_000
    timeout_seconds: int = 200

    def __post_init__(self) -> None:
        if self.provider != "openai_chat":
            raise ValueError(f"Unsupported summarizer provider: {self.provider}")
        if not self.model.strip():
            raise ValueError("summarizer model cannot be empty")
        if self.max_output_tokens < 1 or self.timeout_seconds < 1:
            raise ValueError("summarizer budgets must be positive")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    benchmark: str
    split: str
    sample_size: int
    sample_seed: int
    run_seed: int
    replicates: int
    retrieval: RetrievalConfig
    agent: AgentConfig
    summarizer: SummarizerConfig
    judge_model: str = "gpt-5.4"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.benchmark.strip() or not self.split.strip():
            raise ValueError("benchmark and split are required")
        if self.sample_size < 1 or self.replicates < 1:
            raise ValueError("sample_size and replicates must be positive")
        if not self.judge_model.strip():
            raise ValueError("judge_model must be explicit")


def _resolve_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    missing = sorted({name for name in _ENV_REFERENCE.findall(value) if name not in os.environ})
    if missing:
        raise ValueError(f"Missing environment variable(s): {', '.join(missing)}")
    return _ENV_REFERENCE.sub(lambda match: os.environ[match.group(1)], value)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    path: Path,
    *,
    model_key: str | None = None,
    dataset_key: str | None = None,
) -> ExperimentConfig:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    if "models" in raw or "datasets" in raw:
        if not model_key or not dataset_key:
            raise ValueError("model_key and dataset_key are required for a matrix config")
        models = raw.get("models") or {}
        datasets = raw.get("datasets") or {}
        if model_key not in models:
            raise ValueError(f"Unknown model key: {model_key}")
        if dataset_key not in datasets:
            raise ValueError(f"Unknown dataset key: {dataset_key}")
        selected = _deep_merge(raw.get("defaults") or {}, datasets[dataset_key])
        selected["agent"] = _deep_merge(selected.get("agent") or {}, models[model_key])
        raw = selected
    value = _resolve_environment(raw)
    return ExperimentConfig(
        benchmark=str(value["benchmark"]),
        split=str(value["split"]),
        sample_size=int(value["sample_size"]),
        sample_seed=int(value["sample_seed"]),
        run_seed=int(value["run_seed"]),
        replicates=int(value["replicates"]),
        retrieval=RetrievalConfig(**value["retrieval"]),
        agent=AgentConfig(**value["agent"]),
        summarizer=SummarizerConfig(**value["summarizer"]),
        judge_model=str(value.get("judge_model") or ""),
        metadata=dict(value.get("metadata") or {}),
    )
