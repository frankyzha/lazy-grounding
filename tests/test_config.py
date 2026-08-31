from __future__ import annotations

from pathlib import Path

import pytest

from lazy_grounding.config import (
    AgentConfig,
    ExperimentConfig,
    RetrievalConfig,
    SummarizerConfig,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_load_paper_config() -> None:
    config = load_config(
        ROOT / "configs" / "paper" / "main.yaml",
        model_key="gpt5mini",
        dataset_key="xbench",
    )
    assert config.sample_size == 100
    assert config.agent.max_llm_calls == 120
    assert config.retrieval.top_k == 10
    assert config.retrieval.embedding_revision == "57c266a740f537b4dc058e1b0cda161fd15afa75"
    assert config.judge_model == "gpt-5.4"


def test_missing_required_environment_variable_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BROWSECOMP_INDEX_PATH", raising=False)
    with pytest.raises(ValueError, match="BROWSECOMP_INDEX_PATH"):
        load_config(
            ROOT / "configs" / "paper" / "main.yaml",
            model_key="gpt5mini",
            dataset_key="browsecomp_plus",
        )


def test_matrix_selection_and_value_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model_key"):
        load_config(ROOT / "configs" / "paper" / "main.yaml")
    with pytest.raises(ValueError, match="Unknown model"):
        load_config(
            ROOT / "configs" / "paper" / "main.yaml",
            model_key="missing",
            dataset_key="xbench",
        )
    with pytest.raises(ValueError, match="Unsupported retrieval"):
        RetrievalConfig(backend="wrong")
    with pytest.raises(ValueError, match="display_url_style"):
        RetrievalConfig(backend="serper", display_url_style="wrong")
    with pytest.raises(ValueError, match="display_base_alias_mode"):
        RetrievalConfig(backend="serper", display_base_alias_mode="wrong")
    with pytest.raises(ValueError, match="embedding_model"):
        RetrievalConfig(backend="serper", embedding_revision="")
    with pytest.raises(ValueError, match="positive"):
        RetrievalConfig(backend="serper", top_k=0)
    with pytest.raises(ValueError, match="browsecomp_index_path"):
        RetrievalConfig(backend="browsecomp_bm25")
    with pytest.raises(ValueError, match="Unsupported provider"):
        AgentConfig(provider="wrong", model="x")
    with pytest.raises(ValueError, match="model cannot be empty"):
        AgentConfig(provider="openai", model="")
    with pytest.raises(ValueError, match="budgets"):
        AgentConfig(provider="openai", model="x", max_llm_calls=0)
    with pytest.raises(ValueError, match="temperature"):
        AgentConfig(provider="openai", model="x", temperature=3)
    with pytest.raises(ValueError, match="top_p"):
        AgentConfig(provider="openai", model="x", top_p=0)
    with pytest.raises(ValueError, match="judge_model"):
        ExperimentConfig(
            benchmark="b",
            split="s",
            sample_size=1,
            sample_seed=1,
            run_seed=1,
            replicates=1,
            retrieval=RetrievalConfig(backend="serper"),
            agent=AgentConfig(provider="openai", model="x"),
            summarizer=SummarizerConfig(provider="openai_chat", model="summary"),
            judge_model="",
        )
    with pytest.raises(ValueError, match="summarizer provider"):
        SummarizerConfig(provider="wrong", model="summary")

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- not\n- a mapping\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(invalid)
