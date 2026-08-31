# Lazy Grounding

[![CI](https://github.com/frankyzha/lazy-grounding/actions/workflows/ci.yml/badge.svg)](https://github.com/frankyzha/lazy-grounding/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](pyproject.toml)

Official implementation of **Lazy Grounding: Attacking Search Agents with
Factual Evidence** (EMNLP 2026).

Search agents can fail without false documents. This repository evaluates
whether an agent transfers a factual answer from a nearby question to the
original question after nearby evidence enters its retrieval stream.

## Method

For an original question `q` with answer `a`, we construct a verified nearby
question `q'` with a different answer `b`. Both experimental arms still ask
`q`:

- **Clean:** the agent searches the ordinary retrieval environment.
- **Augmented:** factual records supporting `(q', b)` are added to the search
  candidate pool before dense reranking.

The main metrics are clean accuracy, augmented accuracy, accuracy drop
(`clean - augmented`), rewrite-answer adoption (RAA), and RAA conditioned on
whether the paired clean trajectory was correct or wrong (RAA-C/F).

## Main results

Values are percentages reported as mean +/- sample SD over three stochastic
runs. The paper contains confidence intervals and complete analysis.

| Model | Benchmark | Clean | Aug. | RAA | RAA-C/F |
|---|---|---:|---:|---:|---:|
| Tongyi Deep Research | XBench | 69.3 +/- 2.1 | 52.0 +/- 6.1 | 27.0 +/- 5.6 | 20.7 / 41.3 |
| Tongyi Deep Research | GAIA | 66.0 +/- 6.6 | 57.3 +/- 0.6 | 17.7 +/- 3.1 | 14.1 / 24.5 |
| Tongyi Deep Research | BrowseComp+ | 27.0 +/- 8.2 | 19.3 +/- 0.6 | 36.3 +/- 24.0 | 28.4 / 39.3 |
| Tongyi Deep Research | HLE | 28.7 +/- 6.1 | 26.7 +/- 5.1 | 17.7 +/- 3.1 | 25.6 / 14.5 |
| GPT-5 Mini | XBench | 65.0 +/- 6.0 | 52.7 +/- 1.5 | 23.0 +/- 3.6 | 19.0 / 30.5 |
| GPT-5 Mini | GAIA | 59.0 +/- 4.4 | 53.3 +/- 5.8 | 22.0 +/- 9.6 | 14.7 / 32.5 |
| GPT-5 Mini | BrowseComp+ | 31.7 +/- 3.2 | 22.0 +/- 8.2 | 29.0 +/- 2.6 | 33.7 / 26.8 |
| GPT-5 Mini | HLE | 30.7 +/- 5.5 | 24.3 +/- 3.1 | 15.0 +/- 2.6 | 13.0 / 15.9 |
| Gemini 3 Flash | XBench | 74.3 +/- 2.1 | 71.0 +/- 1.7 | 7.7 +/- 2.1 | 6.7 / 10.4 |
| Gemini 3 Flash | GAIA | 62.3 +/- 8.5 | 57.0 +/- 4.6 | 10.7 +/- 3.1 | 9.6 / 12.4 |
| Gemini 3 Flash | BrowseComp+ | 43.0 +/- 6.1 | 52.3 +/- 2.3 | 5.0 +/- 2.6 | 6.2 / 4.1 |
| Gemini 3 Flash | HLE | 46.7 +/- 2.5 | 44.7 +/- 2.5 | 13.7 +/- 5.5 | 10.0 / 16.9 |

## Installation

Python 3.10 is the reference environment. Dependencies are locked with `uv`.

```bash
git clone https://github.com/frankyzha/lazy-grounding.git
cd lazy-grounding
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra dev
```

Install the optional components required for a paper run:

```bash
uv sync --extra ranker       # EmbeddingGemma dense reranking
uv sync --extra browsecomp   # BrowseComp+ local Lucene index
uv sync --extra gemini       # Gemini API adapter
uv sync --all-extras         # Complete development/reproduction environment
```

BrowseComp+ retrieval additionally requires Java 11 or newer for Pyserini.

## Offline smoke test

The toy fixture contains no benchmark questions and makes no API calls:

```bash
uv run lazy-grounding validate-data examples/toy/manifest.jsonl
uv run lazy-grounding summarize examples/toy/outcomes.jsonl \
  --bootstrap-samples 1000 --output outputs/toy-summary.json
uv run pytest
```

## Reproduce an experiment

1. Obtain each benchmark from its official source.
2. Prepare a private manifest with verified rewrites and the exact evidence
   records to surface. Paper configs require ten prepared records per question.
3. Copy `.env.example` to `.env` and set only the providers used.
4. Validate, run, score, and summarize:

```bash
uv run lazy-grounding validate-data data/xbench.jsonl \
  --require-prepared-evidence
uv run lazy-grounding validate-config configs/paper/main.yaml \
  --model gpt5mini --dataset xbench
uv run lazy-grounding run \
  --config configs/paper/main.yaml --model gpt5mini --dataset xbench \
  --manifest data/xbench.jsonl --output outputs/gpt5mini-xbench --workers 4
uv run lazy-grounding score \
  --runs outputs/gpt5mini-xbench \
  --output outputs/gpt5mini-xbench/scored --judge-model gpt-5.4
uv run lazy-grounding summarize outputs/gpt5mini-xbench/scored/outcomes.jsonl \
  --bootstrap-samples 100000 --seed 20260808 \
  --output outputs/gpt5mini-xbench/summary.json
```

See [the exact protocol](docs/REPRODUCIBILITY.md), [manifest
schema](docs/MANIFEST.md), [data policy](docs/DATA.md), and [third-party
terms](docs/THIRD_PARTY.md).

## Statistical procedure

The statistical unit is the benchmark question. Each bootstrap draw samples
100 question IDs with replacement while retaining all three replicates and both
clean/augmented arms for every sampled question. The reported percentile 95%
interval uses 100,000 draws. Standard deviations are sample SDs across the
three run-level metric values.

## Responsible release

Do not publish benchmark-targeted nearby-evidence pages to the public web.
Persistent indexing could contaminate future evaluations. This repository uses
a controlled retrieval layer to measure agent susceptibility conditional on
nearby evidence being surfaced.

Benchmark questions, generated evidence, raw search results, full trajectories,
model caches, and credentials are deliberately excluded. See [SECURITY.md](SECURITY.md).

## Development

```bash
uv sync --all-extras
make check
make build
```

CI runs formatting, linting, strict type checking, tests, branch coverage, and
package builds on Python 3.10 and 3.12.

## Citation

```bibtex
@inproceedings{zhang2026lazygrounding,
  title = {Lazy Grounding: Attacking Search Agents with Factual Evidence},
  author = {Zhang, Yulin and Huang, Yukun and Chen, Sanxing and Lin, Tianyi and
            Yang, Ziang and Yin, Xunjian and Dhingra, Bhuwan},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in
               Natural Language Processing},
  year = {2026}
}
```

## License

Original code is released under the [Apache License 2.0](LICENSE). Models,
benchmarks, search results, and optional third-party software retain their own
licenses and terms. See [NOTICE](NOTICE) and [THIRD_PARTY.md](docs/THIRD_PARTY.md).
