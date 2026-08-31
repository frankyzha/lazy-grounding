# Reproducibility protocol

This document specifies the main experiment independently of shell history or
provider defaults. Any deviation should be recorded in the output manifest.

## Experimental unit

- Benchmarks: XBench DeepSearch, GAIA, BrowseComp+, and Humanity's Last Exam.
- Sample: 100 fixed questions per benchmark, selected with seed `20260524`.
- Models: Tongyi-DeepResearch-30B-A3B, GPT-5 Mini, and Gemini 3 Flash.
- Replicates: three paired stochastic runs per model and benchmark.
- Arms: clean and augmented. Both arms ask the same original question.

## Nearby evidence

Each item has verified answer-changing rewrites. The private manifest records
which rewrite was selected and stores the exact ten evidence records used in
the experiment. One record uses the rewritten question; the other nine use
answer-preserving paraphrases. Only question wording varies across records.
The `selected_rewrite_index` field preserves the rewrite chosen by the original
single-transfer selection procedure; the runtime does not select a new rewrite.

Paper runs load these prepared records verbatim. The evidence bank is fixed
across stochastic replicates. This is validated by
`metadata.require_prepared_evidence: true` in the paper config.

## Retrieval

- Web benchmarks: Serper Google Search API, `location=United States`, `gl=us`,
  `hl=en`, and `num=10`.
- BrowseComp+: the benchmark's fixed local Lucene corpus and BM25 index.
- Augmented pool: 10 real candidates plus 10 nearby-evidence candidates.
- Reranker: `google/embeddinggemma-300m` at revision
  `57c266a740f537b4dc058e1b0cda161fd15afa75`, maximum length 512.
- Candidate text: title, source, date, and snippet joined with newlines.
- Pooling: attention-mask-aware mean pooling of final hidden states, followed
  by L2 normalization and query-candidate dot product.
- Returned budget: top 10 candidates in both arms.

Injected results borrow the top real result's display domain and include an
internal record ID in the path. Visiting that displayed record returns the
prepared nearby-evidence page. Removing the record suffix still routes to the
highest-ranked nearby record shown for that display base, matching the paper
configuration `contam_if_shown`.

## Agent

- Text ReAct tools: search, visit, and Google Scholar.
- Maximum ReAct calls: 120.
- Wall-time budget: 2,400 seconds, including a finalization margin.
- Maximum generated tokens per model call: 10,000.
- Temperature: 0.6; top-p: 0.95; presence penalty: 1.1 where supported.
- Webpage extractor: `gpt-5-nano`, configured separately from the evaluated
  base model.

When wall-time, context, or call budget is reached, the agent receives one
forced-finalization request and must return an `<answer>` from evidence already
in the conversation.

## Scoring

The scorer extracts the primary final answer. Trivial normalized matches are
resolved deterministically; all other cases use one explicit GPT-5.4 semantic
judge. The scorer compares against the original answer and only the rewrite
actually injected in that trajectory. It never silently substitutes a judge.

## Uncertainty

For each metric, the point estimate averages all question-replicate outcomes.
Sample SD is computed over the three run-level values. Percentile 95% intervals
use 100,000 paired cluster-bootstrap draws with seed `20260808`: sample the 100
question IDs with replacement and retain all replicates and both arms inside
each sampled cluster.

## Outputs

Every output directory contains a `run_manifest.json` with the resolved config,
input SHA-256, package version, Python version, platform, and creation time.
Resume mode accepts only complete trajectories and refuses an output directory
whose config or input hash differs.
