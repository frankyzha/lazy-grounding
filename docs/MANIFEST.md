# Private manifest schema

Paper manifests are JSON Lines files with one object per benchmark question.
They remain local because they contain benchmark questions and generated
benchmark-targeted evidence.

```json
{
  "item_id": "stable-upstream-id",
  "benchmark": "xbench-deepsearch",
  "question": "Original benchmark question",
  "answer": "Original answer",
  "selected_rewrite_index": 0,
  "rewrites": [
    {
      "attempt_index": 0,
      "question": "Verified nearby question",
      "answer": "Different nearby answer",
      "rationale": "Why the nearby answer supports the nearby question",
      "evidence_records": [
        {
          "question": "Source-form nearby question or paraphrase",
          "title": "Exact search-result title",
          "snippet": "Exact search-result snippet",
          "body": "Exact webpage body returned on visit",
          "source": "Displayed source label"
        }
      ]
    }
  ],
  "metadata": {
    "topic": "Shared lookup topic",
    "answer_type": "person"
  }
}
```

The selected rewrite must contain ten prepared evidence records for the main
paper protocol. Rewrite answers must differ from the original answer under the
same deterministic normalization used by the scorer. Item IDs must be unique.
`selected_rewrite_index` records the output of the original rewrite-selection
procedure and is held fixed across clean/augmented arms and all replicates.

Validate without printing record contents:

```bash
uv run lazy-grounding validate-data data/xbench.jsonl \
  --question-forms 10 --require-prepared-evidence
```

The validator reports only counts, the resolved path, and a SHA-256 digest.
