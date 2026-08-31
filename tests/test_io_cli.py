from __future__ import annotations

import json
from pathlib import Path

from lazy_grounding.cli import main
from lazy_grounding.io import atomic_write_json, read_json, read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def test_atomic_json_and_jsonl_round_trip(tmp_path: Path) -> None:
    json_path = tmp_path / "nested" / "value.json"
    atomic_write_json(json_path, {"b": 2, "a": 1})
    assert read_json(json_path) == {"a": 1, "b": 2}
    jsonl_path = tmp_path / "rows.jsonl"
    write_jsonl(jsonl_path, ({"id": index} for index in range(2)))
    assert read_jsonl(jsonl_path) == [{"id": 0}, {"id": 1}]


def test_cli_validate_and_summarize(tmp_path: Path) -> None:
    main(["validate-data", str(ROOT / "examples" / "toy" / "manifest.jsonl")])
    output = tmp_path / "summary.json"
    main(
        [
            "summarize",
            str(ROOT / "examples" / "toy" / "outcomes.jsonl"),
            "--bootstrap-samples",
            "50",
            "--output",
            str(output),
        ]
    )
    assert json.loads(output.read_text())["replicate_count"] == 3
