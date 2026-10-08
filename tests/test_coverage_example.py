import copy
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from examples.coverage_audit import coverage_rows

ROOT = Path(__file__).resolve().parents[1]


def test_teaching_export_preserves_seeds_and_matches_archived_operating_points():
    folder = ROOT / "artifacts/adaptation-v1"
    report = json.loads((folder / "evaluation.json").read_text())
    output = coverage_rows(pq.read_table(folder / "predictions.parquet").to_pylist(), report["config"])
    assert len(output) == 4 * 3 * 2 * 10
    for result in report["results"]:
        for split, expected_n in (("validation", 660), ("test", 630)):
            point = next(
                row
                for row in output
                if (row["condition"], row["seed"], row["split"], row["threshold"])
                == (result["condition"], result["seed"], split, 0.75)
            )
            expected = result[split]["fixed_threshold"]
            assert point["trials"] == expected_n
            assert point["accepted"] == expected["retained"]
            assert point["correct"] == expected["correct"]
            assert point["accepted_accuracy"] == expected["accuracy"]


def small_example():
    return [
        {
            "condition": "example",
            "seed": 1,
            "split": split,
            "subject": person,
            "trial_id": f"{split}-{trial}",
            "label": trial % 2,
            "probability": 0.9 if trial % 2 else 0.1,
        }
        for split, person in (("validation", 1), ("test", 2))
        for trial in range(4)
    ]


CONFIG = {
    "thresholds": [0.5, 0.75, 0.95],
    "target_retained": 2,
    "target_coverage": 0.5,
    "target_accuracy": 0.75,
    "fallback_threshold": 0.75,
}


def test_test_outcomes_cannot_change_validation_choice_and_empty_accuracy_is_missing():
    rows = small_example()
    before = coverage_rows(rows, CONFIG)
    for row in rows:
        if row["split"] == "test":
            row["label"] = 1 - row["label"]
    after = coverage_rows(rows, CONFIG)
    assert {r["validation_rule_threshold"] for r in before + after} == {0.5}
    assert all(r["accepted_accuracy"] is None for r in before if r["threshold"] == 0.95)
    assert next(r for r in before if r["split"] == "test" and r["threshold"] == 0.5)["correct"] == 4
    assert next(r for r in after if r["split"] == "test" and r["threshold"] == 0.5)["correct"] == 0


def test_teaching_export_rejects_overlap_duplicates_and_changed_trial_identity():
    rows = small_example()
    with pytest.raises(ValueError, match="unique"):
        coverage_rows(rows + [rows[0]], CONFIG)
    changed = copy.deepcopy(rows)
    changed[4]["subject"] = 1
    with pytest.raises(ValueError, match="disjoint"):
        coverage_rows(changed, CONFIG)
    changed = copy.deepcopy(rows[0])
    changed.update(seed=2, label=1)
    with pytest.raises(ValueError, match="metadata"):
        coverage_rows(rows + [changed], CONFIG)
