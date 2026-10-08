"""Reconstruct validation/test coverage from published predictions without training.

Run from any directory: python examples/coverage_audit.py --output /tmp/coverage.csv
Each seed remains separate; repeated predictions are never counted as new trials.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from intentlab.adaptation import select_threshold, selective_metrics


def coverage_rows(predictions, config):
    groups = defaultdict(list)
    trial_metadata = {}
    for row in predictions:
        if row["split"] not in {"validation", "test"}:
            raise ValueError("Only validation and test predictions belong in this exercise")
        key = (row["condition"], row["seed"], row["split"])
        groups[key].append(row)
        metadata = (row["subject"], row["split"], row["label"])
        if trial_metadata.setdefault(row["trial_id"], metadata) != metadata:
            raise ValueError("Trial metadata changed between repeated predictions")
    if not groups:
        raise ValueError("Prediction table is empty")
    people = {
        split: {r["subject"] for r in predictions if r["split"] == split} for split in ("validation", "test")
    }
    if people["validation"] & people["test"]:
        raise ValueError("Validation and test participants must be disjoint")
    output = []
    for condition, seed in sorted({(c, s) for c, s, _ in groups}):
        values = {}
        for split in ("validation", "test"):
            rows = groups[(condition, seed, split)]
            if not rows or len({r["trial_id"] for r in rows}) != len(rows):
                raise ValueError("Every condition and seed needs unique validation and test trials")
            values[split] = (
                np.array([r["label"] for r in rows]),
                np.array([r["probability"] for r in rows]),
            )
        choice = select_threshold(*values["validation"], config)
        for split in ("validation", "test"):
            for threshold in config["thresholds"]:
                point = selective_metrics(*values[split], threshold)
                output.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "split": split,
                        "threshold": threshold,
                        "trials": point["n"],
                        "accepted": point["retained"],
                        "correct": point["correct"],
                        "coverage": point["coverage"],
                        "accepted_accuracy": point["accuracy"],
                        "validation_rule_threshold": choice["threshold"],
                        "validation_target_met": choice["target_met"],
                    }
                )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New CSV file; existing files are preserved"
    )
    args = parser.parse_args()
    folder = ROOT / "artifacts/adaptation-v1"
    checksums = json.loads((folder / "checksums.json").read_text())
    for name in ("predictions.parquet", "evaluation.json"):
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != checksums[name]:
            raise SystemExit(f"Published study checksum mismatch: {name}")
    report = json.loads((folder / "evaluation.json").read_text())
    rows = coverage_rows(pq.read_table(folder / "predictions.parquet").to_pylist(), report["config"])
    with args.output.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} operating points to {args.output}")
    print("Seeds are separate; blank accepted accuracy means no predictions were accepted.")
    print("Threshold choices use validation only; test curves are descriptive.")


if __name__ == "__main__":
    main()
