"""Data-isolation and archived-evidence tests without a PyTorch dependency."""

import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from intentlab.pretraining import (
    development_rows,
    labelled_subset,
    mask_channels,
    participant_metrics,
    summarise_pairs,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/pretraining-v1"


def test_development_rows_exclude_test_and_reject_participant_overlap():
    subjects = np.repeat([1, 2, 3], 4)
    splits = np.repeat(["train", "validation", "test"], 4)
    source, valid = development_rows(subjects, splits)
    np.testing.assert_array_equal(source, np.arange(4))
    np.testing.assert_array_equal(valid, np.arange(4, 8))
    assert not set(source) & set(valid)
    subjects[-1] = 1
    with pytest.raises(ValueError, match="overlap"):
        development_rows(subjects, splits)


@pytest.mark.parametrize("fraction", [0.1, 1.0])
def test_budget_is_per_source_person_and_class_and_reproducible(fraction):
    subjects = np.repeat([1, 2], 44)
    labels = np.tile(np.repeat([0, 1], 22), 2)
    positions = labelled_subset(labels, subjects, fraction, 2027)
    np.testing.assert_array_equal(positions, labelled_subset(labels, subjects, fraction, 2027))
    assert len(np.unique(positions)) == len(positions)
    for person in (1, 2):
        for label in (0, 1):
            assert np.sum((subjects[positions] == person) & (labels[positions] == label)) == max(
                1, int(22 * fraction)
            )
    if fraction == 1:
        np.testing.assert_array_equal(positions, np.arange(88))


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1])
def test_invalid_label_budgets_fail(fraction):
    with pytest.raises(ValueError):
        labelled_subset([0, 1], [1, 1], fraction, 1)


def test_mask_hides_whole_channels_only_and_does_not_mutate_input():
    x = np.arange(10 * 9 * 20, dtype=np.float32).reshape(10, 9, 20) + 1
    original = x.copy()
    masked, mask = mask_channels(x, np.random.default_rng(7), 2)
    assert (mask.sum(axis=1) == 2).all()
    assert (masked[mask] == 0).all()
    np.testing.assert_array_equal(masked[~mask], x[~mask])
    np.testing.assert_array_equal(x, original)
    second, second_mask = mask_channels(x, np.random.default_rng(7), 2)
    np.testing.assert_array_equal(masked, second)
    np.testing.assert_array_equal(mask, second_mask)


@pytest.mark.parametrize("count", [0, 9, 10])
def test_mask_must_hide_some_but_not_all_channels(count):
    with pytest.raises(ValueError):
        mask_channels(np.ones((4, 9, 20)), np.random.default_rng(7), count)


def test_metric_weights_people_equally():
    # One perfect person with 2 trials, one always-wrong person with 6 trials.
    metrics = participant_metrics([0, 1] * 4, [0, 1, 1, 0, 1, 0, 1, 0], [1, 1, 2, 2, 2, 2, 2, 2])
    assert metrics["participant_balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["participant_brier"] == pytest.approx(0.5)


def pair_records():
    rows = []
    for seed in (1, 2, 3):
        for arm in ("scratch", "masked_pretraining"):
            values = [0.5, 0.5] if arm == "scratch" else [0.6, 0.7]
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "label_fraction": 0.1,
                    "labelled_trials": 10,
                    "validation": {
                        "participants": [
                            {"subject": p, "balanced_accuracy": v}
                            for p, v in zip([1, 2], values, strict=True)
                        ]
                    },
                }
            )
    return rows


def test_bootstrap_averages_seeds_within_people_before_resampling():
    rows = pair_records()
    summary = summarise_pairs(rows, 500, 42)[0]
    assert summary["paired_difference"] == pytest.approx(0.15)
    assert summary["participants"] == 2  # never 2 people x 3 seeds
    one_seed = summarise_pairs([r for r in rows if r["seed"] == 1], 500, 42)[0]
    np.testing.assert_allclose(summary["paired_difference_95_ci"], one_seed["paired_difference_95_ci"])


def test_pairing_rejects_missing_seed_or_changed_participant_identity():
    rows = pair_records()
    with pytest.raises(ValueError, match="Exactly one"):
        summarise_pairs(rows[:-1])
    rows = copy.deepcopy(rows)
    rows[-1]["validation"]["participants"][-1]["subject"] = 9
    with pytest.raises(ValueError, match="identities"):
        summarise_pairs(rows)


def test_archived_pretraining_membership_uses_only_source_trials():
    manifest = json.loads((OUTPUT / "split_manifest.json").read_text())
    rows = pq.read_table(ROOT / "data/derived/trials.parquet").to_pylist()
    by_id = {r["id"]: r for r in rows}
    source = manifest["pretraining_trial_ids"]
    valid = manifest["validation_trial_ids"]
    assert len(source) == len(set(source)) == 2832
    assert len(valid) == len(set(valid)) == 989
    assert {by_id[i]["split"] for i in source} == {"train"}
    assert {by_id[i]["split"] for i in valid} == {"validation"}
    assert not {by_id[i]["subject"] for i in source} & {by_id[i]["subject"] for i in valid}
    source_y = np.asarray([by_id[i]["label"] for i in source])
    source_subjects = np.asarray([by_id[i]["subject"] for i in source])
    assert len(manifest["labelled_subsets"]) == 6
    for subset in manifest["labelled_subsets"]:
        chosen = labelled_subset(source_y, source_subjects, subset["label_fraction"], subset["seed"])
        assert subset["trial_ids"] == [source[i] for i in chosen]


def test_archived_predictions_reproduce_every_result_and_pair():
    report = json.loads((OUTPUT / "evaluation.json").read_text())
    with (OUTPUT / "predictions.csv").open() as f:
        predictions = list(csv.DictReader(f))
    assert len(report["results"]) == 12
    assert len(predictions) == 12 * 989
    assert {p["split"] for p in predictions} == {"validation"}
    assert report["status"] == "exploratory_validation_only"
    assert report["test_evaluated"] is False
    assert report["production_model_changed"] is False
    assert report["counts"] == {
        "training_participants": 63,
        "training_trials": 2832,
        "training_epoch_hours": 2.36,
        "validation_participants": 22,
        "validation_trials": 989,
    }
    for record in report["results"]:
        selected = [
            p
            for p in predictions
            if (int(p["seed"]), p["arm"], float(p["label_fraction"]))
            == (record["seed"], record["arm"], record["label_fraction"])
        ]
        assert len(selected) == len({p["trial_id"] for p in selected}) == 989
        rebuilt = participant_metrics(
            [int(p["label"]) for p in selected],
            [float(p["probability"]) for p in selected],
            [int(p["subject"]) for p in selected],
        )
        assert rebuilt == record["validation"]
        assert len(record["supervised_history"]) == 20
    assert summarise_pairs(report["results"], 2000, 5027) == report["summary"]


def test_archived_code_data_protocol_and_production_hashes_match():
    report = json.loads((OUTPUT / "evaluation.json").read_text())
    for name, expected in json.loads((OUTPUT / "checksums.json").read_text()).items():
        assert hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest() == expected
    for name, expected in report["run"]["input_and_code_sha256"].items():
        if name == "data/derived/epochs.npz":  # full archive is intentionally absent in Git/CI
            continue
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    for name, expected in report["run"]["production_artifacts_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
