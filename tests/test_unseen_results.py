"""Recompute the archived study from CSV evidence without loading fitted models."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from intentlab.unseen import bootstrap_comparison

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "artifacts/unseen-v1"
PROTOCOL = ROOT / "experiments/unseen-v1"
MODELS = ("bandpower", "covariance_tangent")


def json_file(path):
    return json.loads(path.read_text())


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def assert_close(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            assert_close(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_close(left, right)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=1e-12, rel=1e-12)
    else:
        assert actual == expected


@pytest.fixture(scope="module")
def archive():
    result = {
        "evaluation": json_file(FOLDER / "evaluation.json"),
        "selection": json_file(FOLDER / "selection.json"),
        "provenance": json_file(FOLDER / "provenance.json"),
        "protocol": json_file(PROTOCOL / "protocol.json"),
        "manifest": json_file(PROTOCOL / "source_manifest.json"),
        "splits": json_file(PROTOCOL / "split_manifest.json")["participants"],
    }
    for split in ("validation", "test"):
        rows = csv_rows(FOLDER / f"{split}_predictions.csv")
        for row in rows:
            for field in ("label", "accepted"):
                row[field] = int(row[field])
            for field in ("left_probability", "right_probability", "threshold"):
                row[field] = float(row[field])
        result[split] = rows
    return result


def classify(rows):
    labels = [row["label"] for row in rows]
    probabilities = [row["right_probability"] for row in rows]
    guesses = [int(row["right_probability"] > row["left_probability"]) for row in rows]
    correct = [label == guess for label, guess in zip(labels, guesses, strict=True)]
    recall = [
        sum(hit for hit, label in zip(correct, labels, strict=True) if label == target) / labels.count(target)
        for target in (0, 1)
    ]
    positives = [score for score, label in zip(probabilities, labels, strict=True) if label == 1]
    negatives = [score for score, label in zip(probabilities, labels, strict=True) if label == 0]
    auc = sum(
        (positive > negative) + 0.5 * (positive == negative)
        for positive in positives
        for negative in negatives
    )
    return {
        "balanced_accuracy": sum(recall) / 2,
        "accuracy": sum(correct) / len(rows),
        "roc_auc": auc / (len(positives) * len(negatives)),
        "brier": sum((score - label) ** 2 for score, label in zip(probabilities, labels, strict=True))
        / len(rows),
        "confusion_matrix": [
            [
                sum(
                    label == truth and guess == prediction
                    for label, guess in zip(labels, guesses, strict=True)
                )
                for prediction in (0, 1)
            ]
            for truth in (0, 1)
        ],
    }


def participant_metrics(rows, threshold):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["participant"]].append(row)
    output = []
    for person, items in sorted(grouped.items()):
        retained = [
            row for row in items if max(row["left_probability"], row["right_probability"]) >= threshold
        ]
        correct = sum(
            int(row["right_probability"] > row["left_probability"]) == row["label"] for row in retained
        )
        output.append(
            {
                "participant": person,
                "cohort": items[0]["cohort"],
                "trials": len(items),
                "left_trials": sum(row["label"] == 0 for row in items),
                "right_trials": sum(row["label"] == 1 for row in items),
                **{key: value for key, value in classify(items).items() if key != "confusion_matrix"},
                "accepted": len(retained),
                "correct_accepted": correct,
                "coverage": len(retained) / len(items),
                "accepted_accuracy": correct / len(retained) if retained else None,
            }
        )
    return output


def selective(people, threshold):
    accepted = sum(person["accepted"] for person in people)
    correct = sum(person["correct_accepted"] for person in people)
    defined = [person["accepted_accuracy"] for person in people if person["accepted_accuracy"] is not None]
    return {
        "threshold": threshold,
        "accepted": accepted,
        "correct_accepted": correct,
        "pooled_accuracy": correct / accepted if accepted else None,
        "participant_macro_accuracy": sum(defined) / len(defined) if defined else None,
        "participant_macro_coverage": sum(person["coverage"] for person in people) / len(people),
        "pooled_coverage": accepted / sum(person["trials"] for person in people),
        "participants_with_acceptance": len(defined),
        "participants_without_acceptance": len(people) - len(defined),
        "minimum_accepted_per_participant": min(person["accepted"] for person in people),
    }


def test_archived_code_inputs_and_outputs_match_provenance(archive):
    provenance, selection, evaluation = (archive[key] for key in ("provenance", "selection", "evaluation"))
    assert provenance["raw_data_redistributed"] is False
    assert provenance["production_model_changed"] is evaluation["production_model_changed"] is False
    assert selection["test_participants_accessed"] is False
    assert provenance["fitted_model_sha256"] == selection["model_sha256"]
    assert provenance["protocol_commit"] == selection["protocol_commit"] == evaluation["protocol_commit"]
    assert provenance["selection_commit"] == evaluation["selection_commit"]
    assert provenance["code_sha256"] == selection["code_sha256"]
    for name, expected in provenance["code_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    assert set(provenance["protocol_sha256"]) == {
        "protocol.json",
        "PROTOCOL.md",
        "source_manifest.json",
        "split_manifest.json",
    }
    for name, expected in provenance["protocol_sha256"].items():
        assert hashlib.sha256((PROTOCOL / name).read_bytes()).hexdigest() == expected, name
    assert set(provenance["output_sha256"]) == {
        "selection.json",
        "validation_predictions.csv",
        "test_predictions.csv",
        "participant_metrics.csv",
        "evaluation.json",
    }
    for name, expected in provenance["output_sha256"].items():
        assert hashlib.sha256((FOLDER / name).read_bytes()).hexdigest() == expected, name
    assert (
        selection["validation_predictions_sha256"]
        == provenance["output_sha256"]["validation_predictions.csv"]
    )


def test_original_source_and_participant_counts_are_complete(archive):
    records = {row["participant"]: row for row in archive["splits"]}
    manifests = {row["participant"]: row for row in archive["manifest"]["participants"]}
    sources = archive["provenance"]["source"]
    assert len(sources) == len(records) == len(manifests) == 87
    assert {source["archive"]["participant"] for source in sources} == set(records)
    counts = Counter()
    for source in sources:
        original = source["archive"]
        person = original["participant"]
        assert original == manifests[person]
        assert source["dropped_trials"] == 0
        assert len(source["recordings"]) == 2
        assert {record["run"] for record in source["recordings"]} == {"R1acquisition", "R2acquisition"}
        for record in source["recordings"]:
            assert record["participant"] == person
            assert record["sampling_hz"] == 512
            assert record["left_cues"] > 0 and record["right_cues"] > 0
            assert len(record["selected_channels"]) == 27
            counts[records[person]["split"]] += record["left_cues"] + record["right_cues"]
    for split, expected in {"train": 45, "validation": 15, "test": 27}.items():
        saved = archive["evaluation"]["counts"][split]
        assert saved["participants"] == expected
        assert saved["trials"] == counts[split]
    fit_people = {source["archive"]["participant"] for source in archive["selection"]["source"]}
    assert len(archive["selection"]["source"]) == len(fit_people) == 60
    assert fit_people == {
        person for person, record in records.items() if record["split"] in {"train", "validation"}
    }
    assert sources[:60] == archive["selection"]["source"]
    final_people = {source["archive"]["participant"] for source in sources[60:]}
    assert len(sources[60:]) == len(final_people) == 27
    assert final_people == {person for person, record in records.items() if record["split"] == "test"}


def test_csv_rows_keep_identical_people_trials_and_thresholds_across_models(archive):
    splits = {row["participant"]: row for row in archive["splits"]}
    for split in ("validation", "test"):
        assert len(archive[split]) == 2 * archive["evaluation"]["counts"][split]["trials"]
        by_model = {name: [row for row in archive[split] if row["model"] == name] for name in MODELS}
        assert set(row["model"] for row in archive[split]) == set(MODELS)
        reference = None
        for name, rows in by_model.items():
            assert len({row["trial_id"] for row in rows}) == len(rows)
            identities = {
                (row["trial_id"], row["participant"], row["cohort"], row["run"], row["label"]) for row in rows
            }
            if reference is None:
                reference = identities
            assert identities == reference
            assert {row["participant"] for row in rows} == {
                person for person, record in splits.items() if record["split"] == split
            }
            for row in rows:
                assert row["cohort"] == splits[row["participant"]]["cohort"]
                assert row["run"] in {"R1acquisition", "R2acquisition"}
                assert row["label"] in {0, 1}
                probabilities = [row["left_probability"], row["right_probability"]]
                assert all(np.isfinite(value) and 0 <= value <= 1 for value in probabilities)
                assert sum(probabilities) == pytest.approx(1, abs=1e-12)
                threshold = archive["selection"]["models"][name]["threshold_selection"]["threshold"]
                assert row["threshold"] == threshold
                assert row["accepted"] == int(max(probabilities) >= threshold)


@pytest.mark.parametrize("split", ["validation", "test"])
@pytest.mark.parametrize("model", MODELS)
def test_metrics_recompute_independently_from_csv(archive, split, model):
    rows = [row for row in archive[split] if row["model"] == model]
    saved = archive["evaluation"]["models"][model][split]
    people = participant_metrics(rows, saved["selective"]["threshold"])
    assert saved["trials"] == len(rows)
    assert saved["participant_count"] == len(people)
    assert_close(saved["participants"], people)
    for output, field in (
        ("participant_macro_balanced_accuracy", "balanced_accuracy"),
        ("participant_macro_accuracy", "accuracy"),
        ("participant_macro_auc", "roc_auc"),
        ("participant_macro_brier", "brier"),
    ):
        assert saved[output] == pytest.approx(
            sum(person[field] for person in people) / len(people), abs=1e-12
        )
    pooled = classify(rows)
    assert_close({key: saved["pooled"][key] for key in pooled}, pooled)
    for index, record in enumerate(saved["pooled"]["calibration"]):
        members = [row for row in rows if min(int(row["right_probability"] * 10), 9) == index]
        assert record["count"] == len(members)
        expected_probability = (
            sum(row["right_probability"] for row in members) / len(members) if members else None
        )
        expected_fraction = sum(row["label"] for row in members) / len(members) if members else None
        assert_close(record["mean_probability"], expected_probability)
        assert_close(record["observed_right_fraction"], expected_fraction)
    assert_close(saved["selective"], selective(people, saved["selective"]["threshold"]))


def test_threshold_and_model_choice_recompute_using_validation_only(archive):
    selection = archive["selection"]
    rule = archive["protocol"]["selective_prediction"]
    scores = {}
    for model in MODELS:
        rows = [row for row in archive["validation"] if row["model"] == model]
        candidates = [
            selective(participant_metrics(rows, threshold), threshold) for threshold in rule["threshold_grid"]
        ]
        qualifying = [
            point
            for point in candidates
            if point["participant_macro_accuracy"] is not None
            and point["participant_macro_accuracy"] >= rule["minimum_participant_macro_accuracy"]
            and point["participant_macro_coverage"] >= rule["minimum_participant_macro_coverage"]
            and point["accepted"] >= rule["minimum_total_accepted"]
            and point["minimum_accepted_per_participant"] >= rule["minimum_accepted_every_participant"]
        ]
        expected_threshold = qualifying[0]["threshold"] if qualifying else rule["fallback"]
        saved = selection["models"][model]["threshold_selection"]
        assert saved["threshold"] == expected_threshold
        assert saved["validation_target_met"] is bool(qualifying)
        assert_close(saved["candidates"], candidates)
        people = participant_metrics(rows, 0.5)
        scores[model] = sum(person["balanced_accuracy"] for person in people) / len(people)
    expected_model = (
        "covariance_tangent" if scores["covariance_tangent"] > scores["bandpower"] else "bandpower"
    )
    assert selection["selection"]["model"] == archive["evaluation"]["selection"]["model"] == expected_model


def test_stratified_paired_bootstrap_and_primary_target_recompute(archive):
    by_model = {
        model: sorted(
            [row for row in archive["test"] if row["model"] == model], key=lambda row: row["trial_id"]
        )
        for model in MODELS
    }
    reference = by_model["bandpower"]
    probabilities = {
        model: [[row["left_probability"], row["right_probability"]] for row in rows]
        for model, rows in by_model.items()
    }
    expected = bootstrap_comparison(
        [row["label"] for row in reference],
        probabilities,
        [row["participant"] for row in reference],
        [row["cohort"] for row in reference],
        repeats=archive["protocol"]["bootstrap"]["repeats"],
        seed=archive["protocol"]["bootstrap"]["seed"],
    )
    assert expected["strata_counts"] == {"A": 19, "B": 6, "C": 2}
    assert_close(archive["evaluation"]["test_bootstrap"], expected)
    primary = archive["evaluation"]["selection"]
    assert primary["primary_criterion_met"] is (expected["model_intervals"][primary["model"]][0] > 0.5)


def test_participant_download_matches_all_test_people_and_preserves_nulls(archive):
    rows = csv_rows(FOLDER / "participant_metrics.csv")
    assert len(rows) == 54
    expected = {
        (model, person["participant"]): person
        for model in MODELS
        for person in archive["evaluation"]["models"][model]["test"]["participants"]
    }
    assert len({(row["model"], row["participant"]) for row in rows}) == len(rows)
    for row in rows:
        person = expected[(row["model"], row["participant"])]
        for key, value in person.items():
            if value is None:
                assert row[key] == ""
            elif isinstance(value, str):
                assert row[key] == value
            else:
                assert float(row[key]) == pytest.approx(value, abs=1e-12)
