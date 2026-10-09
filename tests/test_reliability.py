import copy
import json

import numpy as np
import pytest

from examples.reliability_audit import bundled_audit, serialise
from intentlab.reliability import build_audit

RULE = {
    "thresholds": [0.5, 0.75, 1.0],
    "target_accuracy": 0.75,
    "target_coverage": 0.2,
    "target_retained": 2,
    "fallback_threshold": 0.75,
}


def example():
    return [
        {
            "condition": "a",
            "seed": 1,
            "split": split,
            "subject": f"{split}-{person}",
            "trial_id": f"{split}-{person}-{label}",
            "label": label,
            "probability": p if label else 1 - p,
        }
        for split in ("validation", "test")
        for person, p in ((1, 0.9), (2, 0.6), (3, 0.1))
        for label in (0, 1)
    ]


def test_hand_calculated_counts_keep_people_without_commands_and_missing_accuracy():
    result = build_audit(example(), RULE, repeats=100)
    point = result["groups"][1]["points"][1]
    assert (point["accepted"], point["correct"], point["participants_without_commands"]) == (4, 2, 1)
    assert point["coverage"] == 4 / 6
    assert point["selective_risk"] == 0.5
    assert point["participants"][1]["accepted_accuracy"] is None
    empty = result["groups"][1]["points"][2]
    assert empty["accepted_accuracy"] is None
    assert empty["selective_risk"] is None
    assert empty["accepted_accuracy_ci95"] is None
    assert empty["accuracy_bootstrap_valid"] == 0
    assert json.loads(serialise(result))["groups"][1]["points"][2]["accepted_accuracy"] is None


def test_test_outcomes_do_not_select_threshold():
    rows = example()
    before = build_audit(rows, RULE, repeats=100)
    for row in rows:
        if row["split"] == "test":
            row["label"] = 1 - row["label"]
    after = build_audit(rows, RULE, repeats=100)
    assert [(g["selected_threshold"], g["validation_target_met"]) for g in before["groups"]] == [
        (g["selected_threshold"], g["validation_target_met"]) for g in after["groups"]
    ]
    assert before["groups"][0] == after["groups"][0]
    assert before["groups"][1]["points"][0]["correct"] != after["groups"][1]["points"][0]["correct"]


def test_repeated_seeds_do_not_narrow_participant_intervals_and_row_order_is_irrelevant():
    rows = example()
    first = build_audit(rows, RULE, repeats=100)
    repeated = [dict(r, seed=2) for r in rows]
    second = build_audit(list(reversed(rows + repeated)), RULE, repeats=100)
    assert first["groups"] == second["groups"][:2]
    assert second["groups"][1]["points"] == second["groups"][3]["points"]
    assert second["groups"][3]["participant_count"] == 3


def test_bootstrap_resamples_whole_participants_not_individual_trials():
    point = build_audit(example(), RULE, repeats=100)["groups"][1]["points"][1]
    draws = np.random.default_rng(4027).integers(0, 3, size=(100, 3))
    retained = np.array([2, 0, 2])[draws].sum(axis=1)
    correct = np.array([2, 0, 0])[draws].sum(axis=1)
    np.testing.assert_allclose(point["coverage_ci95"], np.quantile(retained / 6, [0.025, 0.975]))
    np.testing.assert_allclose(
        point["accepted_accuracy_ci95"],
        np.quantile(correct[retained > 0] / retained[retained > 0], [0.025, 0.975]),
    )
    assert point["accuracy_bootstrap_valid"] == int((retained > 0).sum())


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda r: r.append(r[0]), "unique"),
        (lambda r: r[0].update(probability=float("nan")), "finite probability"),
        (lambda r: r[0].update(label=2), "binary label"),
        (lambda r: r[-1].update(subject=r[0]["subject"]), "disjoint"),
        (lambda r: r.append(dict(r[0], seed=2, label=1)), "metadata"),
        (lambda r: r.extend([dict(x, seed=2) for x in r[:-1]]), "same trials"),
    ],
)
def test_invalid_comparisons_fail(mutation, message):
    rows = copy.deepcopy(example())
    mutation(rows)
    with pytest.raises(ValueError, match=message):
        build_audit(rows, RULE, repeats=100)


def test_single_class_and_single_contributor_are_not_reported_as_valid_estimates():
    rows = [r for r in example() if r["label"] == 1 and r["subject"].endswith("1")]
    point = build_audit(rows, RULE, repeats=100)["groups"][1]["points"][1]
    assert point["participants"][0]["balanced_accuracy"] is None
    assert point["accepted_accuracy_ci95"] is None
    assert point["coverage_ci95"] is None


def test_archived_counts_and_public_route(client):
    report = bundled_audit(repeats=100)
    g = next(
        g
        for g in report["groups"]
        if g["condition"] == "baseline" and g["seed"] == "2027" and g["split"] == "test"
    )
    p = next(p for p in g["points"] if p["threshold"] == 0.75)
    assert (p["accepted"], p["correct"], p["participants_without_commands"]) == (9, 9, 18)
    assert g["validation_target_met"] is False
    assert client.get("/research/reliability").status_code == 200
    assert client.get("/static/reliability.json").json()["source"] == report["source"]
