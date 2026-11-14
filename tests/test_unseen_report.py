"""Keep the published participant report faithful to its archived evidence."""

import copy
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from intentlab.unseen import bootstrap_comparison, evaluate_predictions, select_threshold
from scripts.render_unseen_report import (
    numerical_export,
    participant_plot,
    render,
    table,
    validate_report,
    validation_threshold_note,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/unseen-v1"


@pytest.fixture(scope="module")
def example_report():
    """Synthetic predictions on the declared identifiers, never published results."""
    manifest = json.loads((ROOT / "experiments/unseen-v1/split_manifest.json").read_text())
    splits = {
        split: [row["participant"] for row in manifest["participants"] if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    cohorts = {row["participant"]: row["cohort"] for row in manifest["participants"]}
    inputs = {}
    for split in ("validation", "test"):
        people = np.repeat(splits[split], 4)
        labels = np.tile([0, 1, 0, 1], len(splits[split]))
        group = [cohorts[person] for person in people]
        bandpower = np.tile([[0.8, 0.2], [0.8, 0.2], [0.2, 0.8], [0.2, 0.8]], (len(splits[split]), 1))
        tangent = np.column_stack([1 - labels * 0.6 - 0.2, labels * 0.6 + 0.2])
        # Include a person with no accepted predictions under the fallback
        bandpower[:4] = 0.5
        tangent[:4] = 0.5
        inputs[split] = (labels, people, group, {"bandpower": bandpower, "covariance_tangent": tangent})
    models = {}
    for model in ("bandpower", "covariance_tangent"):
        labels, people, group, probabilities = inputs["validation"]
        threshold = select_threshold(labels, probabilities[model], people, group)
        values = {"threshold_selection": threshold}
        for split, (labels, people, group, probabilities) in inputs.items():
            values[split] = evaluate_predictions(
                labels, probabilities[model], people, group, threshold["threshold"]
            )
        models[model] = values
    labels, people, group, probabilities = inputs["test"]
    bootstrap = bootstrap_comparison(labels, probabilities, people, group, repeats=5000, seed=2027)
    return {
        "schema_version": "1.0",
        "kind": "unseen_participants",
        "experiment": "unseen-v1",
        "dataset": "Dreyer2023",
        "production_model_changed": False,
        "completed_utc": "2026-11-14T18:00:00+00:00",
        "protocol_commit": "a" * 40,
        "selection_commit": "b" * 40,
        "counts": {
            split: {"participants": len(ids), "trials": 4 * len(ids)} for split, ids in splits.items()
        },
        "splits": splits,
        "models": models,
        "selection": {
            "model": "covariance_tangent",
            "rule": "Highest validation participant macro balanced accuracy",
            "primary_criterion_met": bootstrap["model_intervals"]["covariance_tangent"][0] > 0.5,
        },
        "test_bootstrap": bootstrap,
    }


def test_synthetic_metrics_render_without_changing_the_evidence(example_report):
    original = copy.deepcopy(example_report)
    html = render(example_report)
    assert example_report == original
    assert "Covariance tangent space was selected using validation participants" in html
    assert "The original deployed decoder remains unchanged" in html
    assert "sham visual feedback" in html
    assert "not a transfer test of the deployed PhysioNet model" in html
    assert "$primary_finding" not in html
    assert json.loads(numerical_export(example_report)) == example_report


def test_every_test_person_is_visible_including_zero_coverage(example_report):
    html = render(example_report)
    content = re.search(r'id="unseen-test-participants".*?<tbody>(.*?)</tbody>', html, re.DOTALL).group(1)
    assert content.count("<tr>") == 27
    for person in example_report["splits"]["test"]:
        assert f'<th scope="row">{person}</th>' in content
    assert "<td>0 / 4</td><td>0.00%</td><td>—</td>" in content
    assert "Cohort C has only two test participants" in html


def test_participant_plot_uses_the_same_scores_and_fixed_reference_as_the_table(example_report):
    metrics = example_report["models"]["covariance_tangent"]["test"]
    markup = participant_plot(metrics)
    plot = ET.fromstring(re.search(r"(<svg.*?</svg>)", markup).group(1))
    points = plot.findall("circle")
    assert len(points) == len(metrics["participants"]) == 27
    scores = {row["participant"]: row["balanced_accuracy"] for row in metrics["participants"]}
    for point in points:
        score = scores[point.attrib["data-participant"]]
        assert float(point.attrib["data-score"]) == score
        assert float(point.attrib["cx"].rstrip("%")) == pytest.approx(26 + 52 * score, abs=1e-6)
    reference = [line for line in plot.findall("line") if "stroke-dasharray" in line.attrib]
    assert len(reference) == 1 and reference[0].attrib["x1"] == "52%"
    assert plot.attrib["width"] == "100%"
    assert "viewBox" not in plot.attrib  # Text retains its physical size on narrow displays
    assert "style=" not in markup and "<script" not in markup


@pytest.mark.parametrize("mutation", ["membership", "overlap", "count", "cohort"])
def test_report_rejects_changes_to_the_declared_participant_split(example_report, mutation):
    report = copy.deepcopy(example_report)
    if mutation == "membership":
        report["splits"]["test"][0] = "S99"
    elif mutation == "overlap":
        report["splits"]["test"][0] = report["splits"]["train"][0]
    elif mutation == "count":
        report["counts"]["test"]["participants"] = 26
    else:
        report["models"]["bandpower"]["test"]["participants"][0]["cohort"] = "C"
    with pytest.raises(ValueError):
        validate_report(report)


@pytest.mark.parametrize("mutation", ["mean", "accepted", "undefined", "brier", "matrix", "cohort_mean"])
def test_report_rejects_summary_values_that_disagree_with_people(example_report, mutation):
    report = copy.deepcopy(example_report)
    metrics = report["models"]["covariance_tangent"]["test"]
    if mutation == "mean":
        metrics["participant_macro_balanced_accuracy"] = 0.5
    elif mutation == "accepted":
        metrics["selective"]["accepted"] += 1
    elif mutation == "undefined":
        metrics["participants"][0]["accepted_accuracy"] = 0
    elif mutation == "brier":
        metrics["participants"][0]["brier"] = float("nan")
    elif mutation == "matrix":
        metrics["pooled"]["confusion_matrix"][0][0] += 1
    else:
        metrics["cohorts"][0]["participant_macro_balanced_accuracy"] = 0
    with pytest.raises(ValueError):
        validate_report(report)


def test_test_results_cannot_change_validation_model_choice(example_report):
    report = copy.deepcopy(example_report)
    report["selection"]["model"] = "bandpower"
    with pytest.raises(ValueError, match="selection"):
        validate_report(report)


def test_threshold_and_failure_status_are_locked_to_validation(example_report):
    report = copy.deepcopy(example_report)
    report["models"]["bandpower"]["test"]["selective"]["threshold"] = 0.8
    with pytest.raises(ValueError, match="stay fixed"):
        validate_report(report)
    report = copy.deepcopy(example_report)
    report["models"]["bandpower"]["threshold_selection"]["validation_target_met"] = True
    with pytest.raises(ValueError, match="recorded validation candidates"):
        validate_report(report)


def test_threshold_explanation_retains_participant_shortfall_despite_high_pooled_accuracy(example_report):
    selection = example_report["models"]["bandpower"]["threshold_selection"]
    # A small constructed summary isolates the participant minimum requirement
    metrics = {
        "participant_count": 2,
        "participants": [{"accepted": 1}, {"accepted": 199}],
        "selective": {
            "participant_macro_accuracy": 0.8,
            "participant_macro_coverage": 0.25,
            "accepted": 200,
            "minimum_accepted_per_participant": 1,
            "pooled_accuracy": 0.9,
        },
    }
    note = validation_threshold_note(selection, metrics)
    assert "90.00%" in note
    assert "1 of 2 validation participants has fewer than 5 accepted trials" in note
    assert "lowest count of 1" in note
    assert "accuracy is" not in note


def test_primary_decision_uses_the_selected_model_lower_bound(example_report):
    report = copy.deepcopy(example_report)
    report["selection"]["primary_criterion_met"] = not report["selection"]["primary_criterion_met"]
    with pytest.raises(ValueError, match="lower test bound"):
        validate_report(report)
    report = copy.deepcopy(example_report)
    report["test_bootstrap"]["model_intervals"]["covariance_tangent"] = [0.5, 1]
    report["selection"]["primary_criterion_met"] = False
    assert "was not met because the lower bound does not exceed" in render(report)


def test_bootstrap_must_retain_the_actual_resampling_unit(example_report):
    report = copy.deepcopy(example_report)
    report["test_bootstrap"]["unit"] = "trial"
    with pytest.raises(ValueError, match="complete test participants"):
        validate_report(report)
    report = copy.deepcopy(example_report)
    report["test_bootstrap"]["strata_counts"]["C"] = 3
    with pytest.raises(ValueError, match="cohort counts"):
        validate_report(report)


@pytest.mark.parametrize("field,value", [("repeats", 4999), ("seed", 2028)])
def test_bootstrap_configuration_is_frozen_in_the_report(example_report, field, value):
    report = copy.deepcopy(example_report)
    report["test_bootstrap"][field] = value
    with pytest.raises(ValueError, match="5000 repetitions and seed 2027"):
        validate_report(report)


def test_original_model_change_cannot_be_published_as_this_experiment(example_report):
    report = copy.deepcopy(example_report)
    report["production_model_changed"] = True
    with pytest.raises(ValueError, match="deployed model"):
        validate_report(report)


def test_generated_tables_escape_supplied_text():
    html = table(
        'A <title> "quoted"',
        ["Participant", "Value"],
        [["<script>alert(1)</script>", "<b>value</b>"]],
        "safe-table",
    )
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;b&gt;value&lt;/b&gt;" in html
    assert 'aria-label="A &lt;title&gt; &quot;quoted&quot;"' in html


def test_archived_report_matches_both_published_exports():
    report = json.loads((ARTIFACTS / "evaluation.json").read_text())
    assert (ROOT / "web/unseen.html").read_text() == render(report)
    exported = (ROOT / "web/unseen-study.json").read_text()
    assert exported == numerical_export(report)
    assert len(exported.encode()) < 1_000_000
    assert len(report["models"][report["selection"]["model"]]["test"]["participants"]) == 27
