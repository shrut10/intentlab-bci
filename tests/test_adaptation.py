import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from intentlab.adaptation import (
    apply_alignment,
    binary_metrics,
    fit_alignment,
    fit_temperature,
    paired_interval,
    partition_rows,
    probabilities,
    select_threshold,
)

CONFIG = json.loads(
    (Path(__file__).resolve().parents[1] / "experiments/adaptation-v1/protocol.json").read_text()
)


def rows():
    return [
        {
            "id": f"{person}-{run}-{event}",
            "subject": person,
            "run": run,
            "event": event,
            "cue_seconds": event * 8,
            "label": event % 2,
            "split": split,
        }
        for person, split in [(1, "train"), (2, "test")]
        for run in (4, 8, 12)
        for event in range(12)
    ]


def test_calibration_selection_uses_chronology_not_labels_or_input_order():
    data = rows()[::-1]
    early, later = partition_rows(data)
    for person in early:
        assert [data[i]["event"] for i in early[person]] == list(range(10))
        assert {data[i]["run"] for i in early[person]} == {4}
        assert {data[i]["run"] for i in later[person]} == {8, 12}
        assert not set(early[person]) & set(later[person])
    changed = [dict(row, label=1 - row["label"]) for row in data]
    changed_early, _ = partition_rows(changed)
    for person in early:
        np.testing.assert_array_equal(early[person], changed_early[person])
    with pytest.raises(ValueError, match="budget"):
        partition_rows(data, budget=13)
    with pytest.raises(ValueError, match="crosses"):
        partition_rows([dict(row, split="validation") if i == 0 else row for i, row in enumerate(data)])


def test_unregularised_alignment_whitens_calibration_second_moment():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(10, 3, 480)) * np.array([1, 4, 10])[None, :, None]
    x[:, 1] += 0.5 * x[:, 2]
    aligned = apply_alignment(x, fit_alignment(x, shrinkage=0))
    mean = np.einsum("nct,ndt->cd", aligned, aligned) / (10 * 480)
    np.testing.assert_allclose(mean, np.eye(3), atol=1e-5)


def test_alignment_is_stable_for_correlated_channels_and_cannot_see_future_trials():
    rng = np.random.default_rng(2)
    early = np.repeat(rng.normal(size=(10, 1, 480)), 3, axis=1)
    matrix = fit_alignment(early)
    future = rng.normal(size=(5, 3, 480))
    whole = np.concatenate([early, future])
    before = apply_alignment(whole, matrix)[:10]
    whole[10:] *= 1000
    after = apply_alignment(whole, fit_alignment(whole[:10]))[:10]
    np.testing.assert_allclose(before, after)
    assert np.isfinite(matrix).all()
    with pytest.raises(ValueError, match="zero-energy"):
        fit_alignment(np.zeros_like(early))
    with pytest.raises(ValueError, match="finite"):
        fit_alignment(np.full_like(early, np.nan))


def test_personal_temperature_changes_confidence_without_changing_decisions_or_ranking():
    logits = np.array([-5.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    labels = np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 1])
    fitted = fit_temperature(logits, labels)
    before, after = expit(logits), probabilities(logits, fitted["temperature"])
    np.testing.assert_array_equal(before >= 0.5, after >= 0.5)
    # Opposite-sign logits of equal magnitude have tied confidence. Tiny
    # platform-dependent sigmoid rounding must not impose an ordering on ties.
    magnitude_difference = abs(logits)[:, None] - abs(logits)[None, :]
    for probability in (before, after):
        confidence = np.maximum(probability, 1 - probability)
        difference = confidence[:, None] - confidence[None, :]
        assert (difference[magnitude_difference > 0] > 0).all()
        np.testing.assert_allclose(difference[magnitude_difference == 0], 0, atol=1e-15)
    assert fitted["temperature"] > 1
    fallback = fit_temperature(logits, np.ones(10), fallback=2.5)
    assert fallback == {"temperature": 2.5, "fallback": "single_class", "at_bound": False}


def test_threshold_rule_enforces_accuracy_coverage_and_sample_count():
    y = np.tile([0, 1], 100)
    high_quality = np.where(y, 0.8, 0.2)
    selected = select_threshold(y, high_quality, CONFIG)
    assert selected["target_met"] and selected["threshold"] == 0.5
    bad = select_threshold(y, np.full(200, 0.6), CONFIG)
    assert not bad["target_met"] and bad["threshold"] == 0.75 and bad["accuracy"] is None
    too_small = select_threshold(y[:50], high_quality[:50], CONFIG)
    assert not too_small["target_met"]


def test_participant_macro_and_bootstrap_do_not_count_seeds_as_extra_people():
    y = np.array([0, 1, 0, 0, 1, 1])
    p = np.array([0.1, 0.9, 0.9, 0.9, 0.1, 0.1])
    result = binary_metrics(y, p, [1, 1, 2, 2, 2, 2])
    assert result["participant_balanced_accuracy"] == 0.5
    assert result["accuracy"] == pytest.approx(1 / 3)
    baseline = np.array([[0.5, 0.6], [0.4, 0.7], [0.5, 0.8]])
    change = paired_interval(baseline, baseline + 0.1)
    assert change["participants"] == 2 and change["seeds"] == 3
    np.testing.assert_allclose(change["bootstrap_95_ci"], [0.1, 0.1])
