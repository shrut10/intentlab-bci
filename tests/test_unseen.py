"""Scientific checks for fixed features, source fitting and unseen people metrics."""

import copy
import inspect
import json
import warnings

import numpy as np
import pytest

from intentlab.unseen import (
    bootstrap_comparison,
    choose_model,
    evaluate_predictions,
    extract_features,
    fit_models,
    participant_class_weights,
    predict_models,
    preprocess_epochs,
    select_threshold,
    stratified_participant_draws,
)


def features(seed=2027, size=24):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(size, 9, 9))
    covariance = base @ base.swapaxes(-1, -2) + np.eye(9)[None] * 4
    covariance /= np.trace(covariance, axis1=1, axis2=2)[:, None, None]
    return {"bandpower": rng.normal(size=(size, 27)), "covariance": covariance}


@pytest.fixture(scope="module")
def fitted():
    training = features()
    labels = np.tile([0, 1], 12)
    participants = ["A1"] * 12 + ["B2"] * 12
    return fit_models(training, labels, participants)


def test_participant_and_class_weights_have_equal_total_mass():
    labels = [0, 0, 0, 1, 0, 1, 1, 1, 1, 1]
    people = ["A1"] * 4 + ["B2"] * 6
    weights = participant_class_weights(labels, people)
    assert weights.sum() == pytest.approx(10)
    for person in set(people):
        for label in (0, 1):
            mask = (np.asarray(people) == person) & (np.asarray(labels) == label)
            assert weights[mask].sum() == pytest.approx(2.5)


def test_weights_reject_participant_with_only_one_class():
    with pytest.raises(ValueError, match="both classes"):
        participant_class_weights([0, 0, 1, 1], ["A1", "A1", "B2", "B2"])


def test_completed_epoch_preprocessing_is_independent_and_does_not_mutate():
    rng = np.random.default_rng(12)
    epochs = rng.normal(size=(3, 9, 2048))
    original = epochs.copy()
    batch = preprocess_epochs(epochs)
    np.testing.assert_array_equal(epochs, original)
    np.testing.assert_array_equal(batch[:1], preprocess_epochs(epochs[:1]))
    np.testing.assert_allclose(np.sqrt(np.mean(batch**2, axis=(1, 2))), 1)
    epoch_features = extract_features(epochs)
    assert epoch_features["bandpower"].shape == (3, 27)
    assert epoch_features["covariance"].shape == (3, 9, 9)
    assert np.all(np.linalg.eigvalsh(epoch_features["covariance"]) > 0)
    np.testing.assert_allclose(np.trace(epoch_features["covariance"], axis1=1, axis2=2), 1)


@pytest.mark.parametrize("shape", [(9, 2048), (1, 8, 2048), (1, 9, 2047), (0, 9, 2048)])
def test_epoch_shape_is_fixed(shape):
    with pytest.raises(ValueError, match="Expected"):
        preprocess_epochs(np.ones(shape))


def test_flat_nonfinite_and_wrong_sampling_epochs_are_rejected():
    with pytest.raises(ValueError, match="RMS"):
        preprocess_epochs(np.zeros((1, 9, 2048)))
    with pytest.raises(ValueError, match="finite"):
        preprocess_epochs(np.full((1, 9, 2048), np.nan))
    with pytest.raises(ValueError, match="512"):
        preprocess_epochs(np.ones((1, 9, 2048)), sampling_hz=250)


def test_oas_regularises_nearly_identical_channel_signals():
    time = np.arange(2048) / 512
    shared = np.sin(2 * np.pi * 10 * time)
    epochs = np.broadcast_to(shared, (2, 9, 2048)).copy()
    epochs += np.random.default_rng(9).normal(scale=1e-10, size=epochs.shape)
    covariance = extract_features(epochs)["covariance"]
    assert np.isfinite(covariance).all()
    assert np.all(np.linalg.eigvalsh(covariance) > 0)
    np.testing.assert_allclose(np.trace(covariance, axis1=1, axis2=2), 1)


def test_fit_signature_contains_only_training_arrays_and_scaler_uses_training_weights():
    assert list(inspect.signature(fit_models).parameters) == [
        "train_features",
        "train_labels",
        "train_participants",
    ]
    training = features(size=10)
    labels = [0, 0, 0, 1, 0, 1, 1, 1, 1, 1]
    people = ["A1"] * 4 + ["B2"] * 6
    weights = participant_class_weights(labels, people)
    models = fit_models(training, labels, people)
    expected = np.average(training["bandpower"], axis=0, weights=weights)
    np.testing.assert_allclose(models["bandpower"].named_steps["scale"].mean_, expected)
    from pyriemann.tangentspace import TangentSpace

    reference = TangentSpace(metric="riemann", tsupdate=False).fit(
        training["covariance"], sample_weight=weights
    )
    pipeline = models["covariance_tangent"]
    np.testing.assert_allclose(pipeline.named_steps["tangent"].reference_, reference.reference_)
    expected_tangent_mean = np.average(reference.transform(training["covariance"]), axis=0, weights=weights)
    np.testing.assert_allclose(pipeline.named_steps["scale"].mean_, expected_tangent_mean, atol=1e-12)


def test_training_weights_reach_each_classifier(monkeypatch):
    from sklearn.linear_model import LogisticRegression

    original, calls = LogisticRegression.fit, []

    def capture(self, x, y, sample_weight=None):
        calls.append(np.asarray(sample_weight).copy())
        assert self.class_weight is None
        return original(self, x, y, sample_weight=sample_weight)

    monkeypatch.setattr(LogisticRegression, "fit", capture)
    labels = [0, 0, 0, 1, 0, 1, 1, 1, 1, 1]
    people = ["A1"] * 4 + ["B2"] * 6
    fit_models(features(size=10), labels, people)
    assert len(calls) == 2
    for weights in calls:
        np.testing.assert_array_equal(weights, participant_class_weights(labels, people))


def test_missing_weight_support_stops_before_any_classifier_fit(monkeypatch):
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression

    def unsupported(self, x, y=None):
        raise AssertionError("The unsupported reference must not be fitted")

    def forbidden(*args, **kwargs):
        raise AssertionError("A classifier must not fit after an unsupported reference was detected")

    monkeypatch.setattr(TangentSpace, "fit", unsupported)
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    with pytest.raises(RuntimeError, match="weights"):
        fit_models(features(), np.tile([0, 1], 12), ["A1"] * 24)


def test_unconverged_tangent_reference_stops_fitting(monkeypatch):
    from pyriemann.tangentspace import TangentSpace

    def unconverged(self, x, y=None, sample_weight=None):
        warnings.warn("Convergence not reached", stacklevel=1)

    monkeypatch.setattr(TangentSpace, "fit_transform", unconverged)
    with pytest.raises(UserWarning, match="Convergence not reached"):
        fit_models(features(), np.tile([0, 1], 12), ["A1"] * 24)


def test_target_companions_order_and_prediction_cannot_update_training_state(fitted):
    target = features(seed=7, size=5)
    original = copy.deepcopy(fitted)
    batch = predict_models(fitted, target)
    alone = predict_models(fitted, {key: value[:1] for key, value in target.items()})
    reverse = predict_models(fitted, {key: value[::-1] for key, value in target.items()})
    replacement = features(seed=999, size=5)
    for key in replacement:
        replacement[key][0] = target[key][0]
    changed = predict_models(fitted, replacement)
    for name in fitted:
        np.testing.assert_allclose(alone[name][0], batch[name][0], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(reverse[name][::-1], batch[name], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(changed[name][0], batch[name][0], rtol=1e-10, atol=1e-12)
        for step, attributes in (
            ("scale", ["mean_", "scale_", "var_"]),
            ("classifier", ["coef_", "intercept_"]),
        ):
            for attribute in attributes:
                np.testing.assert_array_equal(
                    getattr(fitted[name].named_steps[step], attribute),
                    getattr(original[name].named_steps[step], attribute),
                )
    np.testing.assert_array_equal(
        fitted["covariance_tangent"].named_steps["tangent"].reference_,
        original["covariance_tangent"].named_steps["tangent"].reference_,
    )


def test_target_reference_updates_are_refused(fitted):
    altered = copy.deepcopy(fitted)
    altered["covariance_tangent"].named_steps["tangent"].tsupdate = True
    with pytest.raises(ValueError, match="forbidden"):
        predict_models(altered, features(size=2))


@pytest.mark.parametrize("corruption", ["nonfinite", "asymmetric", "singular", "trace", "extra"])
def test_malformed_features_are_rejected(fitted, corruption):
    values = features(size=2)
    if corruption == "nonfinite":
        values["bandpower"][0, 0] = np.nan
    elif corruption == "asymmetric":
        values["covariance"][0, 0, 1] += 0.1
    elif corruption == "singular":
        values["covariance"][0] = np.zeros((9, 9))
    elif corruption == "trace":
        values["covariance"] *= 2
    else:
        values["test"] = values["bandpower"]
    with pytest.raises(ValueError):
        predict_models(fitted, values)


def test_hand_computed_participant_metrics_keep_zero_coverage_people():
    labels = [0, 1, 0, 0, 0, 1, 1, 1]
    probabilities = [[0.9, 0.1], [0.1, 0.9]] + [[0.8, 0.2]] * 6
    people, cohorts = ["A1"] * 2 + ["B2"] * 6, ["A"] * 2 + ["B"] * 6
    result = evaluate_predictions(labels, probabilities, people, cohorts, threshold=0.85)
    assert result["participant_macro_balanced_accuracy"] == 0.75
    assert result["participant_macro_accuracy"] == 0.75
    assert result["participant_macro_brier"] == pytest.approx(0.175)
    assert result["pooled"]["accuracy"] == 0.625
    assert result["pooled"]["confusion_matrix"] == [[4, 0], [3, 1]]
    assert result["participants"][1]["accepted_accuracy"] is None
    assert result["participants"][1]["coverage"] == 0
    assert result["selective"] == {
        "threshold": 0.85,
        "accepted": 2,
        "correct_accepted": 2,
        "pooled_accuracy": 1.0,
        "participant_macro_accuracy": 1.0,
        "participant_macro_coverage": 0.5,
        "pooled_coverage": 0.25,
        "participants_with_acceptance": 1,
        "participants_without_acceptance": 1,
        "minimum_accepted_per_participant": 0,
    }
    assert sum(row["count"] for row in result["pooled"]["calibration"]) == 8
    json.dumps(result, allow_nan=False)


def test_no_accepted_trials_have_null_accuracy_and_extreme_probabilities_fit_bins():
    result = evaluate_predictions([0, 1], [[0.6, 0.4], [0.4, 0.6]], ["A1"] * 2, ["A"] * 2, 0.9)
    assert result["selective"]["pooled_accuracy"] is None
    assert result["selective"]["participant_macro_accuracy"] is None
    assert result["selective"]["participants_without_acceptance"] == 1
    edge = evaluate_predictions([0, 1], [[1, 0], [0, 1]], ["A1"] * 2, ["A"] * 2)
    assert edge["pooled"]["calibration"][0]["count"] == 1
    assert edge["pooled"]["calibration"][-1]["count"] == 1


def test_model_selection_uses_validation_and_exact_tie_favours_bandpower():
    summaries = {
        name: {"participant_macro_balanced_accuracy": 0.6} for name in ("bandpower", "covariance_tangent")
    }
    assert choose_model(summaries) == "bandpower"
    summaries["covariance_tangent"]["participant_macro_balanced_accuracy"] = 0.61
    assert choose_model(summaries) == "covariance_tangent"


def test_threshold_selects_lowest_qualifying_value_and_small_sample_fails():
    labels = np.tile([0, 1], 60)
    probabilities = np.column_stack([np.where(labels == 0, 0.8, 0.2), np.where(labels == 1, 0.8, 0.2)])
    people, cohorts = ["A1"] * 40 + ["A2"] * 40 + ["B3"] * 40, ["A"] * 80 + ["B"] * 40
    result = select_threshold(labels, probabilities, people, cohorts)
    assert result["validation_target_met"] and result["threshold"] == 0.5
    assert len(result["candidates"]) == 10
    small = select_threshold(labels[:80], probabilities[:80], people[:80], cohorts[:80])
    assert not small["validation_target_met"] and small["threshold"] == 0.75


def test_threshold_cannot_discard_difficult_people_to_pass_target():
    labels = np.tile([0, 1], 120)
    probabilities = np.array([[0.9, 0.1] if label == 0 else [0.1, 0.9] for label in labels])
    probabilities[120:] = [[0.4, 0.6] if label == 0 else [0.6, 0.4] for label in labels[120:]]
    result = select_threshold(labels, probabilities, ["A1"] * 120 + ["B2"] * 120, ["A"] * 120 + ["B"] * 120)
    assert not result["validation_target_met"]
    candidate = next(row for row in result["candidates"] if row["threshold"] == 0.75)
    assert candidate["accepted"] == 120
    assert candidate["participant_macro_accuracy"] == 1
    assert candidate["minimum_accepted_per_participant"] == 0


def test_stratified_bootstrap_counts_and_reproducibility():
    cohorts = np.array(["A"] * 19 + ["B"] * 6 + ["C"] * 2)
    draws = stratified_participant_draws(cohorts, repeats=100)
    np.testing.assert_array_equal(draws, stratified_participant_draws(cohorts, repeats=100))
    assert draws.shape == (100, 27)
    for cohort, count in (("A", 19), ("B", 6), ("C", 2)):
        np.testing.assert_array_equal((cohorts[draws] == cohort).sum(axis=1), np.full(100, count))


def test_paired_bootstrap_identical_predictions_have_exact_zero_difference():
    labels = np.tile([0, 1], 8)
    right = np.array([0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.2, 0.8] * 2)
    probabilities = np.column_stack([1 - right, right])
    people = np.repeat(["A1", "A2", "B3", "C4"], 4)
    cohorts = np.repeat(["A", "A", "B", "C"], 4)
    result = bootstrap_comparison(
        labels,
        {"bandpower": probabilities, "covariance_tangent": probabilities},
        people,
        cohorts,
        repeats=100,
    )
    assert result["paired_difference"]["estimate"] == 0
    assert result["paired_difference"]["ci95"] == [0, 0]
    assert result["model_intervals"]["bandpower"] == result["model_intervals"]["covariance_tangent"]
    assert result["strata_counts"] == {"A": 2, "B": 1, "C": 1}
    assert result["participant_ids"] == ["A1", "A2", "B3", "C4"]
    json.dumps(result, allow_nan=False)


def test_paired_bootstrap_does_not_weight_participants_by_trial_count():
    labels = np.array([0, 1] + [0, 1] * 10)
    left_only = np.tile([0.8, 0.2], (len(labels), 1))
    tangent = left_only.copy()
    tangent[1] = [0.2, 0.8]
    result = bootstrap_comparison(
        labels,
        {"bandpower": left_only, "covariance_tangent": tangent},
        ["A1"] * 2 + ["B2"] * 20,
        ["A"] * 2 + ["B"] * 20,
        repeats=100,
    )
    assert result["paired_difference"]["estimate"] == 0.25
    assert result["paired_difference"]["ci95"] == [0.25, 0.25]


@pytest.mark.parametrize(
    "labels,probabilities,people,cohorts",
    [
        ([0, 1], [[0.8, 0.3], [0.2, 0.8]], ["A1", "A1"], ["A", "A"]),
        ([0, 2], [[0.8, 0.2], [0.2, 0.8]], ["A1", "A1"], ["A", "A"]),
        ([0, 1], [[0.8, 0.2], [0.2, 0.8]], ["A1", "A1"], ["A", "B"]),
        ([0, 1], [[0.8, 0.2], [0.2, 0.8]], ["A1", "B2"], ["A", "B"]),
        ([0, 1], [[0.8, 0.2], [0.2, 0.8]], ["A1", "A1"], ["Z", "Z"]),
    ],
)
def test_invalid_evaluation_records_fail(labels, probabilities, people, cohorts):
    with pytest.raises(ValueError):
        evaluate_predictions(labels, probabilities, people, cohorts)
