"""Run explicitly in the isolated MOABB environment; no EEG downloads needed."""

from functools import wraps

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.preprocessing import LabelEncoder

from examples.moabb_chronological.example import (
    AlignedLogVariance,
    EarlyRunCalibration,
    demo_dataset,
    run_demo,
)


def fixture_data():
    metadata = pd.DataFrame(
        [
            (subject, "0", run)
            for subject in (1, 2, 3)
            for run, n in [("0", 14), ("1", 12), ("2", 12)]
            for _ in range(n)
        ],
        columns=["subject", "session", "run"],
    )
    # Deliberately duplicate, non-positional labels: splitter must use iloc positions.
    metadata.index = [999] * len(metadata)
    X = np.random.default_rng(42).normal(size=(len(metadata), 3, 64))
    y = np.tile([0, 1], len(metadata) // 2)
    return X, y, metadata


def test_exact_budget_disjointness_chronology_and_discarded_rows():
    _, y, metadata = fixture_data()
    splitter = EarlyRunCalibration()
    assert splitter.get_n_splits(metadata=metadata) == 3
    for target, (train, calibration, test) in enumerate(splitter.split(y, metadata), 1):
        start = (target - 1) * 38
        assert_array_equal(calibration, np.arange(start, start + 10))
        assert_array_equal(test, np.arange(start + 14, start + 38))
        assert len(train) == 76
        assert (metadata.iloc[train].subject != target).all()
        combined = np.concatenate((train, calibration, test))
        assert len(np.unique(combined)) == len(combined)
        assert_array_equal(np.setdiff1d(np.arange(len(y)), combined), np.arange(start + 10, start + 14))
        assert splitter.get_metadata()["discarded_trials"] == 4


def test_selection_is_independent_of_labels_and_run_names_are_not_sorted():
    _, y, metadata = fixture_data()
    metadata["run"] = metadata.run.map({"0": "04", "1": "12", "2": "08"})
    first = list(EarlyRunCalibration(run_order=("04", "12", "08")).split(y, metadata))
    second = list(EarlyRunCalibration(run_order=("04", "12", "08")).split(1 - y, metadata))
    for fold_a, fold_b in zip(first, second, strict=True):
        for a, b in zip(fold_a, fold_b, strict=True):
            assert_array_equal(a, b)
    assert_array_equal(first[0][2], np.arange(14, 38))


@pytest.mark.parametrize("budget", [0, -1, 2.5, True, 15])
def test_invalid_or_unavailable_budget_fails(budget):
    _, y, metadata = fixture_data()
    with pytest.raises(ValueError, match="budget|insufficient"):
        list(EarlyRunCalibration(budget=budget).split(y, metadata))


@pytest.mark.parametrize(
    "problem", ["missing_run", "unexpected_run", "multiple_sessions", "missing_value", "one_subject"]
)
def test_ambiguous_or_incomplete_metadata_fails(problem):
    _, y, metadata = fixture_data()
    if problem == "missing_run":
        metadata.loc[metadata.run == "2", "run"] = "1"
    elif problem == "unexpected_run":
        metadata.iloc[0, 2] = "unexpected"
    elif problem == "multiple_sessions":
        metadata.iloc[0, 1] = "other"
    elif problem == "missing_value":
        metadata.iloc[0, 2] = None
    else:
        metadata["subject"] = 1
    with pytest.raises(ValueError):
        list(EarlyRunCalibration().split(y, metadata))


def fit_fold(X, y, metadata):
    train, calibration, test = next(EarlyRunCalibration().split(y, metadata))
    classifier = AlignedLogVariance().fit(
        X[train],
        y[train],
        subjects=metadata.iloc[train].subject.to_numpy(),
        X_target_labeled=X[calibration],
        y_target_labeled=y[calibration],
    )
    return classifier, calibration, test


def test_future_and_discarded_data_cannot_change_fitted_state():
    X, y, metadata = fixture_data()
    model, _, test = fit_fold(X, y, metadata)
    changed_X, changed_y = X.copy(), y.copy()
    # All target rows after the ten calibration trials, including discarded rows.
    changed_X[10:38] *= 1000
    changed_y[10:38] = 1 - changed_y[10:38]
    other, _, _ = fit_fold(changed_X, changed_y, metadata)
    assert_allclose(model.reference_, other.reference_)
    assert_allclose(model.classifier_.coef_, other.classifier_.coef_)
    assert model.temperature_ == other.temperature_
    reference, temperature = model.reference_.copy(), model.temperature_
    batch = model.predict_proba(X[test])
    single = np.concatenate([model.predict_proba(X[[i]]) for i in test])
    assert_allclose(batch, single)
    model.predict_proba(changed_X[test])
    assert_array_equal(model.reference_, reference)
    assert model.temperature_ == temperature


def test_target_labels_do_not_change_alignment_or_source_classifier():
    X, y, metadata = fixture_data()
    model, calibration, _ = fit_fold(X, y, metadata)
    changed_y = y.copy()
    changed_y[calibration] = 1 - y[calibration]
    other, _, _ = fit_fold(X, changed_y, metadata)
    assert_array_equal(model.reference_, other.reference_)
    assert_array_equal(model.classifier_.coef_, other.classifier_.coef_)
    assert 0.5 <= model.temperature_ <= 5.0


def test_single_class_calibration_retains_neutral_temperature():
    X, y, metadata = fixture_data()
    y[:10] = 0
    model, _, _ = fit_fold(X, y, metadata)
    assert model.temperature_ == 1
    assert "single-class" in model.temperature_status_


def test_real_moabb_process_routes_exact_arrays_in_both_modes(tmp_path, monkeypatch):
    from moabb.paradigms import LeftRightImagery

    X, y, metadata = LeftRightImagery().get_data(demo_dataset())
    folds = list(EarlyRunCalibration().split(y, metadata))
    observed = []
    original_fit = AlignedLogVariance.fit

    @wraps(original_fit)
    def recording_fit(self, source_X, source_y, **kwargs):
        observed.append((source_X.copy(), source_y.copy(), kwargs))
        return original_fit(self, source_X, source_y, **kwargs)

    monkeypatch.setattr(AlignedLogVariance, "fit", recording_fit)
    frames = run_demo(tmp_path)
    assert len(observed) == 6  # three held-out participants, two label-access modes
    for mode, frame in enumerate(frames):
        assert len(frame) == 3
        assert np.isfinite(frame.score).all()
        assert (frame.calibration_trials == 10).all()
        assert (frame.discarded_trials == 4).all()
        for (train, calibration, _), (actual_X, actual_y, kwargs) in zip(
            folds, observed[mode * 3 : mode * 3 + 3], strict=True
        ):
            encoder = LabelEncoder().fit(y[train])
            assert_allclose(actual_X, X[train])
            assert_array_equal(actual_y, encoder.transform(y[train]))
            assert_array_equal(kwargs["subjects"], metadata.iloc[train].subject.to_numpy())
            if mode == 0:
                assert set(kwargs) == {"subjects", "X_target_unlabeled"}
                assert_allclose(kwargs["X_target_unlabeled"], X[calibration])
            else:
                assert set(kwargs) == {"subjects", "X_target_labeled", "y_target_labeled"}
                assert_allclose(kwargs["X_target_labeled"], X[calibration])
                assert_array_equal(kwargs["y_target_labeled"], encoder.transform(y[calibration]))
    assert (tmp_path / "split_manifest.json").is_file()
