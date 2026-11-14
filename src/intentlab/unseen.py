"""Fixed classical models and participant based evaluation for a separate study.

Training functions accept training arrays only. Predictions never update a
reference or scaler from the target batch. This module is not a serving model.
"""

from collections import Counter
from numbers import Integral

import numpy as np

MODEL_NAMES = ("bandpower", "covariance_tangent")
THRESHOLDS = tuple(value / 100 for value in range(50, 100, 5))
THRESHOLD_RULE = {
    "thresholds": list(THRESHOLDS),
    "target_participant_macro_accuracy": 0.75,
    "target_participant_macro_coverage": 0.20,
    "minimum_retained_trials": 100,
    "minimum_retained_per_participant": 5,
    "fallback_threshold": 0.75,
}


def _identifiers(values, size, name):
    values = list(values)
    if len(values) != size:
        raise ValueError(f"{name} must match the number of trials")
    cleaned = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (str, Integral)):
            raise ValueError(f"{name} identifiers must be strings or integers")
        value = str(value)
        if (
            not 1 <= len(value) <= 128
            or value != value.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise ValueError(f"Invalid {name} identifier")
        cleaned.append(value)
    return np.asarray(cleaned)


def _labels(labels):
    result = np.asarray(labels)
    if result.ndim != 1 or not len(result) or result.dtype.kind not in "iuf":
        raise ValueError("Labels must be a nonempty numeric vector with 0 for left and 1 for right")
    if not np.isfinite(result).all() or not np.isin(result, [0, 1]).all():
        raise ValueError("Labels must contain only 0 for left and 1 for right")
    return result.astype(np.int64)


def participant_class_weights(labels, participants):
    """Give each training participant and each of their two classes equal weight."""
    labels = _labels(labels)
    participants = _identifiers(participants, len(labels), "Participant")
    people = sorted(set(participants))
    weights = np.empty(len(labels), dtype=np.float64)
    for person in people:
        for label in (0, 1):
            mask = (participants == person) & (labels == label)
            count = int(mask.sum())
            if not count:
                raise ValueError("Each participant must contribute both classes")
            weights[mask] = len(labels) / (2 * len(people) * count)
    return weights


def preprocess_epochs(epochs, sampling_hz=512):
    """Demean, filter each completed four second epoch, then divide by one RMS.

    Input is already re referenced to all 27 EEG channels and restricted to the
    fixed nine central channels. Filtering uses the completed epoch and is not
    a causal streaming transformation. No statistics are fitted across epochs.
    """
    from scipy.signal import butter, sosfiltfilt

    if sampling_hz != 512:
        raise ValueError("The frozen study requires 512 Hz sampling")
    values = np.asarray(epochs, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (9, 2048) or not len(values):
        raise ValueError("Expected a nonempty array of nine channel, four second epochs at 512 Hz")
    if not np.isfinite(values).all():
        raise ValueError("Epochs must contain only finite samples")
    centred = values - values.mean(axis=-1, keepdims=True)
    coefficients = butter(4, [8, 30], btype="bandpass", fs=sampling_hz, output="sos")
    filtered = sosfiltfilt(coefficients, centred, axis=-1)
    rms = np.sqrt(np.mean(filtered**2, axis=(1, 2), keepdims=True))
    if not np.isfinite(rms).all() or np.any(rms <= 0):
        raise ValueError("An epoch has zero or nonfinite filtered RMS")
    return filtered / rms


def extract_features(epochs, sampling_hz=512):
    """Return 27 log band powers and nine channel OAS covariance matrices."""
    from scipy.signal import welch
    from sklearn.covariance import oas

    values = preprocess_epochs(epochs, sampling_hz)
    frequencies, density = welch(values, fs=512, nperseg=512, noverlap=256, axis=-1)
    bands = [
        np.log(np.maximum(density[..., (frequencies >= low) & (frequencies < high)].sum(-1), 1e-10))
        for low, high in ((8, 13), (13, 20), (20, 30))
    ]
    covariances = np.stack([oas(epoch.T, assume_centered=False)[0] for epoch in values])
    covariances /= np.trace(covariances, axis1=1, axis2=2)[:, None, None]
    return _features(
        {"bandpower": np.stack(bands, axis=-1).reshape(len(values), 27), "covariance": covariances}
    )


def _features(features):
    if not isinstance(features, dict) or set(features) != {"bandpower", "covariance"}:
        raise ValueError("Features must contain exactly bandpower and covariance arrays")
    bandpower = np.asarray(features["bandpower"], dtype=np.float64)
    covariance = np.asarray(features["covariance"], dtype=np.float64)
    if bandpower.ndim != 2 or not len(bandpower) or bandpower.shape[1] != 27:
        raise ValueError("Bandpower must have shape (trials, 27)")
    if covariance.shape != (len(bandpower), 9, 9):
        raise ValueError("Covariance must have shape (trials, 9, 9) and match bandpower")
    if not np.isfinite(bandpower).all() or not np.isfinite(covariance).all():
        raise ValueError("Features must be finite")
    if not np.allclose(covariance, covariance.swapaxes(-1, -2), rtol=1e-10, atol=1e-12):
        raise ValueError("Covariance matrices must be symmetric")
    if np.any(np.linalg.eigvalsh(covariance) <= 0):
        raise ValueError("Covariance matrices must be positive definite")
    if not np.allclose(np.trace(covariance, axis1=1, axis2=2), 1, rtol=1e-10, atol=1e-12):
        raise ValueError("Covariance matrices must have unit trace")
    return {"bandpower": bandpower, "covariance": covariance}


def fit_models(train_features, train_labels, train_participants):
    """Fit two fixed pipelines using only explicitly supplied training arrays.

    The same participant and class weights are passed to the tangent reference,
    each scaler and each logistic regression. Class weights are not applied a
    second time. The signature has no validation or test inputs.
    """
    import inspect
    import warnings

    from pyriemann.tangentspace import TangentSpace
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    features = _features(train_features)
    labels = _labels(train_labels)
    if len(labels) != len(features["bandpower"]):
        raise ValueError("Training labels and features must contain the same number of trials")
    weights = participant_class_weights(labels, train_participants)
    if "sample_weight" not in inspect.signature(TangentSpace.fit).parameters:
        raise RuntimeError("Installed TangentSpace does not support the required training weights")

    def tail():
        return [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(C=1.0, max_iter=5000, random_state=2027, class_weight=None),
            ),
        ]

    models = {
        "bandpower": Pipeline(tail()),
        "covariance_tangent": Pipeline(
            [("tangent", TangentSpace(metric="riemann", tsupdate=False)), *tail()]
        ),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.filterwarnings("error", message="Convergence not reached", category=UserWarning)
        models["bandpower"].fit(
            features["bandpower"], labels, scale__sample_weight=weights, classifier__sample_weight=weights
        )
        models["covariance_tangent"].fit(
            features["covariance"],
            labels,
            tangent__sample_weight=weights,
            scale__sample_weight=weights,
            classifier__sample_weight=weights,
        )
    return models


def predict_models(models, features):
    """Predict left and right probabilities without target fitting or adaptation."""
    features = _features(features)
    if not isinstance(models, dict) or set(models) != set(MODEL_NAMES):
        raise ValueError("Both fixed study models are required")
    if models["covariance_tangent"].named_steps["tangent"].tsupdate:
        raise ValueError("Target tangent reference updates are forbidden in this study")
    output = {}
    for name, feature in (("bandpower", "bandpower"), ("covariance_tangent", "covariance")):
        model = models[name]
        if list(model.classes_) != [0, 1]:
            raise ValueError("Model classes must be ordered left (0), right (1)")
        output[name] = _probabilities(model.predict_proba(features[feature]), len(features[feature]))
    return output


def _probabilities(probabilities, size):
    values = np.asarray(probabilities, dtype=np.float64)
    if (
        values.shape != (size, 2)
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values > 1)
        or not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-9)
    ):
        raise ValueError("Probabilities must have shape (trials, 2), be finite and sum to one")
    return values


def _evaluation_inputs(labels, probabilities, participants, cohorts):
    labels = _labels(labels)
    probabilities = _probabilities(probabilities, len(labels))
    participants = _identifiers(participants, len(labels), "Participant")
    cohorts = _identifiers(cohorts, len(labels), "Cohort")
    if not set(cohorts) <= {"A", "B", "C"}:
        raise ValueError("Source cohorts must be A, B or C")
    for person in set(participants):
        mask = participants == person
        if len(set(cohorts[mask])) != 1:
            raise ValueError("A participant must belong to exactly one source cohort")
        if set(labels[mask]) != {0, 1}:
            raise ValueError("Every evaluated participant must have both classes")
    return labels, probabilities, participants, cohorts


def _classification(labels, probabilities):
    from sklearn.metrics import roc_auc_score

    # An exact tie is assigned to the left class, matching numpy argmax.
    predicted = probabilities.argmax(axis=1)
    correct = predicted == labels
    confusion = [
        [int(((labels == truth) & (predicted == guess)).sum()) for guess in (0, 1)] for truth in (0, 1)
    ]
    return {
        "balanced_accuracy": float(np.mean([correct[labels == label].mean() for label in (0, 1)])),
        "accuracy": float(correct.mean()),
        "roc_auc": float(roc_auc_score(labels, probabilities[:, 1])),
        "brier": float(np.mean((probabilities[:, 1] - labels) ** 2)),
        "confusion_matrix": confusion,
    }


def _calibration(labels, probabilities):
    right = probabilities[:, 1]
    bins = np.minimum((right * 10).astype(int), 9)
    return [
        {
            "lower": index / 10,
            "upper": (index + 1) / 10,
            "count": int((bins == index).sum()),
            "mean_probability": float(right[bins == index].mean()) if np.any(bins == index) else None,
            "observed_right_fraction": float(labels[bins == index].mean()) if np.any(bins == index) else None,
        }
        for index in range(10)
    ]


def evaluate_predictions(labels, probabilities, participants, cohorts, threshold=None):
    """Report equal participant metrics and conditional retained trial accuracy.

    With no threshold, all trials are retained. Participants with zero retained
    trials have null accepted accuracy; their coverage remains zero and they
    remain in the participant table and all unthresholded metrics.
    """
    labels, probabilities, participants, cohorts = _evaluation_inputs(
        labels, probabilities, participants, cohorts
    )
    if threshold is not None and (
        isinstance(threshold, bool) or not np.isfinite(threshold) or not 0.5 <= threshold <= 1
    ):
        raise ValueError("Threshold must lie between 0.5 and 1")
    accepted = (
        np.ones(len(labels), dtype=bool) if threshold is None else probabilities.max(axis=1) >= threshold
    )
    correct = probabilities.argmax(axis=1) == labels
    people = []
    for person in sorted(set(participants)):
        mask = participants == person
        retained = mask & accepted
        count, successful = int(retained.sum()), int((retained & correct).sum())
        metrics = _classification(labels[mask], probabilities[mask])
        people.append(
            {
                "participant": str(person),
                "cohort": str(cohorts[mask][0]),
                "trials": int(mask.sum()),
                "left_trials": int((mask & (labels == 0)).sum()),
                "right_trials": int((mask & (labels == 1)).sum()),
                **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
                "accepted": count,
                "correct_accepted": successful,
                "coverage": count / int(mask.sum()),
                "accepted_accuracy": successful / count if count else None,
            }
        )
    retained_count = int(accepted.sum())
    successful = int((accepted & correct).sum())
    defined = [person["accepted_accuracy"] for person in people if person["accepted_accuracy"] is not None]
    pooled = _classification(labels, probabilities)
    pooled["calibration"] = _calibration(labels, probabilities)
    return {
        "trials": len(labels),
        "participant_count": len(people),
        "participant_macro_balanced_accuracy": float(np.mean([p["balanced_accuracy"] for p in people])),
        "participant_macro_accuracy": float(np.mean([p["accuracy"] for p in people])),
        "participant_macro_auc": float(np.mean([p["roc_auc"] for p in people])),
        "participant_macro_brier": float(np.mean([p["brier"] for p in people])),
        "pooled": pooled,
        "participants": people,
        "cohorts": [
            {
                "cohort": cohort,
                "participant_count": sum(person["cohort"] == cohort for person in people),
                "participant_macro_balanced_accuracy": float(
                    np.mean([person["balanced_accuracy"] for person in people if person["cohort"] == cohort])
                ),
            }
            for cohort in sorted(set(cohorts))
        ],
        "selective": {
            "threshold": threshold,
            "accepted": retained_count,
            "correct_accepted": successful,
            "pooled_accuracy": successful / retained_count if retained_count else None,
            "participant_macro_accuracy": float(np.mean(defined)) if defined else None,
            "participant_macro_coverage": float(np.mean([person["coverage"] for person in people])),
            "pooled_coverage": retained_count / len(labels),
            "participants_with_acceptance": len(defined),
            "participants_without_acceptance": len(people) - len(defined),
            "minimum_accepted_per_participant": min(person["accepted"] for person in people),
        },
    }


def choose_model(validation_summaries):
    """Select using validation mean balanced accuracy, breaking exact ties simply."""
    if set(validation_summaries) != set(MODEL_NAMES):
        raise ValueError("Validation summaries must contain both fixed model names")
    values = {}
    for name in MODEL_NAMES:
        value = validation_summaries[name]["participant_macro_balanced_accuracy"]
        if isinstance(value, bool) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid validation balanced accuracy")
        values[name] = value
    return "covariance_tangent" if values["covariance_tangent"] > values["bandpower"] else "bandpower"


def select_threshold(
    validation_labels, validation_probabilities, validation_participants, validation_cohorts
):
    """Select on source validation people only; a fixed fallback exposes failure."""
    candidates = [
        evaluate_predictions(
            validation_labels,
            validation_probabilities,
            validation_participants,
            validation_cohorts,
            threshold,
        )["selective"]
        for threshold in THRESHOLDS
    ]
    qualifying = [
        candidate
        for candidate in candidates
        if candidate["participant_macro_accuracy"] is not None
        and candidate["participant_macro_accuracy"] >= THRESHOLD_RULE["target_participant_macro_accuracy"]
        and candidate["participant_macro_coverage"] >= THRESHOLD_RULE["target_participant_macro_coverage"]
        and candidate["accepted"] >= THRESHOLD_RULE["minimum_retained_trials"]
        and candidate["minimum_accepted_per_participant"]
        >= THRESHOLD_RULE["minimum_retained_per_participant"]
    ]
    return {
        "threshold": qualifying[0]["threshold"] if qualifying else THRESHOLD_RULE["fallback_threshold"],
        "validation_target_met": bool(qualifying),
        "rule": dict(THRESHOLD_RULE, thresholds=list(THRESHOLDS)),
        "candidates": candidates,
    }


def stratified_participant_draws(participant_cohorts, repeats=5000, seed=2027):
    """Return paired bootstrap indices with the source cohort counts held fixed."""
    if isinstance(repeats, bool) or not isinstance(repeats, Integral) or not 100 <= repeats <= 100000:
        raise ValueError("Use 100 to 100000 participant bootstrap replicates")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("Bootstrap seed must be a nonnegative integer")
    cohorts = _identifiers(participant_cohorts, len(participant_cohorts), "Cohort")
    if not len(cohorts) or not set(cohorts) <= {"A", "B", "C"}:
        raise ValueError("Participant cohorts must be a nonempty A, B or C vector")
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [
            rng.choice(indices, size=(repeats, len(indices)), replace=True)
            for cohort in sorted(set(cohorts))
            for indices in [np.flatnonzero(cohorts == cohort)]
        ],
        axis=1,
    )


def bootstrap_comparison(labels, probabilities_by_model, participants, cohorts, repeats=5000, seed=2027):
    """Resample complete test participants, paired across the two fixed models.

    These intervals condition on fitted models and observed cohort composition.
    They do not resample training, and no individual trial is an independent
    resampling unit. Test labels are used for scoring only, never selection.
    """
    if set(probabilities_by_model) != set(MODEL_NAMES):
        raise ValueError("Bootstrap comparison requires both fixed study models")
    summaries = {
        name: evaluate_predictions(labels, probabilities_by_model[name], participants, cohorts)
        for name in MODEL_NAMES
    }
    people = summaries["bandpower"]["participants"]
    draws = stratified_participant_draws([person["cohort"] for person in people], repeats, seed)
    scores = {
        name: np.asarray([person["balanced_accuracy"] for person in summaries[name]["participants"]])
        for name in MODEL_NAMES
    }
    sampled = {name: values[draws].mean(axis=1) for name, values in scores.items()}
    difference = scores["covariance_tangent"] - scores["bandpower"]
    return {
        "unit": "participant",
        "repeats": int(repeats),
        "seed": int(seed),
        "strata_counts": dict(Counter(person["cohort"] for person in people)),
        "participant_ids": [person["participant"] for person in people],
        "model_intervals": {
            name: np.quantile(sampled[name], [0.025, 0.975]).tolist() for name in MODEL_NAMES
        },
        "paired_difference": {
            "direction": "covariance_tangent minus bandpower",
            "estimate": float(difference.mean()),
            "ci95": np.quantile(difference[draws].mean(axis=1), [0.025, 0.975]).tolist(),
        },
    }
