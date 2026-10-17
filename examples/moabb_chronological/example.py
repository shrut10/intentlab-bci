"""A no-download, chronological target-calibration example for MOABB.

Run from the repository root: python examples/moabb_chronological/example.py
This illustrates the evaluation interface, not a real-EEG performance result.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
from moabb.datasets import FakeDataset
from moabb.evaluations import CrossSubjectEvaluation
from moabb.paradigms import LeftRightImagery
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn import config_context
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_is_fitted


class EarlyRunCalibration(BaseCrossValidator):
    """Leave one person out; adapt on early trials and score only later runs.

    MOABB calls ``split(y, metadata)``, not sklearn's usual ``split(X, y)``.
    Within each run, rows MUST already follow acquisition order. Run chronology
    is specified explicitly; it is not inferred by sorting names. Only a single
    session per person is supported here, to avoid ambiguous cross-session order.
    Indices returned are positional, irrespective of the DataFrame index labels.
    """

    metadata_columns = (
        "calibration_size",
        "calibration_labeled",
        "calibration_trials",
        "discarded_trials",
    )

    def __init__(self, budget=10, run_order=("0", "1", "2"), labeled=True):
        self.budget = budget
        self.run_order = run_order
        self.labeled = labeled

    def get_n_splits(self, X=None, y=None, groups=None, *, metadata=None):
        # MOABB's metadata-aware interface uses keyword `metadata`.
        if metadata is None:
            raise ValueError("metadata is required to count held-out participants")
        return metadata["subject"].nunique()

    def split(self, y, metadata):
        if not isinstance(self.budget, (int, np.integer)) or isinstance(self.budget, bool) or self.budget < 1:
            raise ValueError("budget must be a positive integer")
        if len(self.run_order) < 2 or len(set(self.run_order)) != len(self.run_order):
            raise ValueError("run_order must contain distinct early and later run names")
        required = ["subject", "session", "run"]
        if not set(required).issubset(metadata.columns) or metadata[required].isna().any().any():
            raise ValueError("non-missing subject, session and run metadata are required")
        if len(y) != len(metadata) or metadata["subject"].nunique() < 2:
            raise ValueError("matching labels and metadata for at least two participants are required")
        subjects = metadata["subject"].to_numpy()
        runs = metadata["run"].to_numpy()
        folds = []
        for subject in metadata["subject"].unique():
            target = subjects == subject
            rows = metadata.iloc[np.flatnonzero(target)]
            if rows["session"].nunique() != 1:
                raise ValueError("this example requires exactly one session per participant")
            if set(rows["run"]) != set(self.run_order):
                raise ValueError(f"participant {subject}: missing or unexpected runs")
            early = np.flatnonzero(target & (runs == self.run_order[0]))
            if len(early) < self.budget:
                raise ValueError(f"participant {subject}: insufficient early calibration trials")
            train = np.flatnonzero(~target)
            calibration = early[: self.budget]
            test = np.concatenate([np.flatnonzero(target & (runs == run)) for run in self.run_order[1:]])
            folds.append((train, calibration, test, len(early) - self.budget, int(target.sum())))
        # Validate every participant before yielding any fold.
        for train, calibration, test, discarded, total in folds:
            self._last_metadata = {
                # This key enables MOABB's metadata routing. It describes the
                # realised target fraction; selection above uses an absolute count.
                "calibration_size": self.budget / total,
                "calibration_labeled": self.labeled,
                "calibration_trials": self.budget,
                "discarded_trials": discarded,
            }
            yield train, calibration, test

    def get_metadata(self):
        return dict(self._last_metadata)


def alignment_reference(X):
    """Regularised inverse square root of the mean epoch second moment."""
    reference = np.einsum("nct,ndt->cd", X, X) / (len(X) * X.shape[2])
    channels = reference.shape[0]
    scale = np.trace(reference) / channels
    reference = 0.99 * reference + 0.01 * scale * np.eye(channels)
    eigenvalues, eigenvectors = np.linalg.eigh(reference)
    floor = max(scale * 1e-6, np.finfo(float).tiny)
    return (eigenvectors * np.maximum(eigenvalues, floor) ** -0.5) @ eigenvectors.T


def features(X, reference):
    aligned = np.einsum("cd,ndt->nct", reference, X)
    return np.log(np.maximum(np.mean(aligned**2, axis=2), np.finfo(float).tiny))


class AlignedLogVariance(ClassifierMixin, BaseEstimator):
    """Small illustrative decoder with explicit target-data access.

    Source labels train logistic regression on per-channel log power, after
    independently aligning each source participant using their training data.
    The SAME ten target epochs supply an unlabelled alignment reference and,
    only in labelled mode, labels for fitting one temperature parameter.
    All preprocessing lives here so raw calibration epochs cannot bypass a
    fitted pipeline transform by accident. Prediction never fits a statistic.
    """

    def fit(self, X, y, *, subjects, X_target_labeled=None, y_target_labeled=None, X_target_unlabeled=None):
        if (X_target_labeled is None) == (X_target_unlabeled is None):
            raise ValueError("provide exactly one labelled or unlabelled calibration slice")
        if (X_target_labeled is None) != (y_target_labeled is None):
            raise ValueError("labelled calibration requires both epochs and labels")
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("this teaching classifier supports two classes")
        X = np.asarray(X)
        subjects = np.asarray(subjects)
        if len(subjects) != len(X):
            raise ValueError("source participant identities must match the training epochs")
        source_features = np.empty((len(X), X.shape[1]))
        for subject in np.unique(subjects):
            mask = subjects == subject
            source_features[mask] = features(X[mask], alignment_reference(X[mask]))
        self.classifier_ = LogisticRegression(C=1.0, random_state=42).fit(source_features, y)
        target = np.asarray(X_target_labeled if X_target_labeled is not None else X_target_unlabeled)
        self.reference_ = alignment_reference(target)  # no target labels used here
        self.calibration_count_ = len(target)
        self.temperature_ = 1.0
        self.temperature_status_ = "unlabelled calibration"
        if y_target_labeled is not None:
            labels = np.asarray(y_target_labeled)
            if len(labels) != len(target) or not np.isin(labels, self.classes_).all():
                raise ValueError("calibration labels must match epochs and source classes")
            if len(np.unique(labels)) == 2:
                logits = self.classifier_.decision_function(features(target, self.reference_))
                positive = labels == self.classes_[1]

                def loss(temperature):
                    scaled = logits / temperature
                    return np.mean(np.logaddexp(0, scaled) - positive * scaled)

                optimum = minimize_scalar(loss, bounds=(0.5, 5.0), method="bounded")
                if not optimum.success:
                    raise RuntimeError("temperature optimisation failed")
                self.temperature_ = float(optimum.x)
                self.temperature_status_ = "fitted on calibration labels"
            else:
                self.temperature_status_ = "single-class calibration: retained T=1"
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["reference_", "classifier_", "temperature_"])
        logits = self.classifier_.decision_function(features(np.asarray(X), self.reference_))
        positive = expit(logits / self.temperature_)
        return np.column_stack((1 - positive, positive))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def requested_classifier():
    with config_context(enable_metadata_routing=True):
        return AlignedLogVariance().set_fit_request(
            subjects=True, X_target_labeled=True, y_target_labeled=True, X_target_unlabeled=True
        )


def demo_dataset():
    return FakeDataset(
        event_list=["left_hand", "right_hand"],
        n_subjects=3,
        n_sessions=1,
        n_runs=3,
        n_events=[14, 12, 12],
        duration=[60, 60, 60],
        seed=42,
    )


def run_demo(output_directory):
    """Exercise real MOABB processing in both labelled and unlabelled modes."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    dataset = demo_dataset()
    paradigm = LeftRightImagery()
    _, labels, metadata = paradigm.get_data(dataset)
    manifest = []
    for train, calibration, test in EarlyRunCalibration().split(labels, metadata):
        manifest.append(
            {
                "target_subject": int(metadata.iloc[test[0]]["subject"]),
                "train_positions": train.tolist(),
                "calibration_positions": calibration.tolist(),
                "test_positions": test.tolist(),
                "discarded_positions": np.setdiff1d(
                    np.arange(len(metadata)), np.concatenate((train, calibration, test))
                ).tolist(),
            }
        )
    (output_directory / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    frames = []
    for labeled in (False, True):
        name = "alignment_and_temperature" if labeled else "alignment_only"
        evaluation = CrossSubjectEvaluation(
            paradigm=paradigm,
            datasets=[dataset],
            splitter=EarlyRunCalibration(labeled=labeled),
            hdf5_path=str(output_directory / name),
            suffix="chronological-example",
            overwrite=True,
            random_state=42,
            n_jobs=1,
            error_score="raise",
        )
        results = evaluation.process({name: requested_classifier()})
        results.to_csv(output_directory / f"{name}.csv", index=False)
        frames.append(results)
    return frames


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="moabb-chronological-") as output:
        for frame in run_demo(output):
            print(
                frame[["subject", "pipeline", "score", "calibration_trials", "discarded_trials"]].to_string(
                    index=False
                )
            )
        print("Synthetic integration check only; score is ROC AUC, not evidence of EEG decoding accuracy")
        print(
            "Each fold: 76 source trials, 10 calibration trials, 4 discarded early trials, 24 later test trials"
        )
