"""Offline participant adaptation with explicitly isolated calibration inputs.

This module has no connection to the production predictor and never fits on a
later evaluation epoch. See experiments/adaptation-v1/PROTOCOL.md.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit

CONDITIONS = {
    "baseline": "Baseline",
    "personal_temperature": "Personal temperature",
    "alignment": "Euclidean alignment",
    "alignment_temperature": "Alignment + personal temperature",
}


def partition_rows(rows, budget=10, calibration_run=4, evaluation_runs=(8, 12)):
    """Choose a chronological calibration prefix without consulting trial labels."""
    if budget < 1 or calibration_run in evaluation_runs:
        raise ValueError("Calibration budget/runs are invalid")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate trial IDs")
    calibration, evaluation = {}, {}
    for person in sorted({row["subject"] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row["subject"] == person]
        if len({rows[i]["split"] for i in indices}) != 1:
            raise ValueError("A participant crosses dataset splits")
        early = sorted(
            (i for i in indices if rows[i]["run"] == calibration_run),
            key=lambda i: (rows[i]["cue_seconds"], rows[i]["event"]),
        )
        later = sorted(
            (i for i in indices if rows[i]["run"] in evaluation_runs),
            key=lambda i: (rows[i]["run"], rows[i]["cue_seconds"], rows[i]["event"]),
        )
        if len(early) < budget or not later:
            raise ValueError(f"Participant {person} lacks the fixed calibration/evaluation budget")
        calibration[person] = np.asarray(early[:budget], dtype=int)
        evaluation[person] = np.asarray(later, dtype=int)
    return calibration, evaluation


def fit_alignment(calibration, shrinkage=0.01, eigenvalue_floor=1e-6):
    """Regularised Euclidean reference fitted only from supplied early recordings."""
    x = np.asarray(calibration, dtype=np.float64)
    if x.ndim != 3 or min(x.shape) < 1 or x.shape[-1] < 2 or not np.isfinite(x).all():
        raise ValueError("Expected nonempty finite epochs × channels × samples")
    if not 0 <= shrinkage <= 1 or not 0 < eigenvalue_floor <= 1:
        raise ValueError("Invalid alignment regularisation")
    reference = np.einsum("nct,ndt->cd", x, x) / (len(x) * x.shape[-1])
    scale = float(np.trace(reference) / x.shape[1])
    if scale <= np.finfo(float).tiny:
        raise ValueError("Cannot align a zero-energy calibration block")
    reference = (1 - shrinkage) * reference + shrinkage * scale * np.eye(x.shape[1])
    eigenvalues, vectors = np.linalg.eigh(reference)
    inverse_root = (vectors * (1 / np.sqrt(np.maximum(eigenvalues, scale * eigenvalue_floor)))) @ vectors.T
    return inverse_root


def apply_alignment(epochs, inverse_root):
    x = np.asarray(epochs)
    matrix = np.asarray(inverse_root, dtype=np.float64)
    if x.ndim != 3 or matrix.shape != (x.shape[1], x.shape[1]):
        raise ValueError("Alignment matrix does not match the epoch channels")
    if not np.isfinite(x).all() or not np.isfinite(matrix).all():
        raise ValueError("Alignment inputs must be finite")
    return np.einsum("cd,ndt->nct", matrix, x).astype(np.float32)


def fit_temperature(logits, labels, fallback=1.0, bounds=(0.5, 5.0)):
    """Bounded binary log-loss fit; a one-class calibration block uses the prior."""
    z = np.asarray(logits, dtype=float)
    y = np.asarray(labels)
    lo, hi = bounds
    if z.ndim != 1 or z.shape != y.shape or not len(z) or not np.isfinite(z).all():
        raise ValueError("Expected matching finite logits and labels")
    if not np.isin(y, [0, 1]).all() or not 0 < lo < hi or not lo <= fallback <= hi:
        raise ValueError("Invalid labels or temperature bounds")
    if len(np.unique(y)) < 2:
        return {"temperature": float(fallback), "fallback": "single_class", "at_bound": False}
    optimum = minimize_scalar(
        lambda t: float(np.mean(np.logaddexp(0, z / t) - y * z / t)),
        bounds=(lo, hi),
        method="bounded",
    )
    if not optimum.success or not np.isfinite(optimum.fun):
        raise RuntimeError("Temperature fitting did not converge")
    temperature = float(optimum.x)
    return {
        "temperature": temperature,
        "fallback": None,
        "at_bound": bool(min(temperature - lo, hi - temperature) < 1e-3),
    }


def probabilities(logits, temperature):
    temperature = np.asarray(temperature)
    if not np.isfinite(temperature).all() or (temperature <= 0).any():
        raise ValueError("Temperature must be finite and positive")
    return expit(np.asarray(logits) / temperature)


def selective_metrics(y, p, threshold):
    y, p = np.asarray(y), np.asarray(p)
    if y.ndim != 1 or p.shape != y.shape or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected matching binary labels and probabilities")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not 0.5 <= threshold <= 1:
        raise ValueError("Invalid probabilities or threshold")
    keep = np.maximum(p, 1 - p) >= threshold
    correct = int(((p[keep] >= 0.5) == y[keep]).sum())
    count = int(keep.sum())
    return {
        "threshold": float(threshold),
        "n": len(y),
        "retained": count,
        "correct": correct,
        "coverage": float(keep.mean()),
        "accuracy": correct / count if count else None,
    }


def select_threshold(y, p, config):
    """Only call with validation predictions; never optimise this on test labels."""
    for threshold in config["thresholds"]:
        point = selective_metrics(y, p, threshold)
        if (
            point["retained"] >= config["target_retained"]
            and point["coverage"] >= config["target_coverage"]
            and point["accuracy"] >= config["target_accuracy"]
        ):
            return {"target_met": True, **point}
    return {"target_met": False, **selective_metrics(y, p, config["fallback_threshold"])}


def binary_metrics(y, p, subjects):
    y, p, subjects = np.asarray(y), np.asarray(p), np.asarray(subjects)
    selective_metrics(y, p, 0.5)  # Validate nonempty binary inputs.
    if subjects.shape != y.shape:
        raise ValueError("Subjects must match the labels")
    correct = (p >= 0.5) == y
    people = []
    for person in np.unique(subjects):
        mask = subjects == person
        if len(np.unique(y[mask])) != 2:
            raise ValueError("Both classes are required in each participant's evaluation block")
        people.append(
            {
                "subject": int(person),
                "n": int(mask.sum()),
                "balanced_accuracy": float(np.mean([correct[mask & (y == c)].mean() for c in (0, 1)])),
                "brier": float(np.mean((p[mask] - y[mask]) ** 2)),
            }
        )
    confidence = np.maximum(p, 1 - p)
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = sum(
        abs(confidence[bins == b].mean() - correct[bins == b].mean()) * (bins == b).mean()
        for b in np.unique(bins)
    )
    return {
        "n": len(y),
        "participants": people,
        "participant_balanced_accuracy": float(np.mean([r["balanced_accuracy"] for r in people])),
        "accuracy": float(correct.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": float(ece),
    }


def paired_interval(baseline, candidate, repeats=2000, seed=4027):
    """Rows are seeds, columns are the SAME participants in the SAME order."""
    baseline, candidate = np.asarray(baseline), np.asarray(candidate)
    if baseline.ndim != 2 or baseline.shape != candidate.shape or not baseline.size:
        raise ValueError("Expected paired seed × participant matrices")
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("Paired scores must be finite")
    differences = (candidate - baseline).mean(axis=0)
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(differences, (repeats, len(differences)), replace=True).mean(axis=1)
    return {
        "mean_difference": float(differences.mean()),
        "bootstrap_95_ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "participants": len(differences),
        "seeds": baseline.shape[0],
    }
