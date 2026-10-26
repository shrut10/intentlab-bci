"""Auditable split, label-budget and reporting utilities for pretraining-v1.

This module deliberately has no PyTorch dependency; training lives in the runner.
"""

import numpy as np


def development_rows(subjects, splits):
    subjects, splits = np.asarray(subjects), np.asarray(splits)
    if subjects.ndim != 1 or splits.shape != subjects.shape:
        raise ValueError("Expected matching one-dimensional participant and split arrays")
    if not set(splits) <= {"train", "validation", "test"}:
        raise ValueError("Unknown split")
    groups = {name: np.flatnonzero(splits == name) for name in ("train", "validation", "test")}
    if any(len(rows) == 0 for rows in groups.values()):
        raise ValueError("All original splits must be present")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(subjects[groups[left]]) & set(subjects[groups[right]]):
            raise ValueError("Participant overlap across splits")
    # The test indices are deliberately not returned to the training runner.
    return groups["train"], groups["validation"]


def labelled_subset(labels, subjects, fraction, seed):
    """Select source-local positions, stratified by participant and class.

    At least one trial per present participant/class cell, otherwise floor(n*f).
    These are labelled training budgets, not new-person calibration budgets.
    """
    labels, subjects = np.asarray(labels), np.asarray(subjects)
    if labels.ndim != 1 or subjects.shape != labels.shape or not 0 < fraction <= 1:
        raise ValueError("Invalid labels, participants or fraction")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Both binary classes are required")
    rng = np.random.default_rng(seed)
    selected = []
    for person in np.unique(subjects):
        for label in (0, 1):
            rows = np.flatnonzero((subjects == person) & (labels == label))
            if len(rows) == 0:
                raise ValueError("Each source participant must have both classes")
            n = max(1, int(np.floor(len(rows) * fraction)))
            selected.extend(rng.permutation(rows)[:n].tolist())
    return np.asarray(sorted(selected), dtype=int)


def mask_channels(x, rng, count=2):
    """Return a copy and Boolean mask, hiding exactly count whole channels."""
    x = np.asarray(x)
    if x.ndim != 3 or not 0 < count < x.shape[1] or not np.isfinite(x).all():
        raise ValueError("Expected finite trial/channel/time data and a partial channel mask")
    mask = np.zeros(x.shape[:2], dtype=bool)
    selected = np.argsort(rng.random(x.shape[:2]), axis=1)[:, :count]
    np.put_along_axis(mask, selected, True, axis=1)
    return np.where(mask[:, :, None], 0, x), mask


def participant_metrics(labels, probabilities, subjects):
    labels, probabilities, subjects = map(np.asarray, (labels, probabilities, subjects))
    if labels.ndim != 1 or probabilities.shape != labels.shape or subjects.shape != labels.shape:
        raise ValueError("Labels, probabilities and participants must have matching shapes")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid probabilities")
    people = []
    for person in np.unique(subjects):
        rows = subjects == person
        y, p = labels[rows], probabilities[rows]
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("Each evaluation participant must have both classes")
        balanced_accuracy = np.mean([np.mean((p[y == label] >= 0.5) == label) for label in (0, 1)])
        people.append(
            {
                "subject": int(person),
                "trials": int(rows.sum()),
                "balanced_accuracy": float(balanced_accuracy),
                "brier": float(np.mean((p - y) ** 2)),
            }
        )
    return {
        "participant_balanced_accuracy": float(np.mean([p["balanced_accuracy"] for p in people])),
        "participant_brier": float(np.mean([p["brier"] for p in people])),
        "participants": people,
    }


def summarise_pairs(results, bootstrap_samples=2000, seed=5027):
    """Average seeds within each participant before a paired participant bootstrap."""
    summaries = []
    for budget in sorted({r["label_fraction"] for r in results}):
        records = [r for r in results if r["label_fraction"] == budget]
        seeds = sorted({r["seed"] for r in records})
        people = sorted({p["subject"] for r in records for p in r["validation"]["participants"]})
        arrays = {}
        for arm in ("scratch", "masked_pretraining"):
            matrix = []
            for run_seed in seeds:
                matched = [r for r in records if r["arm"] == arm and r["seed"] == run_seed]
                if len(matched) != 1:
                    raise ValueError("Exactly one record is required per seed, arm and label budget")
                rows = matched[0]["validation"]["participants"]
                by_person = {p["subject"]: p for p in rows}
                if len(by_person) != len(rows) or set(by_person) != set(people):
                    raise ValueError("Participant identities must match across all paired runs")
                matrix.append([by_person[p]["balanced_accuracy"] for p in people])
            arrays[arm] = np.asarray(matrix)
        deltas = (arrays["masked_pretraining"] - arrays["scratch"]).mean(axis=0)
        rng = np.random.default_rng(seed)
        boot = rng.choice(deltas, size=(bootstrap_samples, len(people)), replace=True).mean(axis=1)
        summaries.append(
            {
                "label_fraction": float(budget),
                "labelled_trials_per_seed": {
                    str(s): next(r["labelled_trials"] for r in records if r["seed"] == s) for s in seeds
                },
                "seeds": seeds,
                "participants": len(people),
                "scratch_mean_balanced_accuracy": float(arrays["scratch"].mean()),
                "pretrained_mean_balanced_accuracy": float(arrays["masked_pretraining"].mean()),
                "paired_difference": float(deltas.mean()),
                "paired_difference_95_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
                "participant_differences": [
                    {"subject": p, "difference": float(d)} for p, d in zip(people, deltas, strict=True)
                ],
            }
        )
    return summaries
