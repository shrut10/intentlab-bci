"""Participant-aware audits of matched, saved binary predictions; no model fitting."""

from collections import defaultdict

import numpy as np

REQUIRED = {"condition", "seed", "split", "subject", "trial_id", "label", "probability"}
DEFAULT_RULE = {
    "thresholds": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    "target_accuracy": 0.75,
    "target_coverage": 0.2,
    "target_retained": 100,
    "fallback_threshold": 0.75,
}


def validate_predictions(rows):
    """Require disjoint people, stable trial identities and matched comparison sets."""
    groups, identities, people = defaultdict(list), {}, defaultdict(set)
    for number, raw in enumerate(rows, 1):
        if not REQUIRED.issubset(raw):
            raise ValueError(f"Row {number}: missing columns {sorted(REQUIRED - raw.keys())}")
        row = {key: str(raw[key]).strip() for key in REQUIRED - {"label", "probability"}}
        if any(not value or len(value) > 160 for value in row.values()):
            raise ValueError(f"Row {number}: identifiers must contain 1–160 characters")
        if row["split"] not in {"validation", "test"}:
            raise ValueError(f"Row {number}: split must be validation or test")
        try:
            label, probability = float(raw["label"]), float(raw["probability"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Row {number}: numeric binary label and probability required") from error
        if label not in {0, 1} or not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"Row {number}: binary label and finite probability in [0, 1] required")
        row.update(label=int(label), probability=probability)
        identity = (row["subject"], row["split"], row["label"])
        if identities.setdefault(row["trial_id"], identity) != identity:
            raise ValueError("Trial metadata must stay identical across conditions and seeds")
        groups[(row["condition"], row["seed"], row["split"])].append(row)
        people[row["split"]].add(row["subject"])
    if not groups:
        raise ValueError("Prediction table is empty")
    if people["validation"] & people["test"]:
        raise ValueError("Validation and test participants must be disjoint")
    expected = {}
    pairs = {(condition, seed) for condition, seed, _ in groups}
    for condition, seed in sorted(pairs):
        for split in ("validation", "test"):
            group = groups.get((condition, seed, split), [])
            ids = {r["trial_id"] for r in group}
            if not group or len(ids) != len(group):
                raise ValueError("Each condition/seed needs unique validation and test trials")
            if expected.setdefault(split, ids) != ids:
                raise ValueError("All conditions and seeds must contain the same trials in each split")
    return groups


def validate_rule(rule):
    rule = {key: rule[key] for key in DEFAULT_RULE}
    thresholds = rule["thresholds"]
    if (
        not isinstance(thresholds, list)
        or not 1 <= len(thresholds) <= 21
        or thresholds != sorted(set(thresholds))
        or not all(np.isfinite(t) and 0.5 <= t <= 1 for t in thresholds)
        or rule["fallback_threshold"] not in thresholds
        or not 0 <= rule["target_accuracy"] <= 1
        or not 0 <= rule["target_coverage"] <= 1
        or not isinstance(rule["target_retained"], int)
        or rule["target_retained"] < 1
    ):
        raise ValueError("Invalid threshold rule; use sorted unique thresholds including the fallback")
    return rule


def _interval(values):
    valid = values[np.isfinite(values)]
    return np.quantile(valid, [0.025, 0.975]).tolist() if len(valid) else None


def _points(rows, thresholds, repeats, seed):
    rows = sorted(rows, key=lambda r: (r["subject"], r["trial_id"]))
    subjects = sorted({r["subject"] for r in rows})
    y = np.array([r["label"] for r in rows])
    p = np.array([r["probability"] for r in rows])
    masks = [np.array([r["subject"] == subject for r in rows]) for subject in subjects]
    correct = (p >= 0.5) == y
    counts = np.array([mask.sum() for mask in masks])
    # Resample complete participants, never individual trials or model seeds.
    draws = np.random.default_rng(seed).integers(0, len(subjects), size=(repeats, len(subjects)))
    bootstrap_n = counts[draws].sum(axis=1)
    output = []
    for threshold in thresholds:
        accepted = np.maximum(p, 1 - p) >= threshold
        retained = np.array([(accepted & mask).sum() for mask in masks])
        successes = np.array([(accepted & correct & mask).sum() for mask in masks])
        total_retained, total_correct = int(retained.sum()), int(successes.sum())
        boot_retained = retained[draws].sum(axis=1)
        boot_correct = successes[draws].sum(axis=1)
        boot_accuracy = np.divide(
            boot_correct, boot_retained, out=np.full(repeats, np.nan), where=boot_retained > 0
        )
        participants = []
        for index, (subject, mask) in enumerate(zip(subjects, masks, strict=True)):
            recalls = [
                float(correct[mask & (y == label)].mean()) for label in (0, 1) if (mask & (y == label)).any()
            ]
            participants.append(
                {
                    "subject": subject,
                    "trials": int(counts[index]),
                    "accepted": int(retained[index]),
                    "correct": int(successes[index]),
                    "coverage": float(retained[index] / counts[index]),
                    "accepted_accuracy": float(successes[index] / retained[index])
                    if retained[index]
                    else None,
                    "balanced_accuracy": float(np.mean(recalls)) if len(recalls) == 2 else None,
                }
            )
        contributing = int((retained > 0).sum())
        output.append(
            {
                "threshold": threshold,
                "trials": len(rows),
                "accepted": total_retained,
                "correct": total_correct,
                "coverage": total_retained / len(rows),
                "accepted_accuracy": total_correct / total_retained if total_retained else None,
                "selective_risk": 1 - total_correct / total_retained if total_retained else None,
                "participants_with_commands": contributing,
                "participants_without_commands": len(subjects) - contributing,
                "median_participant_coverage": float(np.median(retained / counts)),
                "coverage_ci95": _interval(boot_retained / bootstrap_n) if len(subjects) >= 2 else None,
                "accepted_accuracy_ci95": _interval(boot_accuracy) if contributing >= 2 else None,
                "accuracy_bootstrap_valid": int(np.isfinite(boot_accuracy).sum()),
                "participants": participants,
            }
        )
    return output


def build_audit(rows, rule=None, repeats=2000, seed=4027):
    """Describe fixed predictions; validation alone selects the operating threshold."""
    if not isinstance(repeats, int) or not 100 <= repeats <= 10000:
        raise ValueError("Use 100–10000 bootstrap repeats")
    rule = validate_rule(DEFAULT_RULE if rule is None else rule)
    groups = validate_predictions(rows)
    report = {
        "schema_version": "1.0",
        "rule": rule,
        "bootstrap": {"unit": "participant", "repeats": repeats, "seed": seed},
        "groups": [],
    }
    for condition, model_seed in sorted({(c, s) for c, s, _ in groups}):
        validation = _points(groups[(condition, model_seed, "validation")], rule["thresholds"], repeats, seed)
        qualifying = [
            p
            for p in validation
            if p["accepted"] >= rule["target_retained"]
            and p["coverage"] >= rule["target_coverage"]
            and p["accepted_accuracy"] >= rule["target_accuracy"]
        ]
        selected = qualifying[0]["threshold"] if qualifying else rule["fallback_threshold"]
        for split in ("validation", "test"):
            points = (
                validation
                if split == "validation"
                else _points(groups[(condition, model_seed, split)], rule["thresholds"], repeats, seed)
            )
            report["groups"].append(
                {
                    "condition": condition,
                    "seed": model_seed,
                    "split": split,
                    "validation_target_met": bool(qualifying),
                    "selected_threshold": selected,
                    "participant_count": len(points[0]["participants"]),
                    "points": points,
                }
            )
    return report
