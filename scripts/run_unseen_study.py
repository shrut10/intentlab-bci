"""Run the frozen Dreyer study with a committed barrier before final evaluation."""

import argparse
import csv
import importlib.metadata
import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from numbers import Integral
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dreyer_source import acquire_subject, file_hash, load_manifest, load_subject

from intentlab.unseen import (
    MODEL_NAMES,
    bootstrap_comparison,
    choose_model,
    evaluate_predictions,
    extract_features,
    fit_models,
    predict_models,
    select_threshold,
)

PROTOCOL_DIR = ROOT / "experiments/unseen-v1"
OUTPUT = ROOT / "artifacts/unseen-v1"
CACHE = ROOT / "data/derived/unseen-v1"
RAW = ROOT / "data/raw/dreyer2023"
FROZEN_FILES = ("protocol.json", "PROTOCOL.md", "source_manifest.json", "split_manifest.json")
CODE_FILES = (
    "scripts/run_unseen_study.py",
    "scripts/dreyer_source.py",
    "src/intentlab/unseen.py",
    "requirements-unseen-study.txt",
    "requirements-command-study.txt",
    "requirements-dev.txt",
    "requirements.txt",
)
PREDICTION_FIELDS = (
    "participant",
    "cohort",
    "run",
    "trial_id",
    "label",
    "model",
    "left_probability",
    "right_probability",
    "threshold",
    "accepted",
)
SOURCE_METADATA_FIELDS = ("trials", "recordings", "archive", "dropped_trials")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_committed(commit, paths):
    """Require the named commit to precede HEAD and match every current byte."""
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT, check=True)
    for path in paths:
        path = Path(path)
        relative = path.relative_to(ROOT).as_posix()
        frozen = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        if frozen != path.read_bytes():
            raise ValueError(f"File differs from its committed seal: {relative}")


def source_hashes():
    return {name: file_hash(ROOT / name) for name in CODE_FILES}


def packages():
    return {
        name: importlib.metadata.version(name)
        for name in ("numpy", "scipy", "scikit-learn", "mne", "pyriemann", "joblib")
    }


def verify_runtime(expected_code, expected_packages):
    """Stop if source or dependencies change while a long study phase is running."""
    if source_hashes() != expected_code or packages() != expected_packages:
        raise ValueError("Source or packages changed during execution; the study phase must stop")


def read_splits(protocol):
    records = json.loads((PROTOCOL_DIR / "split_manifest.json").read_text())["participants"]
    if [record["source_id"] for record in records] != list(range(1, 88)):
        raise ValueError("All 87 source participants must appear once in order")
    if len({record["participant"] for record in records}) != 87:
        raise ValueError("Participant identifiers must be unique")
    if dict(Counter(record["split"] for record in records)) != protocol["partitions"]:
        raise ValueError("Participant counts differ from the frozen split")
    for cohort, counts in protocol["cohort_partitions"].items():
        actual = [
            sum(record["cohort"] == cohort and record["split"] == split for record in records)
            for split in ("train", "validation", "test")
        ]
        if actual != counts:
            raise ValueError("Participant cohort counts differ from the frozen split")
    for record in records:
        subject = record["source_id"]
        expected_cohort = "A" if subject <= 60 else "B" if subject <= 81 else "C"
        if record["participant"] != f"S{subject:02d}" or record["cohort"] != expected_cohort:
            raise ValueError("Participant identity or cohort differs from the source")
    return records


def stage_records(records, stage):
    if stage not in {"fit", "test"}:
        raise ValueError("Stage must be fit or test")
    allowed = {"train", "validation"} if stage == "fit" else {"test"}
    return [record for record in records if record["split"] in allowed]


def validate_trials(trials, record):
    if not trials or len({trial["trial_id"] for trial in trials}) != len(trials):
        raise ValueError("Trial identifiers must be nonempty and unique")
    for trial in trials:
        identifier = trial["trial_id"]
        if (
            not isinstance(identifier, str)
            or not 1 <= len(identifier) <= 128
            or identifier != identifier.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in identifier)
        ):
            raise ValueError("Trial identifiers must be bounded nonempty strings without control characters")
        label = trial["label"]
        if isinstance(label, bool) or not isinstance(label, Integral) or label not in {0, 1}:
            raise ValueError("Trial labels must be integer 0 left or 1 right")
        if trial.get("cue_code", str(769 + label)) != str(769 + label):
            raise ValueError("Trial cue code disagrees with its class label")
        if trial.get("label_name", ("left", "right")[label]) != ("left", "right")[label]:
            raise ValueError("Trial label name disagrees with its class label")
        if any(trial[key] != record[key] for key in ("participant", "cohort")):
            raise ValueError("Trial identity differs from its assigned participant")
        if trial["subject"] != record["source_id"] or trial["run"] not in {"R1acquisition", "R2acquisition"}:
            raise ValueError("Trial source or run differs from its assigned participant")
    if {trial["label"] for trial in trials} != {0, 1}:
        raise ValueError("Both classes are required for every participant")
    if {trial["run"] for trial in trials} != {"R1acquisition", "R2acquisition"}:
        raise ValueError("Both acquisition runs are required for every participant")
    for run in ("R1acquisition", "R2acquisition"):
        if {trial["label"] for trial in trials if trial["run"] == run} != {0, 1}:
            raise ValueError("Both classes are required within each acquisition run")


def canonical_source_metadata(metadata):
    return json.dumps(
        {key: metadata[key] for key in SOURCE_METADATA_FIELDS},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def load_features(record, protocol, manifest, raw_dir, cache_dir, download):
    """Cache only deterministic features, bound to source and executed code hashes."""
    participant = record["participant"]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    array_path, metadata_path = cache_dir / f"{participant}.npz", cache_dir / f"{participant}.json"
    source = next(item for item in manifest["participants"] if item["subject"] == record["source_id"])
    signature = {
        "archive_sha256": source["sha256"],
        "protocol_sha256": file_hash(PROTOCOL_DIR / "protocol.json"),
        "code_sha256": source_hashes(),
        "packages": packages(),
    }
    if array_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["signature"] != signature or metadata["feature_sha256"] != file_hash(array_path):
            raise ValueError(f"Feature cache changed for {participant}; investigate before continuing")
        with np.load(array_path, allow_pickle=False) as arrays:
            if str(arrays["source_metadata_json"].item()) != canonical_source_metadata(metadata):
                raise ValueError(f"Feature cache source metadata changed for {participant}")
            features = {name: arrays[name] for name in ("bandpower", "covariance")}
    else:
        archive = acquire_subject(record["source_id"], raw_dir, manifest, download=download)
        source_data = load_subject(record["source_id"], archive, protocol)
        values = source_data["epochs"]
        values = values - values.mean(axis=1, keepdims=True)
        indices = [source_data["channels"].index(channel) for channel in protocol["channels"]]
        features = extract_features(values[:, indices], protocol["sampling_hz"])
        metadata = {
            "signature": signature,
            "trials": source_data["trials"],
            "recordings": source_data["recordings"],
            "archive": source_data["archive"],
            "dropped_trials": source_data["dropped_trials"],
        }
        validate_trials(metadata["trials"], record)
        np.savez_compressed(array_path, source_metadata_json=canonical_source_metadata(metadata), **features)
        metadata["feature_sha256"] = file_hash(array_path)
        write_json(metadata_path, metadata)
    validate_trials(metadata["trials"], record)
    if any(len(values) != len(metadata["trials"]) for values in features.values()):
        raise ValueError("Feature and trial counts differ")
    return features, metadata


def collect(records, protocol, raw_dir, cache_dir, download):
    manifest = load_manifest()
    feature_blocks, trials, source = [], [], []
    for index, record in enumerate(records, 1):
        print(f"Reading {record['participant']} ({record['split']}) {index}/{len(records)}", flush=True)
        features, metadata = load_features(record, protocol, manifest, raw_dir, cache_dir, download)
        feature_blocks.append(features)
        trials.extend(metadata["trials"])
        source.append({key: metadata[key] for key in ("archive", "recordings", "dropped_trials")})
    features = {
        name: np.concatenate([block[name] for block in feature_blocks])
        for name in ("bandpower", "covariance")
    }
    if len({trial["trial_id"] for trial in trials}) != len(trials):
        raise ValueError("Duplicate trials across participants")
    return features, trials, source


def arrays(trials):
    return (
        np.asarray([trial["label"] for trial in trials]),
        np.asarray([trial["participant"] for trial in trials]),
        np.asarray([trial["cohort"] for trial in trials]),
    )


def split_data(features, trials, people):
    mask = np.asarray([trial["participant"] in people for trial in trials])
    return {name: value[mask] for name, value in features.items()}, [
        trial for trial, keep in zip(trials, mask, strict=True) if keep
    ]


def write_predictions(path, trials, predictions, thresholds):
    with Path(path).open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for name in MODEL_NAMES:
            threshold = thresholds[name]["threshold"]
            for trial, probabilities in zip(trials, predictions[name], strict=True):
                writer.writerow(
                    {
                        **{key: trial[key] for key in ("participant", "cohort", "run", "trial_id", "label")},
                        "model": name,
                        "left_probability": float(probabilities[0]),
                        "right_probability": float(probabilities[1]),
                        "threshold": threshold,
                        "accepted": int(max(probabilities) >= threshold),
                    }
                )


def fit_stage(protocol, records, args):
    selection_path = args.output / "selection.json"
    model_path = args.cache / "fitted_models.joblib"
    if selection_path.exists() or model_path.exists():
        raise ValueError("Existing fitted study found; refusing to overwrite the sealed experiment")
    executed_code, executed_packages = source_hashes(), packages()
    features, trials, source = collect(
        stage_records(records, "fit"), protocol, args.raw, args.cache, args.download
    )
    train_people = {record["participant"] for record in records if record["split"] == "train"}
    validation_people = {record["participant"] for record in records if record["split"] == "validation"}
    train_features, train_trials = split_data(features, trials, train_people)
    val_features, val_trials = split_data(features, trials, validation_people)
    train_y, train_p, _ = arrays(train_trials)
    val_y, val_p, val_c = arrays(val_trials)
    verify_runtime(executed_code, executed_packages)
    print("Fitting the two frozen models on training participants only", flush=True)
    models = fit_models(train_features, train_y, train_p)
    predictions = predict_models(models, val_features)
    thresholds = {name: select_threshold(val_y, predictions[name], val_p, val_c) for name in MODEL_NAMES}
    summaries = {
        name: evaluate_predictions(val_y, predictions[name], val_p, val_c, thresholds[name]["threshold"])
        for name in MODEL_NAMES
    }
    verify_runtime(executed_code, executed_packages)
    joblib.dump(models, model_path, compress=3)
    write_predictions(args.output / "validation_predictions.csv", val_trials, predictions, thresholds)
    selection = {
        "experiment": "unseen-v1",
        "dataset": "Dreyer2023",
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol_commit": args.protocol_commit,
        "code_sha256": executed_code,
        "packages": executed_packages,
        "model_sha256": file_hash(model_path),
        "validation_predictions_sha256": file_hash(args.output / "validation_predictions.csv"),
        "selection": {"model": choose_model(summaries), "rule": protocol["models"]["selection"]},
        "models": {
            name: {"validation": summaries[name], "threshold_selection": thresholds[name]}
            for name in MODEL_NAMES
        },
        "counts": {
            "train": {"participants": len(train_people), "trials": len(train_trials)},
            "validation": {"participants": len(validation_people), "trials": len(val_trials)},
        },
        "source": source,
        "test_participants_accessed": False,
    }
    write_json(selection_path, selection)
    print(json.dumps({"selection": selection["selection"], "counts": selection["counts"]}), flush=True)
    print("Commit selection.json and validation_predictions.csv before running the final stage", flush=True)


def verify_selection(args):
    if not args.selection_commit:
        raise ValueError("Final evaluation requires the committed validation selection")
    selection_path = args.output / "selection.json"
    verify_committed(args.selection_commit, [selection_path, args.output / "validation_predictions.csv"])
    selection = json.loads(selection_path.read_text())
    if selection["protocol_commit"] != args.protocol_commit:
        raise ValueError("Selection refers to a different protocol commit")
    if selection["code_sha256"] != source_hashes() or selection["packages"] != packages():
        raise ValueError("Executed code or packages differ from the sealed fit")
    if selection["model_sha256"] != file_hash(args.cache / "fitted_models.joblib"):
        raise ValueError("Fitted models differ from the committed selection")
    if selection["validation_predictions_sha256"] != file_hash(args.output / "validation_predictions.csv"):
        raise ValueError("Validation predictions changed after fitting")
    if selection["test_participants_accessed"] is not False:
        raise ValueError("Selection must be fixed before final participant access")
    return selection


def test_stage(protocol, records, args):
    if (args.output / "evaluation.json").exists():
        raise ValueError("Final results already exist; refusing to overwrite the published experiment")
    # This barrier runs before any test recording or cached test feature is read.
    selection = verify_selection(args)
    features, trials, source = collect(
        stage_records(records, "test"), protocol, args.raw, args.cache, args.download
    )
    # Downloading every participant can take time; verify the sealed inputs again
    # before model deserialisation or production of any final predictions.
    verify_selection(args)
    models = joblib.load(args.cache / "fitted_models.joblib")
    predictions = predict_models(models, features)
    labels, people, cohorts = arrays(trials)
    thresholds = {name: selection["models"][name]["threshold_selection"] for name in MODEL_NAMES}
    summaries = {
        name: evaluate_predictions(labels, predictions[name], people, cohorts, thresholds[name]["threshold"])
        for name in MODEL_NAMES
    }
    bootstrap = bootstrap_comparison(labels, predictions, people, cohorts)
    verify_selection(args)
    write_predictions(args.output / "test_predictions.csv", trials, predictions, thresholds)
    evaluation = {
        "schema_version": "1.0",
        "kind": "unseen_participants",
        "experiment": "unseen-v1",
        "dataset": "Dreyer2023",
        "production_model_changed": False,
        "completed_utc": datetime.now(UTC).isoformat(),
        "protocol_commit": args.protocol_commit,
        "selection_commit": args.selection_commit,
        "counts": {**selection["counts"], "test": {"participants": len(set(people)), "trials": len(trials)}},
        "splits": {
            split: [r["participant"] for r in records if r["split"] == split]
            for split in protocol["partitions"]
        },
        "models": {name: {**selection["models"][name], "test": summaries[name]} for name in MODEL_NAMES},
        "selection": {
            **selection["selection"],
            "primary_criterion_met": bootstrap["model_intervals"][selection["selection"]["model"]][0] > 0.5,
        },
        "test_bootstrap": bootstrap,
        "limitations": protocol["limitations"],
    }
    write_json(args.output / "evaluation.json", evaluation)
    participant_rows = [
        {"model": name, **person} for name in MODEL_NAMES for person in summaries[name]["participants"]
    ]
    with (args.output / "participant_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(participant_rows[0]))
        writer.writeheader()
        writer.writerows(participant_rows)
    provenance = {
        "experiment": "unseen-v1",
        "protocol_commit": args.protocol_commit,
        "selection_commit": args.selection_commit,
        "code_sha256": source_hashes(),
        "packages": packages(),
        "fitted_model_sha256": selection["model_sha256"],
        "production_model_changed": False,
        "raw_data_redistributed": False,
        "source_license": "CC BY 4.0",
        "source_doi": "10.5281/zenodo.8089820",
        "source": selection["source"] + source,
        "protocol_sha256": {name: file_hash(PROTOCOL_DIR / name) for name in FROZEN_FILES},
        "output_sha256": {
            name: file_hash(args.output / name)
            for name in (
                "selection.json",
                "validation_predictions.csv",
                "test_predictions.csv",
                "participant_metrics.csv",
                "evaluation.json",
            )
        },
    }
    write_json(args.output / "provenance.json", provenance)
    print(
        json.dumps(
            {"selection": evaluation["selection"], "counts": evaluation["counts"], "bootstrap": bootstrap}
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fit", "test"), required=True)
    parser.add_argument("--protocol-commit", required=True)
    parser.add_argument("--selection-commit")
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    verify_committed(args.protocol_commit, [PROTOCOL_DIR / name for name in FROZEN_FILES])
    protocol = json.loads((PROTOCOL_DIR / "protocol.json").read_text())
    records = read_splits(protocol)
    if args.stage == "fit":
        fit_stage(protocol, records, args)
    else:
        test_stage(protocol, records, args)


if __name__ == "__main__":
    main()
