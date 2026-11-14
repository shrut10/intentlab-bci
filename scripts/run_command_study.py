"""Reproduce the frozen Zhou2016 command replay study without changing deployment."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import shutil
import stat
import subprocess
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intentlab.commands import build_command_audit

PROTOCOL_DIR = ROOT / "experiments" / "command-replay-v1"
DEFAULT_RAW = ROOT / "data" / "raw" / "zhou2016"
DEFAULT_OUTPUT = ROOT / "artifacts" / "command-replay-v1"
PREDICTION_FIELDS = ["participant", "session", "time_s", "left_probability", "right_probability"]
EVENT_FIELDS = ["participant", "session", "event_id", "start_s", "end_s", "state"]
SPLITS = ("train", "validation", "test")


def read_protocol():
    return json.loads((PROTOCOL_DIR / "protocol.json").read_text())


def digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_frozen_protocol(commit):
    """Compare the executed protocol with its already committed version."""
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT, check=True)
    for name in ("protocol.json", "PROTOCOL.md"):
        relative = f"experiments/command-replay-v1/{name}"
        committed = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        if committed != (PROTOCOL_DIR / name).read_bytes():
            raise ValueError(f"Protocol differs from frozen commit: {relative}")


def verify_source_amendment(commit):
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT, check=True)
    for name in ("source_layout.json", "AMENDMENT-01.md"):
        relative = f"experiments/command-replay-v1/{name}"
        committed = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        if committed != (PROTOCOL_DIR / name).read_bytes():
            raise ValueError(f"Source amendment differs from committed version: {relative}")


def expected_members(protocol):
    return [
        f"data/S{person}_{session}{run}.cnt"
        for person in protocol["source"]["subjects"]
        for session in protocol["source"]["sessions"]
        for run in protocol["source"]["runs"]
    ]


def safe_extract(archive, destination, expected):
    """Validate every archive path and extract only the prespecified recordings."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        infos = zipped.infolist()
        names = [entry.filename for entry in infos]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive entries")
        if sum(entry.file_size for entry in infos) > 2_000_000_000:
            raise ValueError("Unexpected archive expansion")
        for entry in infos:
            relative = PurePosixPath(entry.filename)
            mode = entry.external_attr >> 16
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in entry.filename
                or stat.S_ISLNK(mode)
                or not (destination / entry.filename).resolve().is_relative_to(destination)
            ):
                raise ValueError(f"Unsafe archive path: {entry.filename}")
        missing = set(expected) - set(names)
        if missing:
            raise ValueError(f"Missing prespecified recordings: {sorted(missing)}")
        for name in expected:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(name) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    return [destination / name for name in expected]


def acquire_source(protocol, raw_dir, download=False):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    source = protocol["source"]
    archive = raw_dir / source["file"]
    if not archive.exists():
        if not download:
            raise FileNotFoundError(
                "Source absent; pass --download to retrieve the original Figshare archive"
            )
        temporary = archive.with_suffix(".part")
        request = urllib.request.Request(source["download_url"], headers={"User-Agent": "IntentLab/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            if temporary.stat().st_size != source["bytes"] or digest(temporary, "md5") != source["md5"]:
                raise ValueError("Downloaded archive does not match the published size and MD5")
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    if archive.stat().st_size != source["bytes"] or digest(archive, "md5") != source["md5"]:
        raise ValueError("Cached archive does not match the frozen source")
    paths = safe_extract(archive, raw_dir, expected_members(protocol))
    return paths, {
        "url": source["download_url"],
        "metadata_url": source["metadata_url"],
        "doi": source["doi"],
        "license": source["license"],
        "bytes": archive.stat().st_size,
        "md5": digest(archive, "md5"),
        "sha256": digest(archive),
    }


def partition_events(cues, duration, participant, session):
    """Fill the recorded timeline using only guarded protocol supported rest."""
    segments = []
    previous = -float("inf")
    for index, cue in enumerate(cues):
        start, state = float(cue["time_s"]), cue["state"]
        if state not in {"left", "right", "other"} or start < 0 or start < previous + 5:
            raise ValueError("Cues must be valid and ordered with disjoint five second imagery intervals")
        end = start + 5.0
        if end > duration:
            raise ValueError("An imagery cue extends beyond the recording")
        identifier = f"{participant}-{session}-cue{index + 1:03d}"
        segments.append((start, end, state, identifier))
        rest_start, rest_end = start + 5.5, start + 8.5
        next_cue = float(cues[index + 1]["time_s"]) if index + 1 < len(cues) else None
        if rest_end <= duration and (next_cue is None or rest_end <= next_cue - 1.0):
            segments.append((rest_start, rest_end, "idle", identifier + "-rest"))
        previous = start
    events, cursor = [], 0.0
    for start, end, state, identifier in segments:
        if start < cursor:
            raise ValueError("Protocol intervals overlap")
        if start > cursor:
            events.append(dict(start_s=cursor, end_s=start, state="unknown"))
        events.append(dict(start_s=start, end_s=end, state=state, event_id=identifier))
        cursor = end
    if cursor < duration:
        events.append(dict(start_s=cursor, end_s=float(duration), state="unknown"))
    if not events:
        raise ValueError("No events in recording")
    for index, event in enumerate(events):
        event.update(participant=str(participant), session=str(session))
        event.setdefault("event_id", f"{participant}-{session}-unknown{index:03d}")
    return events


def causal_signal(signal, fs=250):
    from scipy.signal import butter, sosfilt

    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[0] != 14 or not np.isfinite(signal).all():
        raise ValueError("Expected finite EEG with 14 channels")
    referenced = signal - signal.mean(axis=0, keepdims=True)
    coefficients = butter(4, [8, 30], btype="bandpass", fs=fs, output="sos")
    return sosfilt(coefficients, referenced, axis=-1)


def trailing_features(filtered, fs=250):
    """Use only samples strictly earlier than each recorded decision time."""
    from scipy.signal import welch

    window, stride = 2 * fs, fs // 2
    ends = np.arange(window, filtered.shape[1], stride, dtype=np.int64)
    features = []
    for offset in range(0, len(ends), 128):
        block = np.stack([filtered[:, end - window : end] for end in ends[offset : offset + 128]])
        frequencies, density = welch(block, fs=fs, nperseg=fs, axis=-1)
        bands = [
            np.log(np.maximum(density[..., (frequencies >= low) & (frequencies < high)].sum(axis=-1), 1e-10))
            for low, high in ((8, 13), (13, 20), (20, 30))
        ]
        features.append(np.stack(bands, axis=-1).reshape(len(block), 42))
    if not features:
        raise ValueError("Recording is shorter than the initial window")
    return np.vstack(features), (ends - window) / fs, ends / fs


def label_complete_windows(starts, ends, events):
    labels = np.full(len(ends), "unknown", dtype="U7")
    for event in events:
        if event["state"] in {"left", "right", "idle", "other"}:
            contained = (starts >= event["start_s"]) & (ends <= event["end_s"])
            labels[contained] = event["state"]
    return labels


def read_recording(path, participant, session_number, run, protocol):
    import mne

    layout = json.loads((PROTOCOL_DIR / "source_layout.json").read_text())
    source_record = next(record for record in layout["records"] if record["file"] == path.name)
    raw = mne.io.read_raw_cnt(
        path,
        preload=False,
        eog=["VEOU", "VEOL"],
        data_format=layout["data_format"],
        recompute_n_samples=layout["recompute_n_samples"],
        verbose="ERROR",
    )
    annotations = dict(Counter(map(str, raw.annotations.description)))
    if raw.n_times != source_record["samples"] or annotations != source_record["counts"]:
        raise ValueError(f"Recording differs from the frozen source layout: {path.name}")
    if raw.info["sfreq"] != 250 or raw.first_samp != 0:
        raise ValueError("Source sampling rate or start sample differs from the frozen protocol")
    normalised = [name.upper() for name in raw.ch_names]
    expected = protocol["signal"]["channels"]
    extra = set(normalised) - set(expected)
    if len(normalised) != len(set(normalised)) or set(expected) - set(normalised):
        raise ValueError(f"Missing or duplicate EEG channels: {normalised}")
    if not extra <= {"VEOU", "VEOL", "M1", "M2"}:
        raise ValueError(f"Unexpected additional channels: {sorted(extra)}")
    indices = [normalised.index(name) for name in expected]
    signal = raw.get_data(picks=indices) * 1e6
    duration = raw.n_times / 250
    cue_codes = protocol["source"]["cue_codes"]
    cues = [
        {"time_s": float(onset), "state": cue_codes[str(description)]}
        for onset, description in zip(raw.annotations.onset, raw.annotations.description, strict=True)
        if str(description) in cue_codes
    ]
    counts = Counter(cue["state"] for cue in cues)
    session = f"{session_number}{run}"
    events = partition_events(cues, duration, f"S{participant}", session)
    features, starts, ends = trailing_features(causal_signal(signal))
    return {
        "participant": f"S{participant}",
        "session": session,
        "session_number": session_number,
        "features": features,
        "times": ends,
        "labels": label_complete_windows(starts, ends, events),
        "events": events,
        "metadata": {
            "file": path.name,
            "sha256": digest(path),
            "participant": f"S{participant}",
            "session": session,
            "session_number": session_number,
            "samples": int(raw.n_times),
            "duration_seconds": duration,
            "sampling_hz": raw.info["sfreq"],
            "source_channels": raw.ch_names,
            "selected_channels": expected,
            "excluded_channels": sorted(extra),
            "annotations": annotations,
            "cue_counts": dict(counts),
            "complete_window_counts": dict(Counter(label_complete_windows(starts, ends, events))),
            "rest_intervals": sum(event["state"] == "idle" for event in events),
            "rest_label_basis": "Protocol inferred, not an explicit recorded rest annotation",
        },
    }


def split_for_session(session_number, protocol):
    matches = [name for name in SPLITS if session_number in protocol["split"][name]]
    if len(matches) != 1:
        raise ValueError("Each session must belong to exactly one split")
    return matches[0]


def fit_models(records, protocol):
    """Fitting cannot read validation or test feature arrays."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models, fit_counts = {}, []
    for person in sorted({record["participant"] for record in records}):
        selected = [
            record
            for record in records
            if record["participant"] == person
            and split_for_session(record["session_number"], protocol) == "train"
        ]
        if not selected:
            raise ValueError(f"No training session for {person}")
        features = np.vstack([record["features"] for record in selected])
        labels = np.concatenate([record["labels"] for record in selected])
        move = np.isin(labels, ["left", "right"])
        gate = np.isin(labels, ["left", "right", "idle"])

        def estimator():
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=2027),
            )

        movement_model = estimator().fit(features[move], labels[move])
        gate_labels = np.where(labels[gate] == "idle", "rest", "task")
        gate_model = estimator().fit(features[gate], gate_labels)
        models[person] = (movement_model, gate_model)
        fit_counts.append(
            {
                "participant": person,
                "training_sessions": [record["session"] for record in selected],
                "movement_windows": int(move.sum()),
                "gate_windows": int(gate.sum()),
                "label_counts": dict(Counter(labels[gate])),
            }
        )
    return models, fit_counts


def predict_records(models, records, protocol):
    predictions = {
        condition: {split: [] for split in ("validation", "test")}
        for condition in protocol["model"]["conditions"]
    }
    for record in records:
        split = split_for_session(record["session_number"], protocol)
        if split == "train":
            continue
        movement, gate = models[record["participant"]]
        probabilities = movement.predict_proba(record["features"])
        indices = [list(movement.classes_).index(label) for label in ("left", "right")]
        probabilities = probabilities[:, indices]
        task = gate.predict_proba(record["features"])[:, list(gate.classes_).index("task")]
        for condition in predictions:
            scores = probabilities if condition == "movement_only" else probabilities * task[:, None]
            for time, score in zip(record["times"], scores, strict=True):
                predictions[condition][split].append(
                    {
                        "participant": record["participant"],
                        "session": record["session"],
                        "time_s": float(time),
                        "left_probability": float(score[0]),
                        "right_probability": float(score[1]),
                    }
                )
    return predictions


def policy_grid(protocol):
    settings = protocol["policy"]
    return [
        dict(
            enter_threshold=enter,
            hold_seconds=hold,
            release_threshold=settings["release_threshold"],
            cooldown_seconds=settings["cooldown_seconds"],
            max_gap_seconds=settings["max_gap_seconds"],
        )
        for enter in settings["enter_thresholds"]
        for hold in settings["hold_seconds"]
    ]


def choose_policy(predictions, events, protocol):
    candidates = []
    for policy in policy_grid(protocol):
        summary = build_command_audit(predictions, events, policy)["summary"]
        fraction = summary["cues_first_command_correct"] / summary["cues"] if summary["cues"] else 0.0
        candidates.append(dict(policy=policy, summary=summary, first_command_correct_fraction=fraction))
    target = protocol["policy"]["validation_target"]
    acceptable = [
        candidate
        for candidate in candidates
        if candidate["summary"]["idle_commands_per_minute"] is not None
        and candidate["summary"]["idle_commands_per_minute"] <= target["max_idle_commands_per_minute"]
        and candidate["first_command_correct_fraction"] >= target["min_cues_first_command_correct_fraction"]
    ]
    if acceptable:
        selected = max(
            acceptable,
            key=lambda candidate: (
                candidate["first_command_correct_fraction"],
                -candidate["summary"]["idle_commands_per_minute"],
                candidate["policy"]["enter_threshold"],
                candidate["policy"]["hold_seconds"],
            ),
        )
        return selected["policy"], True, candidates
    fallback = protocol["policy"]["fallback"]
    selected = next(
        candidate
        for candidate in candidates
        if all(candidate["policy"][key] == value for key, value in fallback.items())
    )
    return selected["policy"], False, candidates


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_study(
    protocol_commit, source_layout_commit, download=False, raw_dir=DEFAULT_RAW, output_dir=DEFAULT_OUTPUT
):
    verify_frozen_protocol(protocol_commit)
    verify_source_amendment(source_layout_commit)
    protocol = read_protocol()
    output_dir = Path(output_dir)
    if (output_dir / "evaluation.json").exists():
        raise FileExistsError(
            "Evaluation already exists; use a separate output directory for exact reproduction"
        )
    paths, source = acquire_source(protocol, raw_dir, download)
    records = []
    for path in paths:
        person = int(path.stem[1])
        session, run = int(path.stem[3]), path.stem[4]
        print(f"Reading S{person} session {session} run {run}", flush=True)
        records.append(read_recording(path, person, session, run, protocol))
    models, fit_counts = fit_models(records, protocol)
    predictions = predict_records(models, records, protocol)
    events = {
        split: [
            event
            for record in records
            if split_for_session(record["session_number"], protocol) == split
            for event in record["events"]
        ]
        for split in SPLITS
    }
    conditions = []
    for condition in protocol["model"]["conditions"]:
        policy, target_met, candidates = choose_policy(
            predictions[condition]["validation"], events["validation"], protocol
        )
        conditions.append(
            {
                "condition": condition,
                "policy": policy,
                "validation_target_met": target_met,
                "validation_candidates": candidates,
                "validation": build_command_audit(
                    predictions[condition]["validation"], events["validation"], policy
                ),
                "test": build_command_audit(predictions[condition]["test"], events["test"], policy),
            }
        )
    evaluation = {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "protocol_commit": protocol_commit,
        "source_layout_commit": source_layout_commit,
        "production_model_changed": False,
        "dataset": protocol["source"]["dataset"],
        "split": protocol["split"],
        "participants": 4,
        "training": fit_counts,
        "conditions": conditions,
        "interpretation": protocol["evaluation"]["interpretation"],
        "limitations": [
            "Four experienced healthy participants; no population estimate or unseen participant claim",
            "Recorded replay with protocol inferred rest, not a live asynchronous task",
            "Same people across sessions; this study does not evaluate the deployed PhysioNet CNN",
            "Scores from the rest gate are decision scores, not calibrated joint probabilities",
            "Idle time includes only complete guarded intervals, and other activity includes feet imagery",
            "Feature windows overlap; windows are not independent observations",
            "Original annotation counts differ from the paper: S1 session 1 has 30 per class except 29 right in run B; S2 session 2 run B and S4 session 1 run A have 20 per class; S3 session 2 run B has 26 feet cues",
            "Legacy CNT sample counts were reconstructed from file byte extents using the committed source reader amendment before fitting",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "evaluation.json", evaluation)
    for split in SPLITS:
        write_csv(output_dir / f"events-{split}.csv", events[split], EVENT_FIELDS)
    for condition, by_split in predictions.items():
        for split, rows in by_split.items():
            write_csv(output_dir / f"predictions-{condition}-{split}.csv", rows, PREDICTION_FIELDS)
    # The public demo needs audited commands and events, not every score row.
    demo = {key: value for key, value in evaluation.items() if key not in {"training", "conditions"}}
    demo["conditions"] = [
        {key: value for key, value in condition.items() if key != "validation_candidates"}
        for condition in conditions
    ]
    write_json(output_dir / "demo.json", demo)
    code_paths = [
        Path(__file__),
        ROOT / "src/intentlab/commands.py",
        PROTOCOL_DIR / "protocol.json",
        PROTOCOL_DIR / "PROTOCOL.md",
        PROTOCOL_DIR / "AMENDMENT-01.md",
        PROTOCOL_DIR / "source_layout.json",
    ]
    output_names = ["evaluation.json", "demo.json", *[f"events-{split}.csv" for split in SPLITS]]
    output_names += [
        f"predictions-{condition}-{split}.csv"
        for condition in predictions
        for split in ("validation", "test")
    ]
    provenance = {
        "source": source,
        "source_records": [record["metadata"] for record in records],
        "frozen_protocol_commit": protocol_commit,
        "source_layout_commit": source_layout_commit,
        "repository_commit_at_run": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "code_sha256": {str(path.relative_to(ROOT)): digest(path) for path in code_paths},
        "packages": {
            name: importlib.metadata.version(name) for name in ("numpy", "scipy", "mne", "scikit-learn")
        },
        "python": sys.version,
        "output_sha256": {name: digest(output_dir / name) for name in sorted(output_names)},
        "raw_data_redistributed": False,
    }
    write_json(output_dir / "provenance.json", provenance)
    for condition in conditions:
        print(
            json.dumps(
                {
                    "condition": condition["condition"],
                    "policy": condition["policy"],
                    "validation_target_met": condition["validation_target_met"],
                    "test": condition["test"]["summary"],
                }
            ),
            flush=True,
        )
    return evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-commit", required=True)
    parser.add_argument("--source-layout-commit", help="Commit recording the source reader amendment")
    parser.add_argument(
        "--download", action="store_true", help="Retrieve the original 156 MB Figshare archive"
    )
    parser.add_argument(
        "--download-only", action="store_true", help="Verify and extract source without fitting"
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    verify_frozen_protocol(args.protocol_commit)
    if args.download_only:
        _, provenance = acquire_source(read_protocol(), args.raw_dir, args.download)
        print(json.dumps(provenance, indent=2))
    else:
        if not args.source_layout_commit:
            parser.error("--source-layout-commit is required before fitting")
        run_study(
            args.protocol_commit, args.source_layout_commit, args.download, args.raw_dir, args.output_dir
        )


if __name__ == "__main__":
    main()
