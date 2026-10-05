"""Download verified open EEG, audit eligibility, and produce subject-disjoint epochs.

Usage: python scripts/prepare_data.py [--offline]
Raw recordings remain in ignored data/raw; only licensed derivatives are published.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

import mne
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from intentlab.signal import CHANNELS, SFREQ, SAMPLES, bandpower, preprocess  # noqa: E402

SOURCE = "https://physionet.org/files/eegmmidb/1.0.0"
MIRROR = "https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0"
RUNS = (4, 8, 12)


def subject_splits():
    order = np.random.default_rng(2027).permutation(np.arange(1, 110))
    return {
        "train": sorted(order[:65].tolist()),
        "validation": sorted(order[65:87].tolist()),
        "test": sorted(order[87:].tolist()),
    }


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def acquire(rel, expected, offline):
    path = ROOT / "data/raw" / Path(rel).name
    if path.exists() and sha256(path) == expected:
        return path
    if offline:
        raise RuntimeError(f"Missing or invalid cached file: {rel}")
    for attempt in range(4):
        try:
            request = urllib.request.Request(f"{MIRROR}/{rel}", headers={"User-Agent": "IntentLab-research/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Publisher checksum mismatch: {rel}")
            path.write_bytes(content)
            return path
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    for folder in ["data/raw", "data/derived", "data/demo", "artifacts"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    checksum_file = ROOT / "data/raw/SHA256SUMS.txt"
    if not checksum_file.exists():
        if args.offline:
            raise RuntimeError("Publisher checksum manifest is required")
        with urllib.request.urlopen(f"{MIRROR}/SHA256SUMS.txt", timeout=90) as response:
            checksum_file.write_bytes(response.read())
    hashes = {}
    for line in checksum_file.read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        hashes[rel.lstrip("*./")] = digest
    records = [f"S{s:03d}/S{s:03d}R{r:02d}.edf" for s in range(1, 110) for r in RUNS]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(acquire, rel, hashes[rel], args.offline): rel for rel in records}
        for i, future in enumerate(as_completed(futures), 1):
            future.result()  # A failed acquisition aborts; it is not a silent quality exclusion.
            if i % 30 == 0 or i == len(records):
                print(f"Verified {i}/{len(records)} recordings", flush=True)

    splits = subject_splits()
    split_for = {subject: split for split, subjects in splits.items() for subject in subjects}
    epochs, rows, exclusions, manifest = [], [], [], []
    for rel in records:
        subject = int(rel[1:4])
        run = int(Path(rel).stem[-2:])
        path = ROOT / "data/raw" / Path(rel).name
        entry = {"file": rel, "url": f"{SOURCE}/{rel}", "download_url": f"{MIRROR}/{rel}", "sha256": hashes[rel], "bytes": path.stat().st_size}
        manifest.append(entry)
        try:
            raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        except (ValueError, OSError) as exc:
            exclusions.append({"file": rel, "reason": "unparseable", "detail": str(exc)})
            continue
        if raw.info["sfreq"] != SFREQ:
            exclusions.append({"file": rel, "reason": "sampling_frequency", "value": raw.info["sfreq"]})
            continue
        mapping = {name.strip(".").upper(): i for i, name in enumerate(raw.ch_names)}
        if len(raw.ch_names) != 64 or any(ch.upper() not in mapping for ch in CHANNELS):
            exclusions.append({"file": rel, "reason": "channel_schema"})
            continue
        signal = raw.get_data() * 1e6
        signal -= signal.mean(axis=0, keepdims=True)
        signal = signal[[mapping[ch.upper()] for ch in CHANNELS]]
        for event, annotation in enumerate(raw.annotations):
            if annotation["description"] not in ("T1", "T2"):
                continue
            identifier = f"S{subject:03d}-R{run:02d}-E{event:02d}"
            start = round((annotation["onset"] + 1) * SFREQ)
            stop = start + SAMPLES
            x = signal[:, start:stop]
            reason = None
            if annotation["duration"] < 4 or stop > signal.shape[-1] or x.shape[-1] != SAMPLES:
                reason = "incomplete_labelled_epoch"
            elif not np.isfinite(x).all():
                reason = "nonfinite"
            elif (x.std(-1) < 0.01).any():
                reason = "flat_channel"
            elif (np.ptp(x, axis=-1) > 1000).any():
                reason = "amplitude_over_1000_uV"
            if reason:
                exclusions.append({"file": rel, "trial": identifier, "reason": reason})
                continue
            epochs.append(x.astype(np.float32))
            rows.append({"id": identifier, "subject": subject, "run": run, "event": event,
                         "label": 0 if annotation["description"] == "T1" else 1,
                         "split": split_for[subject], "cue_seconds": float(annotation["onset"]),
                         "source_url": entry["url"], "peak_to_peak_uV": float(np.ptp(x, axis=-1).max())})
    x = preprocess(np.stack(epochs))
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    subjects = np.array([r["subject"] for r in rows], dtype=np.int64)
    split = np.array([r["split"] for r in rows])
    features = bandpower(x)
    feature_rows = [dict(row, **{f"{ch}_{band}_logpower": float(features[i, j * 3 + b])
                               for j, ch in enumerate(CHANNELS) for b, band in enumerate(["alpha", "low_beta", "high_beta"])})
                    for i, row in enumerate(rows)]
    pq.write_table(pa.Table.from_pylist(feature_rows), ROOT / "data/derived/trials.parquet", compression="zstd")
    np.savez_compressed(ROOT / "data/derived/epochs.npz", x=x, y=y, subjects=subjects, split=split)
    demo_indices = []
    for subject in splits["test"]:
        demo_indices.extend(np.flatnonzero(subjects == subject)[:6].tolist())
    np.savez_compressed(ROOT / "data/demo/epochs.npz", x=x[demo_indices])
    demo_rows = [rows[i] for i in demo_indices]
    (ROOT / "data/demo/trials.json").write_text(json.dumps(demo_rows, indent=2) + "\n")
    audit = {"dataset": "PhysioNet EEGMMIDB", "version": "1.0.0", "doi": "10.13026/C28G6P",
             "license": "ODC-BY-1.0", "checksum_manifest_sha256": sha256(checksum_file),
             "seed": 2027, "splits": splits, "channels": CHANNELS, "sampling_hz": SFREQ,
             "epoch_seconds": [1, 4], "records": manifest, "exclusions": exclusions,
             "counts": {name: {"subjects": len(np.unique(subjects[split == name])), "trials": int((split == name).sum()),
                               "left": int(((split == name) & (y == 0)).sum()), "right": int(((split == name) & (y == 1)).sum())}
                        for name in splits}, "demo_trials": len(demo_indices)}
    (ROOT / "data/derived/manifest.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit["counts"], indent=2), flush=True)
    print(f"{len(exclusions)} quality exclusions; {len(demo_indices)} deterministic demo trials", flush=True)


if __name__ == "__main__":
    main()
