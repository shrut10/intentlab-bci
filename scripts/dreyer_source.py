"""Verified acquisition and conservative reading of the Dreyer2023 source mirror."""

import hashlib
import json
import shutil
import stat
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "experiments/unseen-v1/source_manifest.json"
HF_REPOSITORY = "EEGDash/eeg-mi-datasets"
HF_PREFIX = "Dreyer2023/MNE-dreyer2023-data"
OSF_FOLDER_URL = (
    "https://api.osf.io/v2/nodes/8tdk5/files/osfstorage/67be5c8fdef94c1d71556d19/?page%5Bsize%5D=100"
)
EEG_CHANNELS = [
    "Fz",
    "FCz",
    "Cz",
    "CPz",
    "Pz",
    "C1",
    "C3",
    "C5",
    "C2",
    "C4",
    "C6",
    "F4",
    "FC2",
    "FC4",
    "FC6",
    "CP2",
    "CP4",
    "CP6",
    "P4",
    "F3",
    "FC1",
    "FC3",
    "FC5",
    "CP1",
    "CP3",
    "CP5",
    "P3",
]
AUXILIARY_CHANNELS = {"EOG1", "EOG2", "EOG3", "EMGg", "EMGd"}
RUNS = ("R1acquisition", "R2acquisition")


def cohort(subject):
    if isinstance(subject, bool) or not isinstance(subject, int) or not 1 <= subject <= 87:
        raise ValueError("Dreyer participant must be an integer from 1 to 87")
    return "A" if subject <= 60 else "B" if subject <= 81 else "C"


def file_hash(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _metadata(url):
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "IntentLab/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def build_source_manifest():
    """Fetch metadata only, matching mirror SHA256 against original OSF hashes."""
    revision = _metadata(f"https://huggingface.co/api/datasets/{HF_REPOSITORY}")["sha"]
    tree_url = (
        f"https://huggingface.co/api/datasets/{HF_REPOSITORY}/tree/{revision}/Dreyer2023"
        "?recursive=true&expand=false&limit=1000"
    )
    tree = _metadata(tree_url)
    mirrors = {Path(entry["path"]).name: entry for entry in tree if entry["type"] == "file"}
    originals = _metadata(OSF_FOLDER_URL)
    if originals.get("links", {}).get("next"):
        raise ValueError("Unexpected paginated OSF metadata; resolve all pages before freezing")
    originals = {entry["attributes"]["name"]: entry for entry in originals["data"]}
    participants = []
    for subject in range(1, 88):
        name = f"sub-{subject:02d}.zip"
        mirror, original = mirrors[name], originals[name]
        attributes = original["attributes"]
        hashes = attributes["extra"]["hashes"]
        if mirror["size"] != attributes["size"] or mirror["lfs"]["oid"] != hashes["sha256"]:
            raise ValueError(f"Original and mirror disagree for {name}")
        participants.append(
            {
                "subject": subject,
                "participant": f"S{subject:02d}",
                "cohort": cohort(subject),
                "filename": name,
                "bytes": mirror["size"],
                "sha256": hashes["sha256"],
                "md5": hashes["md5"],
                "original_osf_id": original["id"],
                "original_version": attributes["current_version"],
                "original_metadata_url": original["links"]["info"],
                "original_download_url": original["links"]["download"],
                "mirror_path": mirror["path"],
                "mirror_download_url": f"https://huggingface.co/datasets/{HF_REPOSITORY}/resolve/{revision}/{mirror['path']}",
                "mirror_sha256": mirror["lfs"]["oid"],
            }
        )
    return {
        "dataset": "Dreyer2023",
        "metadata_date": "2026-11-14",
        "participants": participants,
        "participant_count": 87,
        "archive_total_bytes": sum(record["bytes"] for record in participants),
        "license": "CC BY 4.0",
        "license_metadata_url": "https://zenodo.org/api/records/8089820",
        "paper_doi": "10.1038/s41597-023-02445-z",
        "source_doi": "10.5281/zenodo.8089820",
        "original_archive_bytes": 27475466178,
        "original_archive_md5": "c8c6f5f09f1882666eec10e29155fdcb",
        "osf_folder_metadata_url": OSF_FOLDER_URL,
        "huggingface_repository": HF_REPOSITORY,
        "huggingface_revision": revision,
        "huggingface_tree_metadata_url": tree_url,
        "verification": "All 87 mirror sizes and SHA256 hashes match independently fetched original OSF publisher metadata",
        "channels": EEG_CHANNELS,
        "auxiliary_channels": sorted(AUXILIARY_CHANNELS),
        "sampling_hz": 512,
        "cohort_ranges": {"A": [1, 60], "B": [61, 81], "C": [82, 87]},
        "selected_runs": list(RUNS),
        "event_codes": {"769": "left", "770": "right"},
        "source_conversion": "MOABB OSF EDF conversion mirrored unchanged on Hugging Face; not the original Zenodo GDF archive",
        "caveats": [
            "The first two acquisition runs include sham visual feedback, not an absence of feedback",
            "Source description gives 40 trials per run; retain every valid recorded hand cue and report actual counts",
            "Participants 40 and 59 have been reported to have missing trials or runs; stop on invalid source data rather than excluding after outcomes",
            "Visual cue responses can confound classification; this is cued motor imagery, not spontaneous intent decoding",
            "The source channels.tsv labels EOG and EMG channels as EEG; select the 27 named EEG channels explicitly",
        ],
    }


def load_manifest(path=MANIFEST_PATH):
    manifest = json.loads(Path(path).read_text())
    subjects = [record["subject"] for record in manifest["participants"]]
    if subjects != list(range(1, 88)):
        raise ValueError("Manifest must contain all 87 participants once in order")
    return manifest


def verify_archive(path, record):
    path = Path(path)
    if path.stat().st_size != record["bytes"]:
        raise ValueError(f"Archive size mismatch: {record['filename']}")
    if file_hash(path) != record["sha256"] or file_hash(path, "md5") != record["md5"]:
        raise ValueError(f"Archive checksum mismatch: {record['filename']}")


def acquire_subject(subject, directory, manifest=None, download=False):
    """Download only on an explicit flag; retain verified archives for reproducibility."""
    cohort(subject)
    manifest = load_manifest() if manifest is None else manifest
    record = next(record for record in manifest["participants"] if record["subject"] == subject)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / record["filename"]
    if target.exists():
        verify_archive(target, record)
        return target
    if not download:
        raise FileNotFoundError(f"Missing {record['filename']}; enable explicit source download")
    temporary = target.with_suffix(".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(
                record["mirror_download_url"], headers={"User-Agent": "IntentLab/1.0"}
            )
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            verify_archive(temporary, record)
            temporary.replace(target)
            return target
        except (urllib.error.URLError, TimeoutError, OSError):
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2**attempt)
        except ValueError:
            temporary.unlink(missing_ok=True)
            raise
    raise AssertionError("Unreachable download state")


def extract_acquisition_runs(archive, directory, subject):
    """Extract only the two acquisition EDFs after checking every archive path."""
    cohort(subject)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    names = [f"sub-{subject:02d}/eeg/sub-{subject:02d}_task-{run}_eeg.edf" for run in RUNS]
    with zipfile.ZipFile(archive) as source:
        entries = source.infolist()
        if len({entry.filename for entry in entries}) != len(entries):
            raise ValueError("Duplicate archive members")
        if sum(entry.file_size for entry in entries) > 600_000_000:
            raise ValueError("Unexpected archive expansion")
        for entry in entries:
            path = PurePosixPath(entry.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in entry.filename
                or stat.S_ISLNK(entry.external_attr >> 16)
            ):
                raise ValueError(f"Unsafe archive member: {entry.filename}")
            if not (directory / entry.filename).resolve().is_relative_to(directory):
                raise ValueError(f"Archive path escapes destination: {entry.filename}")
        if not set(names) <= {entry.filename for entry in entries}:
            raise ValueError(f"Missing acquisition recordings for participant {subject}")
        for name in names:
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(name) as content, target.open("wb") as output:
                shutil.copyfileobj(content, output)
    return [directory / name for name in names]


def read_acquisition_epochs(paths, subject, interval=(0.5, 4.5)):
    """Return named 27 channel epochs in volts without fitting or normalising."""
    import mne
    import numpy as np

    cohort(subject)
    if tuple(interval) != (0.5, 4.5):
        raise ValueError("Reader requires the frozen cue relative interval 0.5 to 4.5 seconds")
    paths = list(map(Path, paths))
    if len(paths) != 2:
        raise ValueError("Exactly two acquisition recordings are required")
    epochs, trials, recordings = [], [], []
    for run, path in zip(RUNS, paths, strict=True):
        if path.name != f"sub-{subject:02d}_task-{run}_eeg.edf":
            raise ValueError("Recording path does not match participant and chronological run")
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        if raw.info["sfreq"] != 512 or raw.first_samp != 0:
            raise ValueError("Unexpected sampling rate or initial sample")
        if (
            len(raw.ch_names) != len(set(raw.ch_names))
            or set(raw.ch_names) != set(EEG_CHANNELS) | AUXILIARY_CHANNELS
        ):
            raise ValueError(f"Unexpected source channels for {path.name}: {raw.ch_names}")
        picks = [raw.ch_names.index(channel) for channel in EEG_CHANNELS]
        cues = [
            (float(onset), str(description))
            for onset, description in zip(raw.annotations.onset, raw.annotations.description, strict=True)
            if str(description) in {"769", "770"}
        ]
        if not cues or {label for _, label in cues} != {"769", "770"}:
            raise ValueError("Both hand cue codes must be present")
        if any(first[0] >= second[0] for first, second in zip(cues, cues[1:])):
            raise ValueError("Hand cues must be strictly chronological")
        for index, (onset, code) in enumerate(cues):
            cue_sample = int(round(onset * 512))
            start, end = cue_sample + 256, cue_sample + 2304
            if start < 0 or end > raw.n_times:
                raise ValueError(f"Incomplete epoch in {path.name}, cue {index + 1}")
            signal = raw.get_data(picks=picks, start=start, stop=end)
            if signal.shape != (27, 2048) or not np.isfinite(signal).all():
                raise ValueError("Nonfinite or incomplete source epoch")
            epochs.append(signal)
            trials.append(
                {
                    "participant": f"S{subject:02d}",
                    "subject": subject,
                    "cohort": cohort(subject),
                    "run": run,
                    "trial_id": f"S{subject:02d}-{run}-{index + 1:03d}",
                    "cue_code": code,
                    "label": 0 if code == "769" else 1,
                    "label_name": "left" if code == "769" else "right",
                    "cue_onset_seconds": onset,
                    "cue_sample": cue_sample,
                    "start_sample": start,
                    "end_sample_exclusive": end,
                }
            )
        recordings.append(
            {
                "participant": f"S{subject:02d}",
                "run": run,
                "filename": path.name,
                "sha256": file_hash(path),
                "samples": int(raw.n_times),
                "sampling_hz": 512,
                "channels": raw.ch_names,
                "selected_channels": EEG_CHANNELS,
                "left_cues": sum(code == "769" for _, code in cues),
                "right_cues": sum(code == "770" for _, code in cues),
            }
        )
    return np.stack(epochs), trials, recordings


def load_subject(subject_id, archive_path, protocol=None):
    """Read verified archives with temporary extraction and no retained raw copies.

    The returned epochs are in volts, have all 27 EEG channels, and have not been
    referenced, filtered, normalised or resampled. Labels are 0 left and 1 right.
    """
    protocol = {} if protocol is None else protocol
    selected_runs = protocol.get("runs", list(RUNS))
    interval = [protocol.get("epoch_start_s", 0.5), protocol.get("epoch_end_s", 4.5)]
    if list(selected_runs) != list(RUNS):
        raise ValueError("Only the two prespecified acquisition runs are supported")
    if list(interval) != [0.5, 4.5]:
        raise ValueError("Only the prespecified cue relative epoch is supported")
    if protocol.get("sampling_hz", 512) != 512 or protocol.get("eeg_channel_count", 27) != 27:
        raise ValueError("The source requires 27 EEG channels recorded at 512 Hz")
    manifest = load_manifest()
    record = next(item for item in manifest["participants"] if item["subject"] == subject_id)
    verify_archive(archive_path, record)
    with tempfile.TemporaryDirectory(prefix="dreyer-extract-", dir=Path(archive_path).parent) as temporary:
        paths = extract_acquisition_runs(archive_path, temporary, subject_id)
        epochs, trials, recordings = read_acquisition_epochs(paths, subject_id, interval)
    return {
        "epochs": epochs,
        "channels": list(EEG_CHANNELS),
        "trials": trials,
        "recordings": recordings,
        "archive": record,
        "dropped_trials": 0,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch and cross verify source metadata only")
    parser.add_argument("--write-manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_source_manifest()
    args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.write_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {manifest['participant_count']} archives, {manifest['archive_total_bytes']} bytes")
