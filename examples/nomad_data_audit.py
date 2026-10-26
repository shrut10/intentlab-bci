"""Verify and optionally inspect the public NoMAD intracortical recordings.

The metadata manifest is available offline. Downloads are opt-in and checksum
verified. Inspecting downloaded NWB files additionally requires h5py.
"""

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/external-data/nomad_manifest.json"
DEFAULT_DATA = ROOT / "data/raw/nomad"
CHUNK_SIZE = 1024 * 1024


def verify_file(path, expected):
    """Verify actual bytes against Dryad's published size and SHA-256."""
    path = Path(path)
    if path.stat().st_size != expected["size_bytes"]:
        raise ValueError(f"{path.name}: byte count does not match the published manifest")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected["published_sha256"]:
        raise ValueError(f"{path.name}: SHA-256 does not match the published manifest")
    return actual


def download_file(expected, directory):
    """Fetch one bounded, allow-listed public file; publish only verified bytes."""
    name = expected["filename"]
    if Path(name).name != name or name not in {"session1.nwb", "session2.nwb"}:
        raise ValueError("unexpected recording filename")
    url = expected["public_download_url"]
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "datadryad.org"
        or not parsed.path.startswith("/downloads/file_stream/")
    ):
        raise ValueError("unexpected public download URL")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    if target.exists():
        verify_file(target, expected)
        return target
    partial = target.with_suffix(".nwb.partial")
    # Exclusive creation prevents clobbering another in-progress download.
    with partial.open("xb") as stream:
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                count = 0
                while chunk := response.read(CHUNK_SIZE):
                    count += len(chunk)
                    if count > expected["size_bytes"]:
                        raise ValueError(f"{name}: download exceeded the published size")
                    stream.write(chunk)
            stream.flush()
            verify_file(partial, expected)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
    partial.replace(target)
    return target


def inspect_nwb(path):
    """Read counts and shapes without training, changing or claiming NWB validation."""
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "NWB inspection requires h5py; install it in a separate audit environment"
        ) from exc

    with h5py.File(path, "r") as handle:
        acquisitions = {}
        for name, group in handle["acquisition"].items():
            if "data" in group:
                acquisitions[name] = {"shape": list(group["data"].shape), "dtype": str(group["data"].dtype)}
        spikes = handle["acquisition/binned_spikes"]
        shape = spikes["data"].shape
        if len(shape) != 2 or shape[1] != 192:
            raise ValueError("expected a samples-by-192-channels binned-spike array")
        timing = {}
        if "timestamps" in spikes:
            timestamps = spikes["timestamps"][:]
            if timestamps.ndim != 1 or len(timestamps) != shape[0] or len(timestamps) < 2:
                raise ValueError("spike timestamps must match the sample count")
            differences = np.diff(timestamps)
            if not np.isfinite(timestamps).all() or (differences <= 0).any():
                raise ValueError("spike timestamps must be finite and strictly increasing")
            timing = {
                "first_timestamp_seconds": float(timestamps[0]),
                "last_timestamp_seconds": float(timestamps[-1]),
                "median_interval_seconds": float(np.median(differences)),
                "minimum_interval_seconds": float(differences.min()),
                "maximum_interval_seconds": float(differences.max()),
            }
        elif "starting_time" in spikes:
            rate = float(spikes["starting_time"].attrs["rate"])
            if not np.isfinite(rate) or rate <= 0:
                raise ValueError("spike sampling rate must be finite and positive")
            timing = {"sampling_rate_hz": rate, "interval_seconds": 1 / rate}
        else:
            raise ValueError("no spike time axis found")
        trials = handle["intervals/trials"]
        return {
            "sample_count": int(shape[0]),
            "channel_count": int(shape[1]),
            "trial_count": int(len(trials["id"])),
            "trial_fields": sorted(trials.keys()),
            "calibration_trial_count": int(np.count_nonzero(trials["is_calibration"][:] == 1)),
            "block_ids": sorted(int(value) for value in np.unique(trials["block_num"][:])),
            "timing": timing,
            "acquisitions": acquisitions,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="download the two public NWB recordings")
    parser.add_argument("--inspect", action="store_true", help="verify and inspect local NWB recordings")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, help="save the report as JSON instead of printing it")
    args = parser.parse_args(argv)
    manifest = json.loads(MANIFEST.read_text())
    report = {"dataset_doi": manifest["dataset_doi"], "status": "metadata_only", "files": []}
    for expected in manifest["files"]:
        path = args.data_dir / expected["filename"]
        if args.download:
            path = download_file(expected, args.data_dir)
        if args.inspect or args.download:
            actual = verify_file(path, expected)
            record = {"filename": path.name, "size_bytes": path.stat().st_size, "verified_sha256": actual}
            if args.inspect:
                record["inspection"] = inspect_nwb(path)
            report["files"].append(record)
    if args.inspect:
        report["status"] = "recording_bytes_verified_and_inspected"
    elif args.download:
        report["status"] = "recording_bytes_verified"
    else:
        report["files"] = manifest["files"]
        report["availability_note"] = manifest["availability_note"]
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, ValueError, RuntimeError, urllib.error.URLError) as error:
        print(
            f"NoMAD audit stopped: {error}. No recording inspection or model result is claimed.",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
