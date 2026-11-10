"""Read-only audit of the frozen dataset's cached EDF files, without MNE or retraining."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intentlab.edf_timing import audit_edf_timing


def audit_cached_recordings(raw_dir: Path, manifest_path: Path) -> dict:
    records = json.loads(manifest_path.read_text())["records"]
    if not records:
        raise ValueError("The manifest contains no recordings")
    total = 0
    for record in records:
        path = raw_dir / Path(record["file"]).name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"{path.name}: checksum differs from the frozen source manifest")
        total += audit_edf_timing(path).records
    return {
        "files": len(records),
        "data_records": total,
        "format": "EDF+C",
        "checksums_verified": True,
        "record_onsets_contiguous": True,
        "first_onset_seconds": 0,
        "scope": "Stored file structure and timing; cannot establish absence of acquisition-time sample loss",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/derived/manifest.json")
    args = parser.parse_args(argv)
    try:
        report = audit_cached_recordings(args.raw_dir, args.manifest)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"EEG timing audit failed: {exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
