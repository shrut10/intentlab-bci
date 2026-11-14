"""Audit saved decoder scores and annotated sessions locally without uploading data."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intentlab.commands import build_command_audit

DEFAULT_POLICY = {
    "enter_threshold": 0.7,
    "release_threshold": 0.4,
    "hold_seconds": 0.5,
    "cooldown_seconds": 1.5,
    "max_gap_seconds": 0.75,
}


def read_csv(path):
    if path.stat().st_size > 20_000_000:
        raise ValueError(f"{path.name}: CSV exceeds the 20 MB limit")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, strict=True)
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"{path.name}: unique CSV column names are required")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(
                    f"{path.name}: CSV row ending at line {reader.line_num} has a different number "
                    "of cells from the header"
                )
            rows.append(row)
        return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        policy = json.loads(args.policy.read_text()) if args.policy else DEFAULT_POLICY
        report = build_command_audit(read_csv(args.predictions), read_csv(args.events), policy)
        report["provenance"] = {
            "status": "User supplied scores and annotations; provenance has not been independently verified",
            "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
            "events_sha256": hashlib.sha256(args.events.read_bytes()).hexdigest(),
        }
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    except (OSError, ValueError, TypeError, KeyError, csv.Error) as exc:
        parser.exit(1, f"Command audit failed: {exc}\n")
    print(f"Saved the command audit to {args.output}")


if __name__ == "__main__":
    main()
