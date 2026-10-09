"""Audit your own prediction CSV, or reproduce IntentLab's archived follow-up audit.

Only NumPy is needed for a custom CSV. The bundled study also requires PyArrow.
This script reads local files and never sends predictions to an external service.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from intentlab.reliability import build_audit


def serialise(report):
    return json.dumps(report, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"


def bundled_audit(repeats=2000):
    import pyarrow.parquet as pq

    folder = ROOT / "artifacts/adaptation-v1"
    checksums = json.loads((folder / "checksums.json").read_text())
    for name in ("predictions.parquet", "evaluation.json"):
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != checksums[name]:
            raise ValueError(f"Published study checksum mismatch: {name}")
    evaluation = json.loads((folder / "evaluation.json").read_text())
    report = build_audit(
        pq.read_table(folder / "predictions.parquet").to_pylist(), evaluation["config"], repeats
    )
    report["source"] = {
        "study": "adaptation-v1",
        "sha256": {name: checksums[name] for name in ("predictions.parquet", "evaluation.json")},
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Your local prediction CSV; omit for the bundled study")
    parser.add_argument("--rule", type=Path, help="JSON threshold rule fixed before inspecting test outcomes")
    parser.add_argument("--output", type=Path, help="New JSON report; existing files are preserved")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument(
        "--check", action="store_true", help="Verify the checked-in public audit matches the archive"
    )
    args = parser.parse_args()
    if args.check and (args.input or args.rule or args.output or args.bootstrap_repeats != 2000):
        parser.error("--check uses only the bundled study and fixed settings")
    if not args.check and not args.output:
        parser.error("--output is required")
    if args.rule and not args.input:
        parser.error("--rule is only for a custom input; the bundled protocol is frozen")
    try:
        if args.input:
            with args.input.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            rule = json.loads(args.rule.read_text()) if args.rule else None
            report = build_audit(rows, rule, args.bootstrap_repeats)
            report["source"] = {"sha256": hashlib.sha256(args.input.read_bytes()).hexdigest()}
        else:
            report = bundled_audit(args.bootstrap_repeats)
        content = serialise(report)
        if args.check:
            if content != (ROOT / "web/reliability.json").read_text():
                raise ValueError(
                    "Public reliability report differs from the archive; regenerate and review it"
                )
            print("Public reliability audit matches the archived predictions")
        else:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(content)
            print(f"Wrote {len(report['groups'])} separate condition/seed/split groups to {args.output}")
            print("Test curves are descriptive; threshold selection uses validation only")
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"Audit failed: {error}\n")


if __name__ == "__main__":
    main()
