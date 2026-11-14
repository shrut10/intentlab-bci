"""Build the public command audit from archived evidence, without rerunning models."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "artifacts/command-replay-v1"
DESTINATION = ROOT / "web/commands-study.json"


def render():
    provenance = json.loads((FOLDER / "provenance.json").read_text())
    for name, expected in provenance["output_sha256"].items():
        if hashlib.sha256((FOLDER / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Archived command study file changed: {name}")
    result = json.loads((FOLDER / "evaluation.json").read_text())
    report = {key: value for key, value in result.items() if key not in {"training", "conditions"}}
    report.update(
        kind="command_study",
        title="Commands during movement imagery and instructed rest",
        description="A separate study of four participants using the first session for fitting, the second for policy selection and the third for evaluation",
        source=provenance["source"],
    )
    labels = {"movement_only": "Movement confidence", "rest_gate": "Movement with a task detector"}
    report["conditions"] = [
        dict(
            {key: value for key, value in condition.items() if key != "validation_candidates"},
            label=labels[condition["condition"]],
        )
        for condition in result["conditions"]
    ]
    # Keep every timeline record while avoiding indentation overhead in the public response.
    return json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    if args.check:
        if not DESTINATION.exists() or DESTINATION.read_text() != expected:
            raise SystemExit("Published command audit differs from the archived evidence")
        print("Published command audit matches the archived evidence")
    else:
        DESTINATION.write_text(expected)
        print(f"Wrote {DESTINATION}")


if __name__ == "__main__":
    main()
