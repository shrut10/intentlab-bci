"""Exercise the public local command audit interface using real CSV files."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "command_audit_sample"


def run_cli(output, *, predictions=None, events=None, policy=True):
    args = [
        sys.executable,
        str(ROOT / "examples" / "command_audit.py"),
        "--predictions",
        str(predictions or SAMPLE / "predictions.csv"),
        "--events",
        str(events or SAMPLE / "events.csv"),
        "--output",
        str(output),
    ]
    if policy:
        args.extend(["--policy", str(SAMPLE / "policy.json")])
    return subprocess.run(args, capture_output=True, text=True, timeout=20, check=False)


def test_real_csv_generates_hand_computed_audit(tmp_path):
    output = tmp_path / "nested" / "audit.json"
    result = run_cli(output)
    assert result.returncode == 0, result.stderr
    audit = json.loads(output.read_text())
    assert audit["kind"] == "command_audit"
    assert audit["schema_version"] == "1.0"
    summary = audit["summary"]
    assert summary["prediction_rows"] == 4
    assert summary["command_count"] == 2
    assert summary["cues"] == 1
    assert summary["cues_with_correct"] == 1
    assert summary["cues_first_command_correct"] == 1
    assert summary["first_correct_latencies_seconds"] == [0.5]
    assert summary["median_first_correct_latency_seconds"] == 0.5
    assert summary["idle_commands"] == 1
    assert summary["idle_seconds"] == 1
    assert summary["idle_commands_per_minute"] == 60
    assert summary["gap_seconds"] == 0
    assert summary["unobserved_edge_seconds"] == 0.5
    assert [(c["time_s"], c["direction"], c["outcome"]) for c in audit["commands"]] == [
        (0.5, "left", "correct"),
        (1.5, "right", "idle"),
    ]
    assert audit["participants"][0]["participant"] == "SYNTHETIC"
    assert audit["policy"] == json.loads((SAMPLE / "policy.json").read_text())


def test_input_hashes_match_exact_csv_bytes(tmp_path):
    output = tmp_path / "audit.json"
    result = run_cli(output)
    assert result.returncode == 0, result.stderr
    provenance = json.loads(output.read_text())["provenance"]
    for name in ("predictions", "events"):
        expected = hashlib.sha256((SAMPLE / f"{name}.csv").read_bytes()).hexdigest()
        assert provenance[f"{name}_sha256"] == expected
    assert "not been independently verified" in provenance["status"]


def test_existing_report_cannot_be_overwritten(tmp_path):
    output = tmp_path / "audit.json"
    assert run_cli(output).returncode == 0
    original = output.read_bytes()
    result = run_cli(output)
    assert result.returncode == 1
    assert "Command audit failed" in result.stderr
    assert output.read_bytes() == original


@pytest.mark.parametrize(
    "csv_text",
    [
        "participant,session,time_s,left_probability,right_probability\nSYNTHETIC,example,0,NaN,0.1\n",
        "participant,session,time_s,left_probability,right_probability\nSYNTHETIC,example,0,0.9,0.9\n",
        "participant,session,time_s,left_probability,right_probability\n"
        "SYNTHETIC,example,0,0.9,0.1\nSYNTHETIC,example,0,0.9,0.1\n",
        "participant,session,time_s,left_probability,left_probability\nSYNTHETIC,example,0,0.9,0.1\n",
        "participant,session,time_s\nSYNTHETIC,example,0\n",
    ],
)
def test_invalid_prediction_csv_does_not_create_output(tmp_path, csv_text):
    predictions = tmp_path / "bad.csv"
    predictions.write_text(csv_text)
    output = tmp_path / "output" / "audit.json"
    result = run_cli(output, predictions=predictions)
    assert result.returncode == 1
    assert "Command audit failed" in result.stderr
    assert not output.exists()
    assert not output.parent.exists()


def test_incomplete_annotation_csv_does_not_create_output(tmp_path):
    events = tmp_path / "bad_events.csv"
    events.write_text("participant,session,event_id,start_s,end_s,state\nSYNTHETIC,example,cue,0,1,left\n")
    output = tmp_path / "audit.json"
    result = run_cli(output, events=events)
    assert result.returncode == 1
    assert not output.exists()


@pytest.mark.parametrize("row", ["SYNTHETIC,example,0,0.9,0.1,unexpected", "SYNTHETIC,example,0,0.9"])
def test_ragged_csv_rows_are_rejected_before_output(tmp_path, row):
    predictions = tmp_path / "ragged.csv"
    predictions.write_text(f"participant,session,time_s,left_probability,right_probability\n{row}\n")
    output = tmp_path / "output" / "audit.json"
    result = run_cli(output, predictions=predictions)
    assert result.returncode == 1
    assert "different number of cells from the header" in result.stderr
    assert not output.parent.exists()


def test_named_extra_csv_column_remains_supported(tmp_path):
    predictions = tmp_path / "extra.csv"
    lines = (SAMPLE / "predictions.csv").read_text().splitlines()
    predictions.write_text(lines[0] + ",note\n" + "\n".join(line + ",ignored" for line in lines[1:]) + "\n")
    output = tmp_path / "audit.json"
    result = run_cli(output, predictions=predictions)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["summary"]["command_count"] == 2


def test_unclosed_csv_quote_is_reported_without_traceback(tmp_path):
    predictions = tmp_path / "unclosed.csv"
    predictions.write_text(
        'participant,session,time_s,left_probability,right_probability\nSYNTHETIC,example,0,0.9,"0.1\n'
    )
    output = tmp_path / "audit.json"
    result = run_cli(output, predictions=predictions)
    assert result.returncode == 1
    assert "Command audit failed" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_omitting_policy_uses_documented_defaults(tmp_path):
    output = tmp_path / "audit.json"
    result = run_cli(output, policy=False)
    assert result.returncode == 0, result.stderr
    audit = json.loads(output.read_text())
    assert audit["policy"] == {
        "enter_threshold": 0.7,
        "release_threshold": 0.4,
        "hold_seconds": 0.5,
        "cooldown_seconds": 1.5,
        "max_gap_seconds": 0.75,
    }
    assert audit["summary"]["command_count"] == 1
