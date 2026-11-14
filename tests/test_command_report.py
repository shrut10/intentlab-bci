"""Check that the published command study is recoverable from its saved evidence."""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from intentlab.commands import build_command_audit

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "artifacts/command-replay-v1"


def read_csv(name, numeric):
    with (FOLDER / name).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [{key: float(value) if key in numeric else value for key, value in row.items()} for row in rows]


def test_study_files_match_recorded_provenance():
    provenance = json.loads((FOLDER / "provenance.json").read_text())
    assert provenance["raw_data_redistributed"] is False
    assert len(provenance["source_records"]) == 24
    for name, expected in provenance["output_sha256"].items():
        assert hashlib.sha256((FOLDER / name).read_bytes()).hexdigest() == expected, name
    for name, expected in provenance["code_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name


@pytest.mark.parametrize("condition", ["movement_only", "rest_gate"])
@pytest.mark.parametrize("split", ["validation", "test"])
def test_archived_command_counts_rebuild_from_scores_and_annotations(condition, split):
    evaluation = json.loads((FOLDER / "evaluation.json").read_text())
    saved = next(row for row in evaluation["conditions"] if row["condition"] == condition)
    predictions = read_csv(
        f"predictions-{condition}-{split}.csv",
        {"time_s", "left_probability", "right_probability"},
    )
    events = read_csv(f"events-{split}.csv", {"start_s", "end_s"})
    assert build_command_audit(predictions, events, saved["policy"]) == saved[split]
    validation = {(row["participant"], row["session"]) for row in saved["validation"]["sessions"]}
    test = {(row["participant"], row["session"]) for row in saved["test"]["sessions"]}
    assert validation.isdisjoint(test)


def test_command_page_and_saved_report_are_served(client):
    page = client.get("/research/commands")
    assert page.status_code == 200
    assert 'id="command-report"' in page.text
    assert 'id="local-command-audit"' in page.text
    assert client.get("/static/commands.js").status_code == 200
    assert client.get("/static/commands.css").status_code == 200
    response = client.get("/static/commands-study.json")
    assert response.status_code == 200
    report = response.json()
    assert len(response.content) < 4_000_000
    assert report["kind"] == "command_study"
    assert report["production_model_changed"] is False
    assert report["participants"] == 4
    assert len(report["conditions"]) == 2
    assert response.content == (ROOT / "web/commands-study.json").read_bytes()
