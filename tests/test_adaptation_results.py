"""Audit the published study independently from its training runner."""

import hashlib
import io
import json
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
import pytest

from intentlab.adaptation import (
    CONDITIONS,
    binary_metrics,
    paired_interval,
    partition_rows,
    probabilities,
    select_threshold,
    selective_metrics,
)
from scripts.render_adaptation_report import render

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "artifacts/adaptation-v1"


@pytest.fixture(scope="module")
def report():
    return json.loads((FOLDER / "evaluation.json").read_text())


@pytest.fixture(scope="module")
def predictions():
    return pq.read_table(FOLDER / "predictions.parquet").to_pylist()


def test_study_inputs_protocol_and_exports_are_auditable(report):
    assert not report["production_model_changed"]
    assert report["status"] == "exploratory_existing_benchmark"
    for name, expected in json.loads((FOLDER / "checksums.json").read_text()).items():
        assert hashlib.sha256((FOLDER / name).read_bytes()).hexdigest() == expected
    for name, expected in report["run"]["input_and_code_sha256"].items():
        if name == "data/derived/epochs.npz":  # Full recordings are intentionally not in Git/CI.
            continue
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    assert len(report["models"]) == 6
    for model in report["models"]:
        assert model["onnx_max_absolute_logit_error"] < 1e-4
        assert model["best_epoch"] in [row["epoch"] for row in model["history"]]
        best = max(row["validation_participant_balanced_accuracy"] for row in model["history"])
        assert model["validation_participant_balanced_accuracy"] == best
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(FOLDER / f"{model['name']}.onnx"), sess_options=options)
        result = session.run(None, {"eeg": np.ones((2, 9, 480), dtype=np.float32)})[0]
        assert result.shape == (2,) and np.isfinite(result).all()


def test_calibration_manifest_excludes_all_later_trials_and_preserves_people(report):
    rows = pq.read_table(ROOT / "data/derived/trials.parquet").to_pylist()
    early, later = partition_rows(rows)
    audit = json.loads((FOLDER / "calibration_manifest.json").read_text())
    assert len(audit) == len(early) == 106
    for person in audit:
        assert person["calibration_ids"] == [rows[i]["id"] for i in early[person["subject"]]]
        assert person["evaluation_ids"] == [rows[i]["id"] for i in later[person["subject"]]]
        assert not set(person["calibration_ids"]) & set(person["evaluation_ids"])
        assert len(person["calibration_ids"]) == 10
        assert np.asarray(person["alignment_matrix"]).shape == (9, 9)
    assert report["counts"]["test"]["participants"] == 21
    assert report["counts"]["test"]["evaluation_trials"] == 630


def test_archived_metrics_and_thresholds_rebuild_from_predictions(report, predictions):
    for record in report["results"]:
        selected_by_split = {}
        for split in ("validation", "test"):
            rows = [
                r
                for r in predictions
                if (r["seed"], r["condition"], r["split"]) == (record["seed"], record["condition"], split)
            ]
            assert len({r["trial_id"] for r in rows}) == len(rows)
            assert {r["run"] for r in rows} == {8, 12}
            y, p, people = [np.array([r[key] for r in rows]) for key in ("label", "probability", "subject")]
            np.testing.assert_allclose(
                p, probabilities([r["logit"] for r in rows], [r["temperature"] for r in rows])
            )
            metrics = binary_metrics(y, p, people)
            for key in ("participant_balanced_accuracy", "brier", "ece", "accuracy"):
                assert metrics[key] == pytest.approx(record[split]["metrics"][key])
            assert metrics["participants"] == record[split]["metrics"]["participants"]
            assert selective_metrics(y, p, 0.75) == record[split]["fixed_threshold"]
            selected_by_split[split] = (y, p)
        # The threshold MUST come from validation predictions, not test performance.
        selection = select_threshold(*selected_by_split["validation"], report["config"])
        assert selection == record["validation"]["threshold_rule"]
        assert (
            selective_metrics(*selected_by_split["test"], selection["threshold"])
            == record["test"]["validation_threshold"]
        )


def test_comparison_uses_paired_people_and_temperature_does_not_change_labels(report, predictions):
    by_key = {(r["seed"], r["condition"], r["split"], r["trial_id"]): r for r in predictions}
    for key, row in by_key.items():
        counterpart = {"personal_temperature": "baseline", "alignment_temperature": "alignment"}.get(
            row["condition"]
        )
        if counterpart:
            base = by_key[(key[0], counterpart, key[2], key[3])]
            assert row["logit"] == base["logit"]
            assert (row["probability"] >= 0.5) == (base["probability"] >= 0.5)
    baseline = [r for r in report["results"] if r["condition"] == "baseline"]
    for condition in CONDITIONS:
        records = [r for r in report["results"] if r["condition"] == condition]
        assert [r["seed"] for r in records] == report["config"]["seeds"]
        for base, candidate in zip(baseline, records, strict=True):
            assert [p["subject"] for p in base["test"]["metrics"]["participants"]] == [
                p["subject"] for p in candidate["test"]["metrics"]["participants"]
            ]
        summary = report["summary"][condition]
        assert summary["mean_participant_balanced_accuracy"] == pytest.approx(
            np.mean([r["test"]["metrics"]["participant_balanced_accuracy"] for r in records])
        )
        for metric in ("balanced_accuracy", "brier"):
            a = [[p[metric] for p in r["test"]["metrics"]["participants"]] for r in baseline]
            b = [[p[metric] for p in r["test"]["metrics"]["participants"]] for r in records]
            expected = paired_interval(
                a, b, report["config"]["bootstrap_samples"], report["config"]["bootstrap_seed"]
            )
            assert expected == summary["paired_change_vs_baseline"][metric]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a":
            self.links.append(attrs.get("href", ""))


def test_published_follow_up_routes_and_report_match_archived_results(client, report):
    response = client.get("/research/adaptation")
    assert response.status_code == 200
    expected = render(report, (ROOT / "experiments/adaptation-v1/interpretation.txt").read_text())
    assert response.text == expected
    links = Links()
    links.feed(response.text)
    for link in links.links:
        if link.startswith("#"):
            assert link[1:] in links.ids
    assert client.get("/api/adaptation").json() == report
    download = client.get("/api/adaptation/predictions")
    assert download.status_code == 200 and "attachment" in download.headers["content-disposition"]
    assert pq.read_table(io.BytesIO(download.content)).num_rows == 15480
    assert 'href="/research/adaptation"' in client.get("/research").text
    assert 'href="/research/adaptation"' in client.get("/").text


def test_adaptation_report_corruption_fails_closed(tmp_path, monkeypatch):
    import intentlab.api as module

    folder = tmp_path / "artifacts/adaptation-v1"
    folder.mkdir(parents=True)
    (folder / "checksums.json").write_text('{"evaluation.json":"wrong"}')
    (folder / "evaluation.json").write_text("{}")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="integrity"):
        module.adaptation_report.__wrapped__()
