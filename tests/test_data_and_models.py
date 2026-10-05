import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from intentlab.inference import Predictor

ROOT = Path(__file__).resolve().parents[1]


def test_people_are_disjoint_and_only_imagery_runs_used():
    rows = pq.read_table(ROOT / "data/derived/trials.parquet").to_pylist()
    sets = {
        split: {row["subject"] for row in rows if row["split"] == split}
        for split in ["train", "validation", "test"]
    }
    assert not sets["train"] & sets["test"]
    assert not sets["train"] & sets["validation"]
    assert not sets["validation"] & sets["test"]
    assert {r["run"] for r in rows} == {4, 8, 12}
    assert {r["label"] for r in rows} == {0, 1}
    assert len({r["id"] for r in rows}) == len(rows)
    manifest = json.loads((ROOT / "data/derived/manifest.json").read_text())
    for split, people in sets.items():
        assert people <= set(manifest["splits"][split])
        assert len(people) == manifest["counts"][split]["subjects"]
    assert len(manifest["records"]) == 327
    assert all(
        len(r["sha256"]) == 64 and r["url"].startswith("https://physionet.org/") for r in manifest["records"]
    )


def test_demo_is_first_six_valid_test_trials_not_cherry_picked():
    all_rows = pq.read_table(ROOT / "data/derived/trials.parquet").to_pylist()
    demo = json.loads((ROOT / "data/demo/trials.json").read_text())
    for subject in {row["subject"] for row in demo}:
        expected = [r["id"] for r in all_rows if r["subject"] == subject and r["split"] == "test"][:6]
        assert [r["id"] for r in demo if r["subject"] == subject] == expected
    x = np.load(ROOT / "data/demo/epochs.npz", allow_pickle=False)["x"]
    assert x.shape == (len(demo), 9, 480) and np.isfinite(x).all()


def test_frozen_artifact_integrity_and_export_parity():
    for filename, digest in json.loads((ROOT / "artifacts/checksums.json").read_text()).items():
        assert hashlib.sha256((ROOT / "artifacts" / filename).read_bytes()).hexdigest() == digest
    report = json.loads((ROOT / "artifacts/evaluation.json").read_text())
    assert report["onnx_max_absolute_logit_error"] < 1e-4
    assert (
        report["data_manifest_sha256"]
        == hashlib.sha256((ROOT / "data/derived/manifest.json").read_bytes()).hexdigest()
    )
    scores = {
        k: v["validation_participant_balanced_accuracy"]
        for k, v in report["models"].items()
        if k != "majority"
    }
    assert report["selected"] == max(scores, key=scores.get)
    for model in report["models"].values():
        assert sum(map(sum, model["test"]["confusion_matrix"])) == model["test"]["n"]


@pytest.mark.parametrize("model", ["majority", "bandpower", "csp_lda", "compact_eeg"])
def test_serving_all_models_produces_finite_probabilities(model):
    predictor = Predictor()
    x = np.load(ROOT / "data/demo/epochs.npz", allow_pickle=False)["x"][:4]
    probabilities = predictor.predict(x, model)
    assert probabilities.shape == (4,)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities, predictor.predict(x, model))
