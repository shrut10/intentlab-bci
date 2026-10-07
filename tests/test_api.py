import csv
import io
from pathlib import Path

import numpy as np
import pytest

from intentlab.api import assets


@pytest.fixture(scope="module")
def identifier(client):
    return client.get("/api/trials").json()["trials"][0]["id"]


def test_health_overview_and_security_headers(client):
    response = client.get("/health")
    assert response.json()["artifacts_verified"] is True
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert client.get("/api/overview").json()["counts"]["test"]["subjects"] > 0
    assert client.get("/api/evaluation").json()["selection_metric"].startswith("validation")


def test_prediction_and_abstention_contract(client, identifier):
    low = client.post("/api/predict", json={"trial_id": identifier, "threshold": 0.5}).json()
    high = client.post("/api/predict", json={"trial_id": identifier, "threshold": 0.99}).json()
    assert low["accepted"] is True
    assert low["decision"] in ["left", "right"]
    assert high["accepted"] is False and high["decision"] == "abstain"
    assert low["right_probability"] == high["right_probability"]
    assert low["right_probability"] + low["left_probability"] == pytest.approx(1)
    assert len(low["occlusion"]) == 9
    assert np.array(low["signal"]).shape == (9, 240)


def test_labels_are_not_used_as_inference_inputs(client, identifier):
    before = client.post("/api/predict", json={"trial_id": identifier}).json()
    record = assets()["trials"][0]
    old = record["label"]
    try:
        record["label"] = 1 - old
        after = client.post("/api/predict", json={"trial_id": identifier}).json()
        assert before["right_probability"] == after["right_probability"]
        assert before["recorded_label"] != after["recorded_label"]
    finally:
        record["label"] = old


def test_noise_and_channel_occlusion_work_over_http(client, identifier):
    payload = {"trial_id": identifier, "drop_channel": "C3", "noise": 0.5}
    first = client.post("/api/predict", json=payload).json()
    second = client.post("/api/predict", json=payload).json()
    assert not np.any(first["signal"][3])
    assert first["right_probability"] == second["right_probability"]
    assert first["signal"] == second["signal"]


@pytest.mark.parametrize(
    "extra",
    [
        {"threshold": 0.1},
        {"threshold": 1.1},
        {"noise": -1},
        {"noise": 100},
        {"drop_channel": "secret"},
        {"model": "unknown"},
        {"label": 1},
    ],
)
def test_invalid_requests_are_rejected(client, identifier, extra):
    response = client.post("/api/predict", json={"trial_id": identifier, **extra})
    assert response.status_code == 422


def test_unknown_and_oversized_requests(client):
    assert client.post("/api/predict", json={"trial_id": "S999-R04-E01"}).status_code == 404
    assert client.get("/api/trials/unknown").status_code == 404
    assert client.post("/api/predict", content=b"a" * 5000).status_code == 413
    assert (
        client.post(
            "/api/predict",
            content='{"threshold":NaN,"trial_id":"S010-R04-E01"}',
            headers={"Content-Type": "application/json"},
        ).status_code
        == 422
    )


def test_signal_download_matches_the_served_data(client, identifier):
    response = client.get(f"/api/trials/{identifier}/csv")
    rows = list(csv.reader(io.StringIO(response.text)))
    assert len(rows) == 481 and rows[0][1] == "FC3"
    assert float(rows[1][0]) == 1
    assert "attachment" in response.headers["Content-Disposition"]
    waveform = client.get(f"/api/trials/{identifier}").json()["signal"]
    assert float(rows[1][1]) == pytest.approx(waveform[0][0], abs=1e-4)


def test_static_app_and_docs_are_available(client):
    page = client.get("/")
    assert page.status_code == 200 and "Decode imagined movement" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/favicon.svg").status_code == 200
    assert "/api/predict" in client.get("/openapi.json").json()["paths"]
    assert client.get("/docs").status_code == 200


def test_model_artifact_corruption_fails_closed(tmp_path, monkeypatch):
    import intentlab.api as module

    folder = tmp_path / "artifacts"
    folder.mkdir()
    (folder / "checksums.json").write_text('{"models.json":"wrong"}')
    (folder / "models.json").write_text("{}")
    monkeypatch.setattr(module, "ROOT", Path(tmp_path))
    with pytest.raises(RuntimeError, match="integrity"):
        module.assets.__wrapped__()
