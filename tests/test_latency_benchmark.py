import io
import json
import urllib.error

import pytest

from scripts.benchmark_latency import benchmark, percentile, request_prediction, summarise


def test_percentiles_handle_empty_singleton_and_interpolation():
    assert percentile([], 0.95) is None
    assert percentile([7], 0.95) == 7
    assert percentile([30, 10, 20, 0], 0.5) == 15
    assert percentile([30, 10, 20, 0], 0.95) == pytest.approx(28.5)


def test_summary_excludes_warmup_and_failures_without_hiding_counts():
    records = [
        {"phase": "first_observed", "success": True, "round_trip_ms": 500, "server_processing_ms": 300},
        {"phase": "warmup", "success": True, "round_trip_ms": 200, "server_processing_ms": 150},
        {"phase": "measured", "success": True, "round_trip_ms": 10, "server_processing_ms": 2},
        {"phase": "measured", "success": False, "round_trip_ms": 1000},
        {"phase": "measured", "success": True, "round_trip_ms": 30, "server_processing_ms": 6},
    ]
    result = summarise(records)
    assert result["client_round_trip"]["p50_ms"] == 20
    assert result["server_processing"]["p95_ms"] == pytest.approx(5.8)
    assert result["measured_attempts"] == 3
    assert result["measured_successes"] == 2
    assert result["measured_failures"] == result["all_phase_failures"] == 1
    assert summarise([records[3]])["client_round_trip"]["p95_ms"] is None


class Response(io.BytesIO):
    status = 200


@pytest.mark.parametrize("latency", [None, -1, True, float("nan"), float("inf")])
def test_invalid_server_timings_are_reported_as_failures(monkeypatch, latency):
    payload = {
        "trial_id": "trial",
        "latency_ms": latency,
        "model": "cnn",
        "model_version": "1",
        "app_version": "1",
    }
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: Response(json.dumps(payload).encode())
    )
    result = request_prediction("http://localhost:8000", {"trial_id": "trial"}, 1)
    assert result["success"] is False
    assert result["status_code"] == 200
    assert "server_processing_ms" not in result


def test_http_failure_preserves_status_and_does_not_publish_body(monkeypatch):
    def failing(*args, **kwargs):
        raise urllib.error.HTTPError("http://localhost", 503, "private body", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", failing)
    result = request_prediction("http://localhost:8000", {"trial_id": "trial"}, 1)
    assert result["status_code"] == 503 and not result["success"]
    assert "private body" not in json.dumps(result)


def test_benchmark_separates_first_warmup_and_measurement_and_flags_version_changes(monkeypatch):
    versions = iter(["1", "1", "1", "2"])

    def response(*args, **kwargs):
        return Response(
            json.dumps(
                {
                    "trial_id": "trial",
                    "latency_ms": 2,
                    "model": "cnn",
                    "model_version": "1",
                    "app_version": next(versions),
                }
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", response)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    result = benchmark("http://localhost:8000", {"trial_id": "trial"}, requests=2, warmup=1)
    assert [row["phase"] for row in result["requests"]] == [
        "first_observed",
        "warmup",
        "measured",
        "measured",
    ]
    assert result["summary"]["measured_successes"] == 2
    assert result["mixed_versions"] is True
    assert result["first_request_is_controlled_cold_start"] is False


@pytest.mark.parametrize("requests,warmup", [(0, 1), (201, 1), (1, 21)])
def test_request_budget_is_bounded(requests, warmup):
    with pytest.raises(ValueError):
        benchmark("http://localhost:8000", {"trial_id": "trial"}, requests=requests, warmup=warmup)
