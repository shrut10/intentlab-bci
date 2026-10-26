"""Measure a bounded sequence of prediction requests without claiming live BCI latency.

Example: python scripts/benchmark_latency.py --url http://127.0.0.1:8000
The first observed request is separate from warm-up and measured requests; it
is not a controlled cold start, because a remote server may already be warm.
"""

import argparse
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def percentile(values, fraction):
    """Linearly interpolated empirical quantile, including singleton samples."""
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def distribution(values):
    return {
        "n": len(values),
        "p50_ms": percentile(values, 0.5),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
    }


def summarise(records):
    measured = [row for row in records if row["phase"] == "measured"]
    successful = [row for row in measured if row["success"]]
    return {
        "measured_attempts": len(measured),
        "measured_successes": len(successful),
        "measured_failures": len(measured) - len(successful),
        "all_phase_failures": sum(not row["success"] for row in records),
        "client_round_trip": distribution([row["round_trip_ms"] for row in successful]),
        "server_processing": distribution([row["server_processing_ms"] for row in successful]),
        "excluded_phases": ["first_observed", "warmup"],
        "failed_requests_excluded_from_percentiles": True,
    }


def request_prediction(url, payload, timeout):
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url.rstrip("/") + "/api/predict", data=body, headers={"Content-Type": "application/json"}
    )
    status = None
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            result = json.load(response)
        elapsed = (time.perf_counter() - started) * 1000
        server = result.get("latency_ms")
        if (
            isinstance(server, bool)
            or not isinstance(server, (float, int))
            or not math.isfinite(server)
            or server < 0
        ):
            raise ValueError("response lacks a finite, non-negative latency_ms")
        if result.get("trial_id") != payload["trial_id"]:
            raise ValueError("response trial does not match the request")
        if not all(result.get(key) for key in ("model", "model_version", "app_version")):
            raise ValueError("response lacks model/application identifiers")
        return {
            "success": True,
            "status_code": status,
            "round_trip_ms": elapsed,
            "server_processing_ms": float(server),
            "model": result["model"],
            "model_version": result["model_version"],
            "app_version": result["app_version"],
        }
    except (OSError, ValueError, AttributeError) as error:
        # Do not save response bodies, query strings or recording arrays in the report.
        if isinstance(error, urllib.error.HTTPError):
            status = error.code
        return {
            "success": False,
            "status_code": status,
            "round_trip_ms": (time.perf_counter() - started) * 1000,
            "error_type": type(error).__name__,
        }


def benchmark(url, payload, requests=20, warmup=3, timeout=10.0, interval=0.1):
    if not 1 <= requests <= 200 or not 0 <= warmup <= 20:
        raise ValueError("requests must be 1–200 and warmup 0–20")
    if not 0 < timeout <= 60 or not 0.05 <= interval <= 2:
        raise ValueError("timeout must be positive and at most 60 s; interval must be 0.05–2 s")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("use an HTTP(S) service URL without embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("the service URL must not contain a query or fragment")
    records = []
    phases = ["first_observed"] + ["warmup"] * warmup + ["measured"] * requests
    for index, phase in enumerate(phases):
        if index:
            time.sleep(interval)
        records.append({"phase": phase, "request_index": index, **request_prediction(url, payload, timeout)})
    versions = sorted(
        {(row["model"], row["model_version"], row["app_version"]) for row in records if row["success"]}
    )
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "service": url.rstrip("/"),
        "payload": payload,
        "timeout_seconds": timeout,
        "interval_seconds": interval,
        "request_mode": "sequential urllib requests; not a concurrency or browser benchmark",
        "first_request_is_controlled_cold_start": False,
        "model_application_versions": [
            {"model": model, "model_version": model_version, "app_version": app_version}
            for model, model_version, app_version in versions
        ],
        "mixed_versions": len(versions) > 1,
        "interpretation": (
            "Client round trip includes HTTP transfer and JSON parsing; server processing is the API's "
            "latency_ms, including prediction and channel analysis. Neither includes EEG acquisition or "
            "establishes live-control delay. Percentiles are sample estimates, especially uncertain for small n. "
            "First-observed and warm-up requests, and failed measured requests, are excluded from percentiles. "
            "Do not compare a mixed-version report as if it measured one unchanged model."
        ),
        "summary": summarise(records),
        "requests": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--trial-id", default="S010-R04-E01")
    parser.add_argument("--model", default=None, help="Omit to use the service's selected model")
    parser.add_argument("--requests", type=int, default=20, help="Measured requests, 1–200")
    parser.add_argument(
        "--warmup", type=int, default=3, help="Warm-up requests after the first observation, 0–20"
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--interval", type=float, default=0.1, help="Pause between requests, 0.05–2 seconds")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {"trial_id": args.trial_id}
    if args.model:
        payload["model"] = args.model
    try:
        result = benchmark(args.url, payload, args.requests, args.warmup, args.timeout, args.interval)
    except ValueError as error:
        parser.error(str(error))
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return int(result["summary"]["all_phase_failures"] > 0 or result["mixed_versions"])


if __name__ == "__main__":
    raise SystemExit(main())
