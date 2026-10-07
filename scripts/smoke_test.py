"""Check the actual running service locally, in Docker, or after deployment."""

import argparse
import json
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--attempts", type=int, default=1)
    args = parser.parse_args()
    base = args.url.rstrip("/")

    def request(path, payload=None):
        body = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(base + path, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=25) as response:
            return json.load(response)

    for attempt in range(args.attempts):
        try:
            health = request("/health")
            assert health["status"] == "ok" and health["artifacts_verified"]
            break
        except (OSError, AssertionError):
            if attempt == args.attempts - 1:
                raise
            time.sleep(2)
    overview = request("/api/overview")
    trials = request("/api/trials")["trials"]
    assert len(trials) == overview["demo_trials"]
    payload = {"trial_id": trials[0]["id"], "threshold": 0.5}
    result = request("/api/predict", payload)
    assert result["accepted"] and abs(result["right_probability"] + result["left_probability"] - 1) < 1e-6
    stressed = request("/api/predict", dict(payload, drop_channel="C3", noise=0.5))
    assert all(value == 0 for value in stressed["signal"][3])
    assert len(result["occlusion"]) == 9
    with urllib.request.urlopen(base + "/", timeout=25) as response:
        assert "Decode imagined movement" in response.read().decode()
    study = request("/api/adaptation")
    assert study["experiment"] == "adaptation-v1" and not study["production_model_changed"]
    assert len(study["summary"]) == 4 and study["counts"]["test"]["evaluation_trials"] == 630
    with urllib.request.urlopen(base + "/research/adaptation", timeout=25) as response:
        assert "Personal calibration and signal alignment" in response.read().decode()
    print(
        json.dumps(
            {
                "health": health,
                "trial": result["trial_id"],
                "right_probability": result["right_probability"],
                "api_checks": "passed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
