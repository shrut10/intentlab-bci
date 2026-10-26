"""Offline download-integrity tests; optional inspection uses synthetic data only."""

import hashlib
import io
import json
import urllib.error

import pytest

from examples import nomad_data_audit as audit


def expected_file(payload=b"synthetic NWB bytes"):
    return {
        "filename": "session1.nwb",
        "size_bytes": len(payload),
        "published_sha256": hashlib.sha256(payload).hexdigest(),
        "public_download_url": "https://datadryad.org/downloads/file_stream/3993226",
    }


def test_manifest_keeps_unobserved_values_unknown():
    manifest = json.loads(audit.MANIFEST.read_text())
    assert manifest["recording_bytes_total"] == sum(item["size_bytes"] for item in manifest["files"])
    assert manifest["inspection_status"] == "metadata_verified_recording_bytes_not_available"
    assert all(item["computed_sha256"] is None and item["inspection"] is None for item in manifest["files"])
    assert all(len(item["published_sha256"]) == 64 for item in manifest["files"])


def test_download_verifies_bytes_and_reuses_verified_file(tmp_path, monkeypatch):
    payload = b"synthetic NWB bytes"
    calls = []

    def response(url, **kwargs):
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(audit.urllib.request, "urlopen", response)
    expected = expected_file(payload)
    downloaded = audit.download_file(expected, tmp_path)
    assert downloaded.read_bytes() == payload
    assert audit.verify_file(downloaded, expected) == expected["published_sha256"]
    assert audit.download_file(expected, tmp_path) == downloaded
    assert len(calls) == 1


@pytest.mark.parametrize("payload", [b"short", b"x" * 18, b"x" * 19])
def test_bad_download_never_leaves_a_recording(tmp_path, monkeypatch, payload):
    # Truncation, same-size corruption and an over-sized response all fail closed.
    monkeypatch.setattr(audit.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(payload))
    with pytest.raises(ValueError):
        audit.download_file(expected_file(), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_access_failure_is_reported_without_retry(tmp_path, monkeypatch):
    calls = []

    def denied(url, **kwargs):
        calls.append(url)
        raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(audit.urllib.request, "urlopen", denied)
    with pytest.raises(urllib.error.HTTPError, match="403"):
        audit.download_file(expected_file(), tmp_path)
    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [("filename", "../session1.nwb"), ("public_download_url", "https://example.com/session1.nwb")],
)
def test_unexpected_download_destination_is_rejected(tmp_path, field, value):
    expected = expected_file()
    expected[field] = value
    with pytest.raises(ValueError, match="unexpected"):
        audit.download_file(expected, tmp_path)


def test_default_report_does_not_download_or_claim_ingestion(tmp_path, monkeypatch):
    def unexpected_network(*args, **kwargs):
        pytest.fail("metadata reporting must not access the network")

    monkeypatch.setattr(audit.urllib.request, "urlopen", unexpected_network)
    output = tmp_path / "report.json"
    audit.main(["--output", str(output)])
    report = json.loads(output.read_text())
    assert report["status"] == "metadata_only"
    assert all(item["computed_sha256"] is None for item in report["files"])


def test_inspect_synthetic_nwb_counts_and_timing(tmp_path):
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    path = tmp_path / "synthetic.nwb"
    with h5py.File(path, "w") as handle:
        spikes = handle.create_group("acquisition/binned_spikes")
        spikes["data"] = np.zeros((5, 192), dtype=np.int16)
        spikes["timestamps"] = np.arange(5) * 0.02
        trials = handle.create_group("intervals/trials")
        trials["id"] = [0, 1, 2]
        trials["is_calibration"] = [1, 0, 0]
        trials["block_num"] = [0, 2, 2]
    result = audit.inspect_nwb(path)
    assert (result["sample_count"], result["channel_count"], result["trial_count"]) == (5, 192, 3)
    assert result["calibration_trial_count"] == 1
    assert result["block_ids"] == [0, 2]
    assert result["timing"]["median_interval_seconds"] == pytest.approx(0.02)
    assert result["acquisitions"]["binned_spikes"]["shape"] == [5, 192]
