"""Scientific boundary checks for the separate Zhou2016 command study."""

import zipfile

import numpy as np
import pytest

import scripts.run_command_study as study
from scripts.run_command_study import (
    causal_signal,
    fit_models,
    label_complete_windows,
    partition_events,
    read_protocol,
    safe_extract,
    split_for_session,
    trailing_features,
)


def test_filter_and_features_cannot_see_future_samples():
    rng = np.random.default_rng(2027)
    original = rng.normal(size=(14, 2500))
    changed = original.copy()
    changed[:, 1250:] += 500 * rng.normal(size=(14, 1250))
    first, second = causal_signal(original), causal_signal(changed)
    np.testing.assert_array_equal(first[:, :1250], second[:, :1250])
    features, starts, ends = trailing_features(first)
    future_features, _, _ = trailing_features(second)
    np.testing.assert_array_equal(features[ends <= 5], future_features[ends <= 5])
    assert starts[0] == 0 and ends[0] == 2
    assert np.all(ends - starts == 2)
    assert np.all(np.diff(ends) == 0.5)
    assert ends[-1] < original.shape[1] / 250


def test_events_partition_unknown_and_rest_with_preparation_guard():
    cues = [
        {"time_s": 2, "state": "left"},
        {"time_s": 12, "state": "right"},
        {"time_s": 21, "state": "other"},
    ]
    events = partition_events(cues, 29, "S1", "1A")
    assert events[0]["start_s"] == 0
    assert events[-1]["end_s"] == 29
    assert all(a["end_s"] == b["start_s"] for a, b in zip(events, events[1:]))
    rest = [event for event in events if event["state"] == "idle"]
    # The second cue has only nine seconds until the next cue, and the last rest
    # would exceed the file. Neither interval is silently shortened or labelled idle.
    assert [(event["start_s"], event["end_s"]) for event in rest] == [(7.5, 10.5)]
    assert any(event["state"] == "other" for event in events)


def test_training_windows_never_straddle_labels_or_rest_buffer():
    events = partition_events([{"time_s": 2, "state": "left"}], 12, "S1", "1A")
    starts = np.array([1.5, 2, 5, 5.5, 7, 7.5, 8.5, 9])
    labels = label_complete_windows(starts, starts + 2, events)
    assert labels.tolist() == ["unknown", "left", "left", "unknown", "unknown", "idle", "idle", "unknown"]


def test_session_split_is_chronological_and_disjoint():
    protocol = read_protocol()
    assert [split_for_session(number, protocol) for number in (1, 2, 3)] == ["train", "validation", "test"]
    with pytest.raises(ValueError, match="exactly one"):
        split_for_session(4, protocol)


def test_scalers_and_fitting_ignore_later_session_arrays():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(12)
    labels = np.array(["left", "right", "idle"] * 20)
    records = [
        {
            "participant": "S1",
            "session": "1A",
            "session_number": 1,
            "features": rng.normal(size=(60, 42)),
            "labels": labels,
        }
    ]

    class Unreadable:
        def __array__(self, *args, **kwargs):
            raise AssertionError("Later session data was accessed during fitting")

    later = [
        {
            "participant": "S1",
            "session": f"{session}A",
            "session_number": session,
            "features": Unreadable(),
            "labels": Unreadable(),
        }
        for session in (2, 3)
    ]
    models, counts = fit_models(records + later, read_protocol())
    movement, gate = models["S1"]
    np.testing.assert_allclose(movement[0].mean_, records[0]["features"][labels != "idle"].mean(axis=0))
    np.testing.assert_allclose(gate[0].mean_, records[0]["features"].mean(axis=0))
    assert counts[0]["training_sessions"] == ["1A"]
    assert counts[0]["movement_windows"] == 40


@pytest.mark.parametrize("path", ["../escape", "/absolute", "data/../../escape", "data\\escape"])
def test_archive_extraction_rejects_unsafe_members(tmp_path, path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("data/S1_1A.cnt", b"safe")
        zipped.writestr(path, b"unsafe")
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract(archive, tmp_path / "extracted", ["data/S1_1A.cnt"])
    assert not (tmp_path / "extracted/data/S1_1A.cnt").exists()


def test_archive_extracts_only_expected_recordings(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("data/S1_1A.cnt", b"recording")
        zipped.writestr("README", b"ignored")
    paths = safe_extract(archive, tmp_path / "extracted", ["data/S1_1A.cnt"])
    assert paths[0].read_bytes() == b"recording"
    assert not (tmp_path / "extracted/README").exists()


def test_selection_uses_validation_target_and_conservative_ties(monkeypatch):
    def audit(predictions, events, policy):
        assert predictions == ["validation scores"]
        assert events == ["validation labels"]
        return {"summary": {"cues": 100, "cues_first_command_correct": 60, "idle_commands_per_minute": 0.5}}

    monkeypatch.setattr(study, "build_command_audit", audit)
    selected, met, candidates = study.choose_policy(
        ["validation scores"], ["validation labels"], read_protocol()
    )
    assert met and len(candidates) == 15
    assert selected["enter_threshold"] == 0.9 and selected["hold_seconds"] == 1


@pytest.mark.parametrize("fraction,idle_rate", [(0.49, 0.0), (0.9, 1.01), (0.9, None)])
def test_policy_falls_back_if_either_target_requirement_fails(monkeypatch, fraction, idle_rate):
    monkeypatch.setattr(
        study,
        "build_command_audit",
        lambda *args: {
            "summary": {
                "cues": 100,
                "cues_first_command_correct": round(fraction * 100),
                "idle_commands_per_minute": idle_rate,
            }
        },
    )
    policy, met, _ = study.choose_policy([], [], read_protocol())
    assert not met
    assert policy["enter_threshold"] == 0.9 and policy["hold_seconds"] == 1
