"""Source integrity, labels and interval boundaries for the unseen study."""

import hashlib
import zipfile

import numpy as np
import pytest

from scripts.dreyer_source import (
    AUXILIARY_CHANNELS,
    EEG_CHANNELS,
    RUNS,
    acquire_subject,
    cohort,
    extract_acquisition_runs,
    load_subject,
    read_acquisition_epochs,
    verify_archive,
)


def test_cohorts_are_disjoint_and_complete():
    assert [sum(cohort(subject) == group for subject in range(1, 88)) for group in "ABC"] == [60, 21, 6]
    for value in (0, 88, True, 1.0, "1"):
        with pytest.raises(ValueError):
            cohort(value)


def test_cached_archive_requires_size_and_both_publisher_hashes(tmp_path):
    content = b"verified data"
    path = tmp_path / "sub-01.zip"
    path.write_bytes(content)
    record = {
        "subject": 1,
        "filename": path.name,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "md5": hashlib.md5(content).hexdigest(),
    }
    assert acquire_subject(1, tmp_path, {"participants": [record]}) == path
    path.write_bytes(b"modified data")
    with pytest.raises(ValueError):
        verify_archive(path, record)


def test_download_requires_explicit_flag(tmp_path):
    with pytest.raises(FileNotFoundError, match="explicit"):
        acquire_subject(1, tmp_path, {"participants": [{"subject": 1, "filename": "sub-01.zip"}]})


@pytest.mark.parametrize(
    "changed",
    [
        {"runs": ["R3online", "R4online"]},
        {"epoch_start_s": 0.0},
        {"epoch_end_s": 5.0},
        {"sampling_hz": 160},
        {"eeg_channel_count": 32},
    ],
)
def test_reader_rejects_protocol_changes_before_reading_data(tmp_path, changed):
    with pytest.raises(ValueError):
        load_subject(1, tmp_path / "not_downloaded.zip", changed)


@pytest.mark.parametrize("unsafe", ["../escape", "/absolute", "sub-01/../../escape", "sub-01\\escape"])
def test_archive_validates_all_paths_before_extracting(tmp_path, unsafe):
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as source:
        source.writestr(unsafe, b"unsafe")
        for run in RUNS:
            source.writestr(f"sub-01/eeg/sub-01_task-{run}_eeg.edf", b"recording")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_acquisition_runs(path, tmp_path / "out", 1)
    assert not (tmp_path / "out/sub-01").exists()


def test_only_acquisition_runs_are_extracted(tmp_path):
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as source:
        for run in (*RUNS, "R3online"):
            source.writestr(f"sub-01/eeg/sub-01_task-{run}_eeg.edf", b"recording")
    paths = extract_acquisition_runs(path, tmp_path / "out", 1)
    assert len(paths) == 2 and all(path.exists() for path in paths)
    assert not (tmp_path / "out/sub-01/eeg/sub-01_task-R3online_eeg.edf").exists()


class FakeRaw:
    def __init__(self, invalid=False):
        self.info = {"sfreq": 512}
        self.first_samp = 0
        self.n_times = 8000
        self.ch_names = list(reversed(EEG_CHANNELS)) + sorted(AUXILIARY_CHANNELS)
        self.annotations = type("Annotations", (), {"onset": [1.0, 8.0], "description": ["769", "770"]})()
        self.invalid = invalid

    def get_data(self, picks, start, stop):
        assert stop - start == 2048
        result = np.broadcast_to(np.asarray(picks)[:, None], (len(picks), stop - start)).astype(float).copy()
        if self.invalid:
            result[0, 0] = np.nan
        return result


def test_reader_preserves_labels_named_channels_and_exact_boundaries(tmp_path, monkeypatch):
    mne = pytest.importorskip("mne")
    raw = FakeRaw()
    monkeypatch.setattr(mne.io, "read_raw_edf", lambda *args, **kwargs: raw)
    paths = [tmp_path / f"sub-01_task-{run}_eeg.edf" for run in RUNS]
    for path in paths:
        path.write_bytes(b"source metadata fixture")
    epochs, trials, recordings = read_acquisition_epochs(paths, 1)
    assert epochs.shape == (4, 27, 2048)
    assert [row["label"] for row in trials] == [0, 1, 0, 1]
    assert [row["run"] for row in trials] == [RUNS[0], RUNS[0], RUNS[1], RUNS[1]]
    assert trials[0]["start_sample"] == 768 and trials[0]["end_sample_exclusive"] == 2816
    np.testing.assert_array_equal(epochs[0, :, 0], np.arange(26, -1, -1))
    assert all(row["left_cues"] == row["right_cues"] == 1 for row in recordings)


@pytest.mark.parametrize("problem", ["bounds", "channel", "nonfinite", "order"])
def test_source_failures_stop_instead_of_dropping_trials(tmp_path, monkeypatch, problem):
    mne = pytest.importorskip("mne")
    raw = FakeRaw(invalid=problem == "nonfinite")
    if problem == "bounds":
        raw.n_times = 5000
    elif problem == "channel":
        raw.ch_names.remove("C3")
    elif problem == "order":
        raw.annotations.onset = [8.0, 1.0]
    monkeypatch.setattr(mne.io, "read_raw_edf", lambda *args, **kwargs: raw)
    paths = [tmp_path / f"sub-01_task-{run}_eeg.edf" for run in RUNS]
    for path in paths:
        path.write_bytes(b"fixture")
    with pytest.raises(ValueError):
        read_acquisition_epochs(paths, 1)
