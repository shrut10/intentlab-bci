"""Check source membership and the committed barrier before final EEG access."""

import copy
import json
import subprocess
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.run_unseen_study as study


def protocol():
    return json.loads((study.PROTOCOL_DIR / "protocol.json").read_text())


def source_trials(person=1):
    return [
        {
            "participant": f"S{person:02d}",
            "subject": person,
            "cohort": "A" if person <= 60 else "B" if person <= 81 else "C",
            "run": run,
            "trial_id": f"S{person:02d}-{run}-{label + 1:03d}",
            "label": label,
            "cue_code": "769" if label == 0 else "770",
            "label_name": "left" if label == 0 else "right",
        }
        for run in ("R1acquisition", "R2acquisition")
        for label in (0, 1)
    ]


def test_frozen_splits_are_disjoint_and_preserve_each_source_cohort():
    specification = protocol()
    records = study.read_splits(specification)
    partitions = {
        split: {row["participant"] for row in records if row["split"] == split}
        for split in specification["partitions"]
    }
    assert {name: len(people) for name, people in partitions.items()} == {
        "train": 45,
        "validation": 15,
        "test": 27,
    }
    assert len(set.union(*partitions.values())) == 87
    assert not (partitions["train"] & partitions["validation"])
    assert not (partitions["train"] & partitions["test"])
    assert not (partitions["validation"] & partitions["test"])
    for cohort, expected in specification["cohort_partitions"].items():
        counts = Counter(row["split"] for row in records if row["cohort"] == cohort)
        assert [counts[split] for split in ("train", "validation", "test")] == expected


def test_split_cohort_swap_is_rejected_even_when_global_counts_stay_the_same(tmp_path, monkeypatch):
    specification = protocol()
    manifest = json.loads((study.PROTOCOL_DIR / "split_manifest.json").read_text())
    first = next(row for row in manifest["participants"] if row["cohort"] == "A" and row["split"] == "train")
    second = next(row for row in manifest["participants"] if row["cohort"] == "B" and row["split"] == "test")
    first["split"], second["split"] = second["split"], first["split"]
    (tmp_path / "split_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(study, "PROTOCOL_DIR", tmp_path)
    with pytest.raises(ValueError, match="[Cc]ohort"):
        study.read_splits(specification)


@pytest.mark.parametrize(
    "field,value",
    [
        ("participant", "S02"),
        ("cohort", "B"),
        ("subject", 2),
        ("run", "R3online"),
        ("trial_id", ""),
        ("trial_id", "x" * 129),
        ("label", True),
        ("label", 2),
    ],
)
def test_invalid_trial_source_and_membership_is_rejected(field, value):
    trials = source_trials()
    trials[0][field] = value
    with pytest.raises(ValueError):
        study.validate_trials(trials, {"participant": "S01", "source_id": 1, "cohort": "A"})


def test_duplicate_trial_ids_and_missing_classes_fail():
    record = {"participant": "S01", "source_id": 1, "cohort": "A"}
    trials = source_trials()
    trials[1]["trial_id"] = trials[0]["trial_id"]
    with pytest.raises(ValueError, match="unique"):
        study.validate_trials(trials, record)
    with pytest.raises(ValueError, match="classes"):
        study.validate_trials([row for row in source_trials() if row["label"] == 0], record)


def test_fit_stage_cannot_collect_test_people(monkeypatch, tmp_path):
    specification = protocol()
    records = study.read_splits(specification)
    args = SimpleNamespace(output=tmp_path, cache=tmp_path, raw=tmp_path, download=False)
    expected = {row["participant"] for row in records if row["split"] != "test"}

    class StoppedAfterCheckingCollection(Exception):
        pass

    def inspect_collection(selected, *args):
        assert {row["participant"] for row in selected} == expected
        assert all(row["split"] in {"train", "validation"} for row in selected)
        raise StoppedAfterCheckingCollection

    monkeypatch.setattr(study, "collect", inspect_collection)
    with pytest.raises(StoppedAfterCheckingCollection):
        study.fit_stage(specification, records, args)


@pytest.fixture
def sealed_selection(tmp_path, monkeypatch):
    output, cache = tmp_path / "output", tmp_path / "cache"
    output.mkdir()
    cache.mkdir()
    model_path = cache / "fitted_models.joblib"
    model_path.write_bytes(b"synthetic model bytes used only to test the seal")
    predictions = output / "validation_predictions.csv"
    predictions.write_text("synthetic validation file used only to test the seal\n")
    selection = {
        "protocol_commit": "frozen-protocol",
        "code_sha256": {"code": "fixed"},
        "packages": {"package": "fixed"},
        "model_sha256": study.file_hash(model_path),
        "validation_predictions_sha256": study.file_hash(predictions),
        "test_participants_accessed": False,
    }
    (output / "selection.json").write_text(json.dumps(selection))
    monkeypatch.setattr(study, "source_hashes", lambda: {"code": "fixed"})
    monkeypatch.setattr(study, "packages", lambda: {"package": "fixed"})
    monkeypatch.setattr(study, "verify_committed", lambda *args: None)
    args = SimpleNamespace(
        output=output,
        cache=cache,
        raw=tmp_path / "raw",
        download=False,
        protocol_commit="frozen-protocol",
        selection_commit="frozen-selection",
    )
    return args, selection


def test_valid_seal_checks_exact_model_and_validation_bytes(sealed_selection):
    args, expected = sealed_selection
    assert study.verify_selection(args) == expected


@pytest.mark.parametrize(
    "change", ["no_commit", "protocol", "code", "packages", "model", "predictions", "access"]
)
def test_invalid_seal_fails_before_reading_any_test_data(sealed_selection, monkeypatch, change):
    args, selection = sealed_selection
    if change == "no_commit":
        args.selection_commit = None
    elif change == "model":
        (args.cache / "fitted_models.joblib").write_bytes(b"changed")
    elif change == "predictions":
        (args.output / "validation_predictions.csv").write_text("changed")
    else:
        key = {
            "protocol": "protocol_commit",
            "code": "code_sha256",
            "packages": "packages",
            "access": "test_participants_accessed",
        }[change]
        selection[key] = True if change == "access" else "changed"
        (args.output / "selection.json").write_text(json.dumps(selection))

    def forbidden(*args, **kwargs):
        pytest.fail("Final participant data was accessed before a valid seal")

    monkeypatch.setattr(study, "collect", forbidden)
    monkeypatch.setattr(study.joblib, "load", forbidden)
    with pytest.raises(ValueError):
        study.test_stage({}, [], args)


def test_committed_selection_check_happens_before_test_collection(sealed_selection, monkeypatch):
    args, _ = sealed_selection
    calls = []

    def reject_commit(commit, paths):
        calls.append((commit, [path.name for path in paths]))
        raise ValueError("Changed committed selection")

    def forbidden(*args, **kwargs):
        pytest.fail("Test data was read before committed selection verification")

    monkeypatch.setattr(study, "verify_committed", reject_commit)
    monkeypatch.setattr(study, "collect", forbidden)
    with pytest.raises(ValueError, match="committed"):
        study.test_stage({}, [], args)
    assert calls == [("frozen-selection", ["selection.json", "validation_predictions.csv"])]


def test_existing_final_results_are_never_overwritten(sealed_selection, monkeypatch):
    args, _ = sealed_selection
    destination = args.output / "evaluation.json"
    destination.write_text("original result")
    monkeypatch.setattr(
        study, "verify_selection", lambda *args: pytest.fail("Existing results were not protected")
    )
    with pytest.raises(ValueError, match="overwrite"):
        study.test_stage({}, [], args)
    assert destination.read_text() == "original result"


def test_actual_git_seal_detects_selection_edits(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    path = tmp_path / "selection.json"
    path.write_text('{"selection":"fixed"}\n')
    subprocess.run(["git", "add", "selection.json"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=IntentLab test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "Freeze synthetic selection",
        ],
        cwd=tmp_path,
        check=True,
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    monkeypatch.setattr(study, "ROOT", tmp_path)
    study.verify_committed(commit, [path])
    path.write_text('{"selection":"changed"}\n')
    with pytest.raises(ValueError, match="committed seal"):
        study.verify_committed(commit, [path])


def test_split_data_keeps_labels_and_arrays_aligned_without_changing_inputs():
    trials = source_trials(1) + source_trials(2)
    features = {"bandpower": np.arange(8 * 27).reshape(8, 27), "covariance": np.tile(np.eye(9), (8, 1, 1))}
    before = copy.deepcopy(features)
    selected, selected_trials = study.split_data(features, trials, {"S02"})
    assert selected_trials == trials[4:]
    np.testing.assert_array_equal(selected["bandpower"], features["bandpower"][4:])
    for name in features:
        np.testing.assert_array_equal(features[name], before[name])


@pytest.mark.parametrize("field,value", [("cue_code", "770"), ("label_name", "right")])
def test_source_cue_mapping_must_match_label(field, value):
    trials = source_trials()
    trials[0][field] = value
    with pytest.raises(ValueError, match="disagrees"):
        study.validate_trials(trials, {"participant": "S01", "source_id": 1, "cohort": "A"})


def test_missing_acquisition_run_cannot_be_silently_accepted():
    with pytest.raises(ValueError, match="Both acquisition runs"):
        study.validate_trials(source_trials()[:2], {"participant": "S01", "source_id": 1, "cohort": "A"})


@pytest.mark.parametrize("change", ["labels", "order", "archive", "arrays"])
def test_cache_binds_source_metadata_to_exact_feature_bytes(tmp_path, monkeypatch, change):
    specification = protocol()
    record = {"participant": "S01", "source_id": 1, "cohort": "A", "split": "train"}
    source = {
        "epochs": np.zeros((4, 27, 2048)),
        "channels": specification["channels"] + [f"extra{index}" for index in range(18)],
        "trials": source_trials(),
        "recordings": [{"run": "synthetic", "sha256": "fixed"}],
        "archive": {"sha256": "fixed"},
        "dropped_trials": 0,
    }
    monkeypatch.setattr(study, "source_hashes", lambda: {"code": "fixed"})
    monkeypatch.setattr(study, "packages", lambda: {"package": "fixed"})
    monkeypatch.setattr(study, "acquire_subject", lambda *args, **kwargs: tmp_path / "synthetic.zip")
    monkeypatch.setattr(study, "load_subject", lambda *args, **kwargs: source)
    monkeypatch.setattr(
        study,
        "extract_features",
        lambda *args: {"bandpower": np.zeros((4, 27)), "covariance": np.tile(np.eye(9) / 9, (4, 1, 1))},
    )
    manifest = {"participants": [{"subject": 1, "sha256": "fixed"}]}
    cache = tmp_path / "features"
    expected, _ = study.load_features(record, specification, manifest, tmp_path, cache, False)

    def forbidden(*args, **kwargs):
        pytest.fail("A verified existing cache should not require another source read")

    monkeypatch.setattr(study, "acquire_subject", forbidden)
    actual, _ = study.load_features(record, specification, manifest, tmp_path, cache, False)
    np.testing.assert_array_equal(actual["bandpower"], expected["bandpower"])
    metadata_path = cache / "S01.json"
    metadata = json.loads(metadata_path.read_text())
    if change == "labels":
        metadata["trials"][0]["label"] = 1
    elif change == "order":
        metadata["trials"].reverse()
    elif change == "archive":
        metadata["archive"]["sha256"] = "changed"
    else:
        with (cache / "S01.npz").open("ab") as handle:
            handle.write(b"changed")
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="cache"):
        study.load_features(record, specification, manifest, tmp_path, cache, False)


def test_source_code_change_during_collection_stops_before_fitting(tmp_path, monkeypatch):
    current = {"value": "fixed"}
    monkeypatch.setattr(study, "source_hashes", lambda: {"code": current["value"]})
    monkeypatch.setattr(study, "packages", lambda: {"package": "fixed"})
    records = [
        {"participant": "S01", "source_id": 1, "cohort": "A", "split": "train"},
        {"participant": "S02", "source_id": 2, "cohort": "A", "split": "validation"},
    ]

    def changed_collection(*args, **kwargs):
        current["value"] = "changed"
        return (
            {"bandpower": np.zeros((8, 27)), "covariance": np.tile(np.eye(9) / 9, (8, 1, 1))},
            source_trials(1) + source_trials(2),
            [],
        )

    monkeypatch.setattr(study, "collect", changed_collection)
    monkeypatch.setattr(study, "fit_models", lambda *args: pytest.fail("Changed code reached model fitting"))
    args = SimpleNamespace(output=tmp_path, cache=tmp_path, raw=tmp_path, download=False)
    with pytest.raises(ValueError):
        study.fit_stage(protocol(), records, args)
    assert not (tmp_path / "selection.json").exists()
    assert not (tmp_path / "fitted_models.joblib").exists()


def test_model_change_during_final_collection_stops_before_deserialisation(sealed_selection, monkeypatch):
    args, _ = sealed_selection

    def changed_collection(*unused, **kwargs):
        (args.cache / "fitted_models.joblib").write_bytes(b"changed during collection")
        return {}, [], []

    monkeypatch.setattr(study, "collect", changed_collection)
    monkeypatch.setattr(study.joblib, "load", lambda *args: pytest.fail("Changed model was deserialised"))
    monkeypatch.setattr(study, "predict_models", lambda *args: pytest.fail("Changed model made predictions"))
    with pytest.raises(ValueError, match="Fitted models differ"):
        study.test_stage({}, [], args)
    assert not (args.output / "test_predictions.csv").exists()
    assert not (args.output / "evaluation.json").exists()
