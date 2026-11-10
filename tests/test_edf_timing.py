"""Small EDF+ fixtures exercise clock failures without the local raw-file cache."""

import hashlib
import json
import sys
from decimal import Inexact, localcontext

import pytest

from intentlab.edf_timing import EdfTimingError, audit_edf_timing
from scripts.audit_edf_timing import audit_cached_recordings, main


def make_edf(
    path,
    onsets=("+0", "+1", "+2"),
    *,
    duration="1",
    format="EDF+C",
    first_tal=None,
    annotations=True,
    second_annotation=False,
):
    """Write fixed-duration records with one EEG signal and optional annotation channels."""
    labels = ["C3"] + (["EDF Annotations"] if annotations else [])
    if second_annotation:
        labels.append("EDF Annotations")
    samples = [160] + [64] * (len(labels) - 1)
    ns = len(labels)

    def field(value, width):
        return str(value).encode("ascii").ljust(width, b" ")

    header = b"".join(
        [
            field("0", 8),
            field("X", 80),
            field("X", 80),
            field("01.01.00", 8),
            field("00.00.00", 8),
            field(256 * (ns + 1), 8),
            field(format, 44),
            field(len(onsets), 8),
            field(duration, 8),
            field(ns, 4),
        ]
    )
    for values, width in [
        (labels, 16),
        ([""] * ns, 80),
        (["uV"] * ns, 8),
        ([-100] * ns, 8),
        ([100] * ns, 8),
        ([-32768] * ns, 8),
        ([32767] * ns, 8),
        ([""] * ns, 80),
        (samples, 8),
        ([""] * ns, 32),
    ]:
        header += b"".join(field(v, width) for v in values)
    payload = bytearray(header)
    for i, onset in enumerate(onsets):
        payload.extend(b"\x00" * 320)
        if annotations:
            tal = onset.encode("ascii") + b"\x14\x14\x00"
            if first_tal is not None and i == 0:
                tal = first_tal
            payload.extend(tal.ljust(128, b"\x00"))
        if second_annotation:
            # Only the first annotation channel is required to provide the record clock.
            payload.extend(b"\x00" * 128)
    path.write_bytes(payload)
    return path


def replace_header(path, start, width, value):
    data = bytearray(path.read_bytes())
    data[start : start + width] = str(value).encode("ascii").ljust(width, b" ")
    path.write_bytes(data)


def manifest_for(path, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "file": "S001/" + path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                ]
            }
        )
    )
    return manifest


def test_continuous_file_and_read_only_audit(tmp_path):
    path = make_edf(tmp_path / "S001R04.edf")
    before = path.read_bytes()
    result = audit_edf_timing(path)
    assert result.file == path.name
    assert result.records == 3
    assert result.record_duration_seconds == 1
    assert result.duration_seconds == 3
    assert result.first_onset_seconds == 0
    assert result.format == "EDF+C"
    report = audit_cached_recordings(tmp_path, manifest_for(path, tmp_path))
    assert report["files"] == 1 and report["data_records"] == 3
    assert report["checksums_verified"] and report["record_onsets_contiguous"]
    assert path.read_bytes() == before


def test_fractional_record_duration_uses_exact_decimal_arithmetic(tmp_path):
    path = make_edf(tmp_path / "fractional.edf", ("+0.0", "+0.1", "+0.2"), duration="0.1")
    assert audit_edf_timing(path).duration_seconds == pytest.approx(0.3)


def test_timing_checks_ignore_callers_decimal_rounding_context(tmp_path):
    path = make_edf(tmp_path / "precision.edf", ("+0", "+0.12", "+0.24"), duration="0.12")
    with localcontext() as context:
        context.prec = 1
        context.traps[Inexact] = True
        assert audit_edf_timing(path).duration_seconds == pytest.approx(0.36)


def test_timekeeping_tal_can_also_contain_an_event(tmp_path):
    path = make_edf(tmp_path / "event.edf", first_tal=b"+0\x14\x14T0\x14\x00", second_annotation=True)
    assert audit_edf_timing(path).records == 3


@pytest.mark.parametrize(
    ("onsets", "reason"),
    [
        (("+0", "+2", "+3"), "gap between records"),
        (("+0", "+0", "+2"), "duplicate onset"),
        (("+0", "+0.5", "+2"), "overlapping records"),
        (("+0", "-1", "+2"), "out-of-order onset"),
        (("+0", "+1.000001", "+2"), "gap between records"),
        (("+0.5", "+1.5", "+2.5"), "nonzero initial onset"),
    ],
)
def test_rejects_discontinuous_or_offset_clocks(tmp_path, onsets, reason):
    path = make_edf(tmp_path / "gap.edf", onsets)
    with pytest.raises(EdfTimingError, match=reason) as exc:
        audit_edf_timing(path)
    assert "gap.edf: record" in str(exc.value)


@pytest.mark.parametrize("format", ["EDF+D", "", "EDF", "BDF+C"])
def test_rejects_unverifiable_or_discontinuous_formats(tmp_path, format):
    path = make_edf(tmp_path / "unsupported.edf", format=format)
    with pytest.raises(EdfTimingError):
        audit_edf_timing(path)


@pytest.mark.parametrize(
    "tal",
    [
        b"",
        b"+0\x14T0\x14\x00",
        b"0\x14\x14\x00",
        b"+nan\x14\x14\x00",
        b"+inf\x14\x14\x00",
        b"+0\x15" + b"1\x14\x14\x00",
        b"+0\x14\x00",
        b"+0\x14\x14" + b"a" * 122 + b"\x00",  # Event lacks terminating separator.
        b"+0\x14\x14" + b"a" * 124,  # No NUL terminator in the annotation channel.
    ],
)
def test_rejects_missing_or_malformed_timekeeping(tmp_path, tal):
    path = make_edf(tmp_path / "bad-tal.edf", first_tal=tal)
    with pytest.raises(EdfTimingError, match="first timekeeping TAL"):
        audit_edf_timing(path)


def test_cannot_use_second_annotation_channel_to_hide_bad_first_clock(tmp_path):
    path = make_edf(tmp_path / "channels.edf", first_tal=b"", second_annotation=True)
    data = bytearray(path.read_bytes())
    second_start = 256 * 4 + 320 + 128
    data[second_start : second_start + 5] = b"+0\x14\x14\x00"
    path.write_bytes(data)
    with pytest.raises(EdfTimingError):
        audit_edf_timing(path)


def test_missing_annotation_channel(tmp_path):
    path = make_edf(tmp_path / "no-clock.edf", annotations=False)
    with pytest.raises(EdfTimingError):
        audit_edf_timing(path)


@pytest.mark.parametrize(
    ("start", "width", "value"),
    [
        (0, 8, "1"),
        (184, 8, "767"),
        (236, 8, "-1"),
        (236, 8, "0"),
        (236, 8, "1.5"),
        (244, 8, "0"),
        (244, 8, "-1"),
        (244, 8, "NaN"),
        (244, 8, "Infinity"),
        (252, 4, "0"),
        (688, 8, "0"),  # Zero samples for the EEG channel in this two-signal fixture.
    ],
)
def test_invalid_headers(tmp_path, start, width, value):
    path = make_edf(tmp_path / "header.edf")
    replace_header(path, start, width, value)
    with pytest.raises(EdfTimingError):
        audit_edf_timing(path)


@pytest.mark.parametrize("change", ["short_header", "missing_sample", "missing_record", "extra_bytes"])
def test_file_size_disagrees_with_declared_records(tmp_path, change):
    path = make_edf(tmp_path / "size.edf")
    data = path.read_bytes()
    altered = {
        "short_header": data[:200],
        "missing_sample": data[:-2],
        "missing_record": data[:-448],
        "extra_bytes": data + b"\x00\x00",
    }[change]
    path.write_bytes(altered)
    with pytest.raises(EdfTimingError):
        audit_edf_timing(path)


def test_cli_checksum_failure_is_nonzero_without_success_report(tmp_path, capsys):
    path = make_edf(tmp_path / "S001R04.edf")
    manifest = manifest_for(path, tmp_path)
    replace_header(path, 88, 1, "Y")
    with pytest.raises(SystemExit) as exc:
        main(["--raw-dir", str(tmp_path), "--manifest", str(manifest)])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "checksum" in captured.err


def test_cli_accepts_valid_cache_and_rejects_checksummed_gap(tmp_path, capsys):
    path = make_edf(tmp_path / "S001R04.edf")
    manifest = manifest_for(path, tmp_path)
    main(["--raw-dir", str(tmp_path), "--manifest", str(manifest)])
    assert json.loads(capsys.readouterr().out)["data_records"] == 3
    make_edf(path, ("+0", "+2", "+3"))
    manifest_for(path, tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["--raw-dir", str(tmp_path), "--manifest", str(manifest)])
    assert exc.value.code == 1
    assert "S001R04.edf" in capsys.readouterr().err


def test_preparation_aborts_on_timing_failure_before_changing_outputs(tmp_path, monkeypatch):
    from scripts import prepare_data

    raw_dir = tmp_path / "data/raw"
    raw_dir.mkdir(parents=True)
    path = make_edf(raw_dir / "S001R04.edf", ("+0", "+2", "+3"))
    records = [f"S{s:03d}/S{s:03d}R{r:02d}.edf" for s in range(1, 110) for r in (4, 8, 12)]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (raw_dir / "SHA256SUMS.txt").write_text("\n".join(f"{digest}  {rel}" for rel in records))
    derived = tmp_path / "data/derived"
    derived.mkdir()
    sentinel = derived / "manifest.json"
    sentinel.write_text("existing frozen result")
    monkeypatch.setattr(prepare_data, "ROOT", tmp_path)
    monkeypatch.setattr(prepare_data, "acquire", lambda *args: path)
    monkeypatch.setattr(sys, "argv", ["prepare_data.py", "--offline"])
    with pytest.raises(EdfTimingError, match="S001R04.edf"):
        prepare_data.main()
    assert sentinel.read_text() == "existing frozen result"
    assert not (derived / "epochs.npz").exists()
    assert not (tmp_path / "data/demo/epochs.npz").exists()
