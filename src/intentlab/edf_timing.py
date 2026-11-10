"""Read-only timing gate for the continuous EDF+ inputs supported by IntentLab.

This checks the stored record timeline, not the acquisition device's clock or
whether samples were lost before export. It neither repairs gaps nor changes EEG
bytes. EDF+ timekeeping is specified at https://www.edfplus.info/specs/edfplus.html
sections 2.1.1 and 2.2.4.
"""

import math
import os
import re
from dataclasses import dataclass
from decimal import Context, Decimal, InvalidOperation, localcontext
from pathlib import Path

_INTEGER = re.compile(rb"[0-9]+")
_NUMBER = re.compile(rb"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[Ee][+-]?[0-9]+)?")
_ONSET = re.compile(rb"[+-][0-9]+(?:\.[0-9]+)?")


class EdfTimingError(ValueError):
    """An EDF file cannot support IntentLab's current zero-origin sample indexing."""


@dataclass(frozen=True)
class EdfTimingAudit:
    file: str
    records: int
    record_duration_seconds: float
    duration_seconds: float
    first_onset_seconds: float = 0.0
    format: str = "EDF+C"


def _positive_integer(value: bytes, name: str, path: Path) -> int:
    value = value.strip(b" ")
    if not _INTEGER.fullmatch(value) or int(value) <= 0:
        raise EdfTimingError(f"{path}: {name} must be a known positive integer")
    return int(value)


def _record_duration(value: bytes, path: Path) -> Decimal:
    value = value.strip(b" ")
    if not _NUMBER.fullmatch(value):
        raise EdfTimingError(f"{path}: record duration must be finite and positive")
    try:
        duration = Decimal(value.decode("ascii"))
    except InvalidOperation as exc:
        raise EdfTimingError(f"{path}: invalid record duration") from exc
    if not duration.is_finite() or duration <= 0 or not 0 < float(duration) < math.inf:
        raise EdfTimingError(f"{path}: record duration must be finite and positive")
    return duration


def _first_onset(data: bytes, path: Path, record: int) -> Decimal:
    location = f"{path}: record {record}"
    end = data.find(b"\x00")
    if end < 0:
        raise EdfTimingError(f"{location}: first timekeeping TAL has no NUL terminator")
    parts = data[:end].split(b"\x14")
    if len(parts) < 3 or parts[1] != b"" or parts[-1] != b"" or not _ONSET.fullmatch(parts[0]):
        raise EdfTimingError(
            f"{location}: malformed first timekeeping TAL; expected a signed onset, "
            "an empty first annotation and a terminating separator"
        )
    # Extra event annotations can share this TAL, following its empty timekeeping
    # annotation. Their text may be UTF-8, but not contain other TAL control bytes.
    for annotation in parts[2:-1]:
        try:
            annotation.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EdfTimingError(f"{location}: first TAL contains invalid UTF-8") from exc
        if any(byte < 32 and byte not in (9, 10, 13) for byte in annotation):
            raise EdfTimingError(f"{location}: first TAL contains an invalid control byte")
    return Decimal(parts[0].decode("ascii"))


def audit_edf_timing(path: Path) -> EdfTimingAudit:
    """Require complete, contiguous EDF+C records with a zero initial onset.

    Only the first EDF Annotations channel provides record timekeeping. Onsets
    are compared as exact decimals, without rounding them onto the sample grid.
    A valid nonzero EDF+ initial offset is unsupported by the existing extraction
    code and is rejected explicitly. Ordinary EDF and EDF+D are also unsupported;
    the function never inserts samples, interpolates or compresses a timeline.

    This is a timing/structure check, not a general-purpose EDF validator or an
    audit of every event annotation. Missing or unreadable files raise OSError.
    """
    path = Path(path)
    with path.open("rb") as stream:
        file_bytes = os.fstat(stream.fileno()).st_size
        header = stream.read(256)
        if len(header) != 256:
            raise EdfTimingError(f"{path}: truncated fixed EDF header")
        if header[:8] != b"0       ":
            raise EdfTimingError(f"{path}: unsupported EDF version")
        if any(byte < 32 or byte > 126 for byte in header):
            raise EdfTimingError(f"{path}: EDF header must contain printable ASCII")
        if not header[192:236].startswith(b"EDF+C"):
            raise EdfTimingError(
                f"{path}: only EDF+C is supported; ordinary EDF and EDF+D are rejected without gap repair"
            )
        signals = _positive_integer(header[252:256], "number of signals", path)
        header_bytes = _positive_integer(header[184:192], "header byte count", path)
        if header_bytes != 256 * (signals + 1):
            raise EdfTimingError(f"{path}: header byte count does not match number of signals")
        records = _positive_integer(header[236:244], "record count", path)
        duration = _record_duration(header[244:252], path)
        signal_headers = stream.read(header_bytes - 256)
        if len(signal_headers) != header_bytes - 256:
            raise EdfTimingError(f"{path}: truncated signal headers")
        if any(byte < 32 or byte > 126 for byte in signal_headers):
            raise EdfTimingError(f"{path}: signal headers must contain printable ASCII")
        labels = [signal_headers[16 * i : 16 * (i + 1)].rstrip(b" ") for i in range(signals)]
        try:
            annotation_index = labels.index(b"EDF Annotations")
        except ValueError as exc:
            raise EdfTimingError(f"{path}: missing EDF Annotations timekeeping channel") from exc
        samples = [
            _positive_integer(
                signal_headers[216 * signals + 8 * i : 216 * signals + 8 * (i + 1)],
                f"samples per record for signal {i + 1}",
                path,
            )
            for i in range(signals)
        ]
        record_bytes = 2 * sum(samples)
        if record_bytes > 61440:
            raise EdfTimingError(f"{path}: record byte count exceeds the EDF+ 61440-byte limit")
        expected_bytes = header_bytes + records * record_bytes
        if file_bytes != expected_bytes:
            raise EdfTimingError(
                f"{path}: declared file size {expected_bytes} bytes differs from actual {file_bytes} bytes"
            )
        annotation_offset = 2 * sum(samples[:annotation_index])
        annotation_bytes = 2 * samples[annotation_index]
        # Header duration has at most eight characters. This independent context
        # keeps duration * record_index exact even if a caller changed Decimal's
        # process context; arbitrary-precision TAL onsets are never rounded.
        precision = len(duration.as_tuple().digits) + len(str(records)) + 2
        with localcontext(Context(prec=max(28, precision))):
            total_duration = duration * records
            if not 0 < float(total_duration) < math.inf:
                raise EdfTimingError(f"{path}: total duration cannot be represented as finite seconds")
            previous = Decimal(0)
            for index in range(records):
                stream.seek(header_bytes + index * record_bytes + annotation_offset)
                data = stream.read(annotation_bytes)
                if len(data) != annotation_bytes:
                    raise EdfTimingError(f"{path}: record {index + 1}: truncated annotation channel")
                onset = _first_onset(data, path, index + 1)
                expected = duration * index
                if index == 0 and onset != 0:
                    raise EdfTimingError(
                        f"{path}: record 1: nonzero initial onset {onset}; "
                        "IntentLab requires a zero origin and does not shift timestamps"
                    )
                if onset != expected:
                    if onset == previous:
                        problem = "duplicate onset"
                    elif onset < previous:
                        problem = "out-of-order onset"
                    elif onset < expected:
                        problem = "overlapping records"
                    else:
                        problem = "gap between records"
                    raise EdfTimingError(
                        f"{path}: record {index + 1}: {problem}; onset {onset}, expected {expected}"
                    )
                previous = onset
    return EdfTimingAudit(
        file=path.name,
        records=records,
        record_duration_seconds=float(duration),
        duration_seconds=float(total_duration),
    )
