"""Replay timestamped class probabilities and audit commands against recorded events.

The controller never receives event labels. It neither fits a policy nor claims
that replay outcomes establish prospective control with a person or a device.
"""

import math
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Mapping
from numbers import Integral
from statistics import median

POLICY_FIELDS = {
    "enter_threshold",
    "release_threshold",
    "hold_seconds",
    "cooldown_seconds",
    "max_gap_seconds",
}
PREDICTION_FIELDS = {"participant", "session", "time_s", "left_probability", "right_probability"}
EVENT_FIELDS = {"participant", "session", "event_id", "start_s", "end_s", "state"}
STATES = {"idle", "left", "right", "other", "unknown"}


def _identifier(value, name):
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise ValueError(f"{name}: use a string or integer identifier")
    value = str(value)
    if (
        not value
        or len(value) > 128
        or value != value.strip()
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise ValueError(f"{name}: use 1 to 128 characters without surrounding spaces or control characters")
    return value


def _number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name}: a finite number is required")
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name}: a finite number is required") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name}: a finite number is required")
    return value


def validate_policy(policy):
    """Require an explicit fixed policy; no selection or optimisation happens here."""
    if not isinstance(policy, Mapping) or set(policy) != POLICY_FIELDS:
        raise ValueError(f"Policy must contain exactly {sorted(POLICY_FIELDS)}")
    policy = {key: _number(policy[key], key) for key in sorted(POLICY_FIELDS)}
    if not 0.5 <= policy["enter_threshold"] <= 1:
        raise ValueError("enter_threshold must be between 0.5 and 1")
    if not 0 <= policy["release_threshold"] < policy["enter_threshold"]:
        raise ValueError("release_threshold must be nonnegative and below enter_threshold")
    if policy["hold_seconds"] < 0 or policy["cooldown_seconds"] < 0 or policy["max_gap_seconds"] <= 0:
        raise ValueError("Hold and cooldown must be nonnegative; max_gap_seconds must be positive")
    return policy


def _predictions(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Predictions must be a nonempty list")
    cleaned, last = [], {}
    for index, raw in enumerate(rows, 1):
        if not isinstance(raw, Mapping) or not PREDICTION_FIELDS <= raw.keys():
            raise ValueError(f"Prediction {index}: required fields are {sorted(PREDICTION_FIELDS)}")
        row = {key: _identifier(raw[key], key) for key in ("participant", "session")}
        for field in ("time_s", "left_probability", "right_probability"):
            row[field] = _number(raw[field], f"Prediction {index} {field}")
        if row["time_s"] < 0:
            raise ValueError(f"Prediction {index}: time_s must be nonnegative")
        left, right = row["left_probability"], row["right_probability"]
        if not (0 <= left <= 1 and 0 <= right <= 1) or left + right > 1 + 1e-9:
            raise ValueError(f"Prediction {index}: probabilities must lie in [0, 1] and sum to at most 1")
        key = (row["participant"], row["session"])
        if key in last and row["time_s"] <= last[key]:
            raise ValueError(f"Prediction {index}: timestamps must increase strictly within each session")
        last[key] = row["time_s"]
        cleaned.append(row)
    return cleaned


def _later(first, second):
    """Compare clock boundaries allowing only two units of floating point roundoff."""
    tolerance = 2 * max(math.ulp(first), math.ulp(second))
    return first > second and not math.isclose(first, second, rel_tol=0, abs_tol=tolerance)


def _simulate(rows, policy):
    commands, gaps, states = [], [], {}
    for row in rows:
        key = (row["participant"], row["session"])
        state = states.setdefault(
            key, {"last": None, "candidate": None, "since": None, "latched": None, "cooldown_until": 0.0}
        )
        time = row["time_s"]
        if state["last"] is not None and _later(time, state["last"] + policy["max_gap_seconds"]):
            gaps.append(
                {
                    "participant": key[0],
                    "session": key[1],
                    "start_s": state["last"],
                    "end_s": time,
                    "duration_s": time - state["last"],
                }
            )
            state.update(candidate=None, since=None, latched=None)
        state["last"] = time
        # Release refers to the previously emitted direction, even if its rival now wins.
        if state["latched"] is not None:
            if row[state["latched"] + "_probability"] <= policy["release_threshold"]:
                state["latched"] = None
            else:
                continue
        # A qualifying row inside cooldown cannot contribute towards the next hold.
        if _later(state["cooldown_until"], time):
            state.update(candidate=None, since=None)
            continue
        left, right = row["left_probability"], row["right_probability"]
        direction = "left" if left > right else "right" if right > left else None
        if direction is None or row[direction + "_probability"] < policy["enter_threshold"]:
            state.update(candidate=None, since=None)
            continue
        if direction != state["candidate"]:
            state.update(candidate=direction, since=time)
        if not _later(state["since"] + policy["hold_seconds"], time):
            commands.append(
                {
                    "participant": key[0],
                    "session": key[1],
                    "time_s": time,
                    "direction": direction,
                    "confidence": row[direction + "_probability"],
                }
            )
            state.update(
                latched=direction,
                candidate=None,
                since=None,
                cooldown_until=time + policy["cooldown_seconds"],
            )
    return {"policy": policy, "commands": commands, "gaps": gaps}


def simulate_commands(predictions, policy):
    """Apply a fixed policy to probabilities, without reading labels or future rows.

    A unique winning direction must reach the entry threshold on consecutive
    observed rows for the hold duration. Emission latches that direction until
    its own probability reaches or falls below the release threshold. Cooldown
    discards candidates, so a fresh hold starts with the first eligible row.

    A gap greater than max_gap_seconds resets the candidate and latch while
    preserving any cooldown deadline. A gap covers the whole interval between
    the two observations; the function does not infer a nominal sample rate.
    Session boundaries reset all controller state. No missing row is filled.
    Clock comparisons allow two floating point units of roundoff at a boundary;
    this prevents decimal sampling intervals from creating artificial gaps.
    """
    return _simulate(_predictions(predictions), validate_policy(policy))


def _events(rows, predictions):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Events must be a nonempty list covering every prediction session")
    cleaned, groups, identifiers = [], defaultdict(list), set()
    for index, raw in enumerate(rows, 1):
        if not isinstance(raw, Mapping) or not EVENT_FIELDS <= raw.keys():
            raise ValueError(f"Event {index}: required fields are {sorted(EVENT_FIELDS)}")
        row = {key: _identifier(raw[key], key) for key in ("participant", "session", "event_id")}
        row.update(start_s=_number(raw["start_s"], "start_s"), end_s=_number(raw["end_s"], "end_s"))
        if not isinstance(raw["state"], str) or raw["state"] not in STATES:
            raise ValueError(f"Event {index}: state must be one of {sorted(STATES)}")
        row["state"] = raw["state"]
        if not 0 <= row["start_s"] < row["end_s"]:
            raise ValueError(f"Event {index}: require 0 <= start_s < end_s")
        key = (row["participant"], row["session"])
        identifier = (*key, row["event_id"])
        if identifier in identifiers:
            raise ValueError(f"Event {index}: event_id must be unique within a session")
        identifiers.add(identifier)
        previous_end = groups[key][-1]["end_s"] if groups[key] else 0.0
        if row["start_s"] != previous_end:
            raise ValueError(f"Event {index}: intervals must be ordered and exactly contiguous from zero")
        groups[key].append(row)
        cleaned.append(row)
    if set(groups) != {(r["participant"], r["session"]) for r in predictions}:
        raise ValueError("Events and predictions must contain the same participant and session pairs")
    for row in predictions:
        key = (row["participant"], row["session"])
        if row["time_s"] >= groups[key][-1]["end_s"]:
            raise ValueError("Every prediction must fall within the session's half open event intervals")
    return cleaned, groups


def _grade(commands, events, groups):
    results = []
    lookup = {}
    for event in events:
        result = dict(
            event,
            command_count=0,
            correct_commands=0,
            wrong_hand_commands=0,
            repeated_correct_commands=0,
            first_command_correct=None,
            first_correct_latency_seconds=None,
        )
        results.append(result)
        lookup[(event["participant"], event["session"], event["event_id"])] = result
    starts = {key: [e["start_s"] for e in group] for key, group in groups.items()}
    graded = []
    for command in commands:
        key = (command["participant"], command["session"])
        event = groups[key][bisect_right(starts[key], command["time_s"]) - 1]
        result = lookup[(*key, event["event_id"])]
        if event["state"] in {"left", "right"}:
            correct = command["direction"] == event["state"]
            outcome = "correct" if correct else "wrong_hand"
            if result["command_count"] == 0:
                result["first_command_correct"] = correct
            if correct:
                if result["correct_commands"]:
                    result["repeated_correct_commands"] += 1
                else:
                    result["first_correct_latency_seconds"] = command["time_s"] - event["start_s"]
                result["correct_commands"] += 1
            else:
                result["wrong_hand_commands"] += 1
        else:
            outcome = event["state"]
        result["command_count"] += 1
        graded.append(dict(command, event_id=event["event_id"], state=event["state"], outcome=outcome))
    return graded, results


def _metrics(rows, events, commands, gaps):
    cues = [e for e in events if e["state"] in {"left", "right"}]
    latencies = [e["first_correct_latency_seconds"] for e in cues if e["correct_commands"]]
    idle_seconds = sum(e["end_s"] - e["start_s"] for e in events if e["state"] == "idle")
    counts = {
        outcome: sum(c["outcome"] == outcome for c in commands)
        for outcome in ("correct", "wrong_hand", "idle", "other", "unknown")
    }
    by_session, by_event = defaultdict(list), defaultdict(list)
    for row in rows:
        by_session[(row["participant"], row["session"])].append(row)
    for event in events:
        by_event[(event["participant"], event["session"])].append(event)
    spans = sum(group[-1]["time_s"] - group[0]["time_s"] for group in by_session.values())
    gap_seconds = sum(gap["duration_s"] for gap in gaps)
    idle_gaps = sum(
        max(0.0, min(gap["end_s"], event["end_s"]) - max(gap["start_s"], event["start_s"]))
        for gap in gaps
        for event in by_event[(gap["participant"], gap["session"])]
        if event["state"] == "idle"
    )
    annotated = sum(e["end_s"] - e["start_s"] for e in events)
    return {
        "prediction_rows": len(rows),
        "command_count": len(commands),
        "cues": len(cues),
        "cues_with_correct": len(latencies),
        "cues_first_command_correct": sum(e["first_command_correct"] is True for e in cues),
        "missed_cues": len(cues) - len(latencies),
        "cues_without_commands": sum(e["command_count"] == 0 for e in cues),
        "median_first_correct_latency_seconds": median(latencies) if latencies else None,
        "first_correct_latencies_seconds": latencies,
        "latency_observations": len(latencies),
        "correct_commands": counts["correct"],
        "wrong_hand_commands": counts["wrong_hand"],
        "repeated_correct_commands": sum(e["repeated_correct_commands"] for e in cues),
        "idle_commands": counts["idle"],
        "other_commands": counts["other"],
        "unknown_commands": counts["unknown"],
        "idle_seconds": idle_seconds,
        "idle_commands_per_minute": 60 * counts["idle"] / idle_seconds if idle_seconds else None,
        "annotated_seconds": annotated,
        "observed_span_seconds": spans,
        "evaluated_span_seconds": max(0.0, spans - gap_seconds),
        "gap_seconds": gap_seconds,
        "idle_gap_seconds": idle_gaps,
        "unobserved_edge_seconds": max(0.0, annotated - spans),
    }


def build_command_audit(predictions, events, policy):
    """Grade replay commands against complete, disjoint event intervals.

    Labels are used only after simulation. Each session must have contiguous
    intervals from zero, with explicit idle, other or unknown states when neither
    hand is cued. Intervals include their start and exclude their end. Median
    latency describes only cues receiving a correct command; missed cues remain
    in the counts and are never assigned zero latency.

    Idle commands per minute uses all annotated idle seconds, including gaps and
    the edges outside the prediction span. Reported gaps and observation spans
    make that denominator inspectable; they are not proof of continuous sensing.
    Validation and test partitions must be audited separately by the caller.
    """
    rows = _predictions(predictions)
    event_rows, groups = _events(events, rows)
    replay = _simulate(rows, validate_policy(policy))
    commands, results = _grade(replay["commands"], event_rows, groups)

    def metrics_for(participant, session=None):
        def matches(row):
            return row["participant"] == participant and (session is None or row["session"] == session)

        return _metrics(
            *[[row for row in items if matches(row)] for items in (rows, results, commands, replay["gaps"])]
        )

    return {
        "schema_version": "1.0",
        "kind": "command_audit",
        "policy": replay["policy"],
        "summary": _metrics(rows, results, commands, replay["gaps"]),
        "participants": [
            dict(participant=participant, **metrics_for(participant))
            for participant in sorted({row["participant"] for row in rows})
        ],
        "sessions": [
            dict(participant=participant, session=session, **metrics_for(participant, session))
            for participant, session in sorted(groups)
        ],
        "commands": commands,
        "gaps": replay["gaps"],
        "event_results": results,
    }
