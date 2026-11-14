"""Hand worked controller traces and event audits with known command outcomes."""

from copy import deepcopy

import pytest

from intentlab.commands import build_command_audit, simulate_commands

POLICY = {
    "enter_threshold": 0.8,
    "release_threshold": 0.4,
    "hold_seconds": 0.5,
    "cooldown_seconds": 0.5,
    "max_gap_seconds": 1,
}


def prediction(time, left=0.9, right=0.1, participant="p1", session="s1", **extra):
    return dict(
        participant=participant,
        session=session,
        time_s=time,
        left_probability=left,
        right_probability=right,
        **extra,
    )


def event(start, end, state="left", event_id="e1", participant="p1", session="s1"):
    return dict(
        participant=participant, session=session, event_id=event_id, start_s=start, end_s=end, state=state
    )


def policy(**changes):
    return dict(POLICY, **changes)


def command_times(rows, rule=None):
    return [row["time_s"] for row in simulate_commands(rows, rule or POLICY)["commands"]]


def test_hold_latch_and_release_emit_one_command_per_activation():
    rows = [
        prediction(0),
        prediction(0.25),
        prediction(0.5),
        prediction(1),
        prediction(1.25, 0.3, 0.3),
        prediction(1.5),
        prediction(2),
    ]
    assert command_times(rows) == [0.5, 2]


def test_release_threshold_is_inclusive_and_entry_is_inclusive():
    rows = [
        prediction(0, 0.8, 0.1),
        prediction(0.25, 0.41, 0.1),
        prediction(0.5, 0.4, 0.1),
        prediction(0.75, 0.8, 0.1),
    ]
    assert command_times(rows, policy(hold_seconds=0, cooldown_seconds=0)) == [0, 0.75]


def test_hold_restarts_after_subthreshold_row_or_direction_change():
    rows = [
        prediction(0),
        prediction(0.25, 0.7, 0.1),
        prediction(0.5),
        prediction(0.75, 0.1, 0.9),
        prediction(1, 0.1, 0.9),
        prediction(1.25, 0.1, 0.9),
    ]
    result = simulate_commands(rows, POLICY)
    assert [(r["time_s"], r["direction"]) for r in result["commands"]] == [(1.25, "right")]


def test_ties_never_emit_even_at_the_entry_threshold():
    rows = [prediction(0, 0.5, 0.5), prediction(0.5, 0.5, 0.5), prediction(1, 0.6, 0.4)]
    assert command_times(rows, policy(enter_threshold=0.5, hold_seconds=0)) == [1]


def test_cooldown_discards_qualifying_rows_before_a_fresh_hold():
    rows = [
        prediction(0),
        prediction(0.5),
        prediction(1, 0.1, 0.1),
        prediction(1.25),
        prediction(1.5),
        prediction(1.75),
        prediction(2),
    ]
    assert command_times(rows, policy(cooldown_seconds=1)) == [0.5, 2]


def test_opposite_winner_can_release_the_emitted_direction():
    rows = [prediction(0), prediction(0.5, 0.1, 0.9)]
    result = simulate_commands(rows, policy(hold_seconds=0, cooldown_seconds=0))
    assert [r["direction"] for r in result["commands"]] == ["left", "right"]


def test_gap_resets_hold_without_interpolation_and_equality_is_not_a_gap():
    result = simulate_commands([prediction(0), prediction(2), prediction(3)], policy(hold_seconds=1))
    assert [row["time_s"] for row in result["commands"]] == [3]
    assert result["gaps"] == [dict(participant="p1", session="s1", start_s=0, end_s=2, duration_s=2)]


def test_gap_resets_latch_but_preserves_cooldown_deadline():
    rows = [prediction(0), prediction(2), prediction(5)]
    assert command_times(rows, policy(hold_seconds=0, cooldown_seconds=5)) == [0, 5]


def test_decimal_clock_roundoff_does_not_create_a_gap_or_delay_hold():
    rows = [prediction(0.1), prediction(0.2), prediction(0.3), prediction(0.4)]
    result = simulate_commands(rows, policy(hold_seconds=0.2, max_gap_seconds=0.1))
    assert result["gaps"] == []
    assert [r["time_s"] for r in result["commands"]] == [0.3]


def test_interleaved_sessions_have_independent_state_and_integer_ids_are_strings():
    rows = [
        prediction(0, participant=1, session=1),
        prediction(0, participant=2, session=1),
        prediction(0.5, participant=1, session=1),
        prediction(0.5, participant=2, session=1),
        prediction(0, participant=1, session=2),
        prediction(0.5, participant=1, session=2),
    ]
    result = simulate_commands(rows, POLICY)
    assert [(r["participant"], r["session"]) for r in result["commands"]] == [
        ("1", "1"),
        ("2", "1"),
        ("1", "2"),
    ]
    assert result["gaps"] == []


def test_labels_do_not_change_commands_and_inputs_are_not_modified():
    rows = [prediction(0, label="left"), prediction(0.5, label="right")]
    original = deepcopy(rows)
    expected = simulate_commands(rows, POLICY)
    assert rows == original
    rows[0]["label"] = "unknown"
    rows[1]["label"] = "idle"
    assert simulate_commands(rows, POLICY) == expected


def test_future_probabilities_do_not_change_earlier_commands():
    rows = [prediction(0), prediction(0.5), prediction(1, 0.1, 0.9), prediction(1.5, 0.1, 0.9)]
    prefix = simulate_commands(rows[:2], POLICY)
    later = simulate_commands(rows, POLICY)
    assert [r for r in later["commands"] if r["time_s"] <= 0.5] == prefix["commands"]


def test_hand_worked_audit_counts_wrong_first_repeat_missed_cue_and_boundary():
    rows = [
        prediction(0.5),
        prediction(1, 0.1, 0.9),
        prediction(1.25),
        prediction(1.5, 0.1, 0.1),
        prediction(1.75),
        prediction(2, 0.1, 0.9),
        prediction(3.5, 0.5, 0.5),
    ]
    events = [
        event(0, 1, "idle", "idle"),
        event(1, 2, "left", "left"),
        event(2, 3, "right", "right"),
        event(3, 4, "left", "missed"),
    ]
    result = build_command_audit(rows, events, policy(hold_seconds=0, cooldown_seconds=0))
    summary = result["summary"]
    assert result["schema_version"] == "1.0" and result["kind"] == "command_audit"
    assert summary["command_count"] == 5
    assert summary["cues"] == 3 and summary["cues_with_correct"] == 2
    assert summary["cues_first_command_correct"] == 1
    assert summary["missed_cues"] == 1 and summary["cues_without_commands"] == 1
    assert summary["correct_commands"] == 3 and summary["wrong_hand_commands"] == 1
    assert summary["repeated_correct_commands"] == 1
    assert summary["idle_commands"] == 1 and summary["idle_seconds"] == 1
    assert summary["idle_commands_per_minute"] == 60
    assert summary["first_correct_latencies_seconds"] == [0.25, 0]
    assert summary["latency_observations"] == 2
    assert summary["median_first_correct_latency_seconds"] == 0.125
    assert result["event_results"][1]["first_command_correct"] is False
    assert result["event_results"][2]["first_correct_latency_seconds"] == 0
    assert result["event_results"][3]["first_correct_latency_seconds"] is None
    assert result["commands"][1]["event_id"] == "left"  # Exactly t=1 belongs to the new interval.


def test_other_unknown_and_idle_are_distinct_and_labels_do_not_reset_controller():
    rows = [prediction(0), prediction(1), prediction(2, 0.1, 0.9), prediction(3), prediction(4, 0.1, 0.9)]
    events = [
        event(0, 1, "left", "l"),
        event(1, 2, "right", "r"),
        event(2, 3, "other", "o"),
        event(3, 4, "unknown", "u"),
        event(4, 5, "idle", "i"),
    ]
    result = build_command_audit(rows, events, policy(hold_seconds=0, cooldown_seconds=0))
    summary = result["summary"]
    assert summary["other_commands"] == summary["unknown_commands"] == summary["idle_commands"] == 1
    assert result["event_results"][1]["command_count"] == 0  # Cue boundaries do not reset a latch.
    assert summary["missed_cues"] == 1


def test_zero_command_participants_and_absent_idle_have_explicit_denominators():
    rows = [prediction(0, 0.2, 0.2), prediction(1, 0.2, 0.2), prediction(0, participant="p2")]
    events = [event(0, 2), event(0, 2, participant="p2")]
    result = build_command_audit(rows, events, policy(hold_seconds=0))
    assert len(result["participants"]) == 2 and len(result["sessions"]) == 2
    zero = result["participants"][0]
    assert zero["participant"] == "p1" and zero["command_count"] == 0
    assert zero["cues_without_commands"] == zero["missed_cues"] == 1
    assert zero["median_first_correct_latency_seconds"] is None
    assert zero["latency_observations"] == 0 and zero["first_correct_latencies_seconds"] == []
    assert zero["idle_commands_per_minute"] is None
    assert result["summary"]["latency_observations"] == 1


def test_idle_denominator_includes_gaps_and_edges_but_reports_both():
    result = build_command_audit(
        [prediction(2), prediction(5), prediction(6)],
        [event(0, 10, "idle")],
        policy(hold_seconds=0, cooldown_seconds=0),
    )
    summary = result["summary"]
    assert summary["idle_seconds"] == summary["annotated_seconds"] == 10
    assert summary["gap_seconds"] == summary["idle_gap_seconds"] == 3
    assert summary["observed_span_seconds"] == 4 and summary["evaluated_span_seconds"] == 1
    assert summary["unobserved_edge_seconds"] == 6
    assert summary["idle_commands"] == 2 and summary["idle_commands_per_minute"] == 12


@pytest.mark.parametrize("times", [[0, 0], [1, 0], [-1], [float("nan")], [float("inf")]])
def test_invalid_timestamps_are_rejected_without_silently_sorting(times):
    with pytest.raises(ValueError):
        simulate_commands([prediction(t) for t in times], POLICY)


@pytest.mark.parametrize(
    ("left", "right"), [(-0.1, 0.9), (1.1, 0), (0.8, 0.3), (float("nan"), 0), (0, float("inf")), (True, 0)]
)
def test_invalid_probabilities(left, right):
    with pytest.raises(ValueError):
        simulate_commands([prediction(0, left, right)], POLICY)


@pytest.mark.parametrize("bad_id", ["", "x" * 129, " leading", "trailing ", "line\nbreak", True, None])
def test_invalid_identifiers(bad_id):
    with pytest.raises(ValueError):
        simulate_commands([prediction(0, participant=bad_id)], POLICY)


@pytest.mark.parametrize(
    "changes",
    [
        {"enter_threshold": 0.49},
        {"enter_threshold": 1.1},
        {"release_threshold": 0.8},
        {"release_threshold": -0.1},
        {"hold_seconds": -1},
        {"cooldown_seconds": -1},
        {"max_gap_seconds": 0},
        {"max_gap_seconds": float("nan")},
        {"hold_seconds": True},
        {"extra": 1},
    ],
)
def test_invalid_policy(changes):
    with pytest.raises(ValueError):
        simulate_commands([prediction(0)], policy(**changes))


def test_missing_policy_fields_empty_predictions_and_missing_columns():
    for rows, rule in [([], POLICY), ([{"participant": "p1"}], POLICY), ([prediction(0)], {})]:
        with pytest.raises(ValueError):
            simulate_commands(rows, rule)


@pytest.mark.parametrize(
    "events",
    [
        [],
        [event(0.1, 2)],
        [event(0, 1), event(1.1, 2, event_id="e2")],
        [event(0, 1), event(0.9, 2, event_id="e2")],
        [event(0, 1), event(1, 2)],
        [event(0, 0)],
        [event(0, 2, "rest")],
        [event(0, float("inf"))],
        [event(0, 2, participant="p2")],
        [event(0, 2), event(0, 2, participant="extra")],
    ],
)
def test_invalid_event_coverage_or_metadata(events):
    with pytest.raises(ValueError):
        build_command_audit([prediction(0.5)], events, POLICY)


def test_prediction_at_final_event_end_is_rejected():
    with pytest.raises(ValueError, match="half open"):
        build_command_audit([prediction(2)], [event(0, 2)], POLICY)


def test_grading_does_not_mutate_input_records():
    rows = [prediction(0), prediction(0.5)]
    events = [event(0, 1)]
    before = deepcopy((rows, events, POLICY))
    build_command_audit(rows, events, POLICY)
    assert (rows, events, POLICY) == before
