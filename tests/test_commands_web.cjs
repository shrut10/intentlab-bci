"use strict";

const assert = require("node:assert/strict");
const {execFileSync} = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const {test} = require("node:test");
const viewer = require("../web/commands.js");
const root = path.resolve(__dirname, "..");

// Build the examples with the production Python auditor rather than reproducing
// its output shape by hand; this catches drift between the CLI and the viewer
const fixtures = JSON.parse(execFileSync(process.env.PYTHON || "python3", ["-c", `
import csv
import json
from pathlib import Path
from intentlab.commands import build_command_audit

folder = Path("examples/command_audit_sample")
with (folder / "predictions.csv").open() as stream:
    predictions = list(csv.DictReader(stream))
with (folder / "events.csv").open() as stream:
    events = list(csv.DictReader(stream))
policy = json.loads((folder / "policy.json").read_text())
sample = build_command_audit(predictions, events, policy)

def row(time, left=0.1, right=0.1, participant="P1"):
    return dict(participant=participant, session="session", time_s=time,
                left_probability=left, right_probability=right)

def event(start, end, state, event_id, participant="P1"):
    return dict(participant=participant, session="session", start_s=start,
                end_s=end, state=state, event_id=event_id)

# A known two second prediction gap overlaps one second of instructed rest
# P2 has no emitted commands and must remain visible in every participant view
trace = [row(0, 0.9), row(0.5, 0.9), row(1), row(3, 0.1, 0.9),
         row(3.5, 0.1, 0.9), row(4)]
annotations = [event(0, 2, "idle", "rest"), event(2, 4, "right", "cue"),
               event(4, 5, "unknown", "transition")]
trace += [row(0, participant="P2"), row(0.5, participant="P2")]
annotations += [event(0, 1, "left", "missed", participant="P2")]
gaps = build_command_audit(trace, annotations, policy)
no_idle = build_command_audit([row(0), row(0.5)], [event(0, 1, "left", "cue")], policy)
print(json.dumps(dict(sample=sample, gaps=gaps, no_idle=no_idle), allow_nan=False))
`], {
  cwd: root,
  env: {...process.env, PYTHONPATH: [path.join(root, "src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter)},
  encoding: "utf8",
}));
const copy = value => structuredClone(value);

function changedAudit(change, source = fixtures.sample) {
  const audit = copy(source);
  change(audit);
  return audit;
}

function makeStudy() {
  const validation = copy(fixtures.sample), testAudit = copy(fixtures.sample);
  for (const field of ["sessions", "commands", "gaps", "event_results"]) {
    for (const row of validation[field]) row.session = "validation";
    for (const row of testAudit[field]) row.session = "test";
  }
  return {
    schema_version: "1.0", kind: "command_study", title: "Test fixture",
    conditions: [{condition: "movement_only", label: "Movement confidence",
      validation_target_met: false, policy: copy(validation.policy), validation, test: testAudit}],
  };
}

function rejects(change, message, source = fixtures.sample) {
  assert.throws(() => viewer.validateAudit(changedAudit(change, source)), message);
}

test("Python generated example remains compatible with the browser viewer", () => {
  const snapshot = JSON.stringify(fixtures.sample);
  const audit = viewer.parseLocalReport(snapshot);
  assert.equal(audit.summary.command_count, 2);
  assert.equal(audit.summary.correct_commands, 1);
  assert.equal(audit.summary.idle_commands, 1);
  assert.equal(audit.summary.idle_commands_per_minute, 60);
  assert.equal(audit.summary.median_first_correct_latency_seconds, 0.5);
  assert.deepEqual(audit.commands.map(row => [row.time_s, row.direction, row.outcome]), [
    [0.5, "left", "correct"], [1.5, "right", "idle"],
  ]);
  viewer.validateAudit(fixtures.sample);
  assert.equal(JSON.stringify(fixtures.sample), snapshot, "Validation must not modify a supplied report");
});

test("zero command participants and undefined rates remain explicit", () => {
  const audit = viewer.validateAudit(fixtures.gaps);
  assert.equal(audit.participants.length, 2);
  const participant = audit.participants.find(row => row.participant === "P2");
  assert.equal(participant.command_count, 0);
  assert.equal(participant.missed_cues, 1);
  assert.equal(participant.median_first_correct_latency_seconds, null);
  assert.equal(participant.latency_observations, 0);
  const noIdle = viewer.validateAudit(fixtures.no_idle);
  assert.equal(noIdle.summary.idle_commands_per_minute, null);
  assert.equal(viewer.timeLabel(noIdle.summary.median_first_correct_latency_seconds), "—");
});

test("local import rejects invalid JSON, other schemas and oversized UTF8 input", () => {
  assert.throws(() => viewer.parseLocalReport("{"), /not valid JSON/);
  for (const value of [null, [], 7, {kind: "command_study", schema_version: "1.0"}]) {
    assert.throws(() => viewer.parseLocalReport(JSON.stringify(value)), /command audit/);
  }
  rejects(audit => { audit.schema_version = "2.0"; }, /schema 1.0/);
  const oversized = JSON.stringify({padding: "é".repeat(viewer.MAX_BYTES / 2)});
  assert(oversized.length < viewer.MAX_BYTES, "Fixture must exercise byte length rather than character length");
  assert.throws(() => viewer.parseLocalReport(oversized), /smaller than 5 MB/);
});

test("numeric and identity checks reject malformed reports", () => {
  for (const value of [NaN, Infinity, -1, true, "1"]) {
    rejects(audit => { audit.summary.idle_seconds = value; }, /counts or exposure/);
  }
  rejects(audit => { audit.summary.command_count = 1.5; }, /counts or exposure/);
  rejects(audit => { audit.commands[0].confidence = 1.1; }, /confidence/);
  rejects(audit => { audit.commands[0].time_s = -1; }, /timing/);
  rejects(audit => { audit.participants[0].participant = " padded"; }, /identifier/);
  rejects(audit => { audit.participants.push(copy(audit.participants[0])); }, /Duplicate participant/);
  rejects(audit => { audit.sessions[0].participant = "missing"; }, /Unknown participant/);
});

test("rates, latency denominators and exposure must agree with the supplied counts", () => {
  rejects(audit => { audit.summary.idle_commands_per_minute = 0; }, /rate does not match/);
  rejects(audit => { audit.summary.median_first_correct_latency_seconds = 0; }, /Median latency/);
  rejects(audit => { audit.summary.latency_observations = 0; }, /Latency counts/);
  rejects(audit => { audit.summary.first_correct_latencies_seconds.push(0.5); }, /Latency counts/);
  rejects(audit => { audit.summary.cues_without_commands = 1; }, /cue counts/);
  rejects(audit => { audit.summary.evaluated_span_seconds += 1; }, /recording exposure/);
  rejects(audit => { audit.summary.idle_commands_per_minute = 0; }, /undefined without idle/, fixtures.no_idle);
  rejects(audit => { audit.summary.median_first_correct_latency_seconds = 0; }, /undefined when no cue/, fixtures.no_idle);
});

test("annotation and command boundaries cannot be silently reassigned", () => {
  rejects(audit => { audit.event_results[0].start_s = 0.1; }, /continuous/);
  rejects(audit => { audit.event_results[1].start_s = 0.9; }, /continuous/);
  rejects(audit => { audit.event_results[0].state = "imagined"; }, /Event intervals/);
  rejects(audit => { audit.commands[0].time_s = 1; }, /timing/);
  rejects(audit => { audit.commands[0].event_id = "missing"; }, /timing/);
  rejects(audit => { audit.commands[0].outcome = "idle"; }, /outcome/);
  rejects(audit => { audit.commands.reverse(); }, /order/);
});

test("event results cannot claim commands or cue outcomes absent from the timeline", () => {
  rejects(audit => {
    audit.event_results[0].command_count = 100;
    audit.event_results[0].correct_commands = 100;
  }, /Event command_count does not match/);
  rejects(audit => { audit.event_results[0].first_command_correct = false; }, /first_command_correct does not match/);
  rejects(audit => { audit.event_results[0].first_correct_latency_seconds = null; }, /first_correct_latency_seconds does not match/);
  rejects(audit => { audit.event_results[1].first_command_correct = true; }, /first_command_correct does not match/);
});

test("consistent looking totals cannot relabel a real idle command as another task", () => {
  rejects(audit => {
    for (const metrics of [audit.summary, ...audit.participants, ...audit.sessions]) {
      metrics.idle_commands = 0;
      metrics.other_commands = 1;
      metrics.idle_commands_per_minute = 0;
    }
  }, /Session idle_commands does not match/);
});

test("an added gap must be reflected in the displayed exposure totals", () => {
  rejects(audit => {
    audit.gaps.push({participant: "SYNTHETIC", session: "example", start_s: 0, end_s: 1, duration_s: 1});
  }, /Session gap_seconds does not match/);
  rejects(audit => {
    for (const metrics of [audit.summary, ...audit.participants, ...audit.sessions]) {
      metrics.first_correct_latencies_seconds = [0.25];
      metrics.median_first_correct_latency_seconds = 0.25;
    }
  }, /Session latencies do not match/);
});

test("adjacent timeline windows include a boundary command only once", () => {
  const key = viewer.sessionKey(fixtures.sample.sessions[0]);
  const before = viewer.windowSlice(fixtures.sample, key, 0, 0.5);
  const after = viewer.windowSlice(fixtures.sample, key, 0.5, 0.5);
  assert.equal(before.commands.length, 0);
  assert.deepEqual(after.commands.map(row => row.time_s), [0.5]);
  const atAnnotationBoundary = viewer.windowSlice(fixtures.sample, key, 1, 0.5);
  assert.deepEqual(atAnnotationBoundary.events.map(row => row.state), ["idle"]);
  assert.equal(viewer.windowSlice(fixtures.sample, key, 100, 1).start, 1);
  assert.equal(viewer.windowSlice(fixtures.sample, key, 0, 120).end, 2);
  assert.throws(() => viewer.windowSlice(fixtures.sample, key, -1, 1), /Invalid time interval/);
  assert.throws(() => viewer.windowSlice(fixtures.sample, key, 0, 0), /Invalid time interval/);
  assert.throws(() => viewer.windowSlice(fixtures.sample, "missing", 0, 1), /no annotation/);
});

test("gaps retain their full exposure and overlap only the visible intervals", () => {
  const audit = viewer.validateAudit(fixtures.gaps), key = viewer.sessionKey(audit.sessions[0]);
  assert.equal(audit.summary.gap_seconds, 2);
  assert.equal(audit.summary.idle_gap_seconds, 1);
  assert.equal(audit.summary.idle_seconds, 2);
  assert.equal(audit.summary.idle_commands_per_minute, 30);
  assert.equal(viewer.windowSlice(audit, key, 0, 1).gaps.length, 0);
  assert.equal(viewer.windowSlice(audit, key, 1, 1).gaps.length, 1);
  assert.equal(viewer.windowSlice(audit, key, 3, 1).gaps.length, 0);
  rejects(value => { value.gaps[0].duration_s = 1; }, /gap interval/, audit);
  rejects(value => { value.gaps[0].end_s = 8; }, /gap interval/, audit);
  rejects(value => { value.gaps.push(copy(value.gaps[0])); }, /gap interval/, audit);
});

test("a published study keeps its policy fixed and its sessions separate", () => {
  assert.equal(viewer.validateStudy(makeStudy()).conditions.length, 1);
  const changedPolicy = makeStudy();
  changedPolicy.conditions[0].test.policy.enter_threshold = 0.8;
  assert.throws(() => viewer.validateStudy(changedPolicy), /test policy must match/);
  const leakedSplit = makeStudy();
  leakedSplit.conditions[0].test = copy(leakedSplit.conditions[0].validation);
  assert.throws(() => viewer.validateStudy(leakedSplit), /sessions must be separate/);
  const duplicateCondition = makeStudy();
  duplicateCondition.conditions.push(copy(duplicateCondition.conditions[0]));
  assert.throws(() => viewer.validateStudy(duplicateCondition), /Invalid study condition/);
  rejects(audit => { audit.policy.release_threshold = audit.policy.enter_threshold; }, /thresholds/);
  rejects(audit => { audit.policy.max_gap_seconds = 0; }, /gap limit/);
  rejects(audit => { audit.policy.unsupported_rule = true; }, /operating rule/);
});

test("displayed times round cleanly at minute boundaries", () => {
  assert.equal(viewer.timeLabel(0), "0:00.0");
  assert.equal(viewer.timeLabel(59.999), "1:00.0");
  assert.equal(viewer.timeLabel(60), "1:00.0");
  assert.equal(viewer.timeLabel(3599.999), "60:00.0");
  for (const value of [null, undefined, NaN, Infinity, -1]) assert.equal(viewer.timeLabel(value), "—");
});

test("policy timings preserve the published fractional gap threshold", () => {
  assert.equal(viewer.policyTimeLabel(0.75), "0.75 s");
  assert.equal(viewer.policyTimeLabel(1.5), "1.5 s");
  assert.equal(viewer.policyTimeLabel(1), "1 s");
  assert.equal(viewer.policyTimeLabel(0.001), "0.001 s");
});

test("published command study remains compatible with its deployed viewer", () => {
  const report = viewer.validateStudy(JSON.parse(fs.readFileSync(path.join(root, "web/commands-study.json"), "utf8")));
  assert.deepEqual(report.conditions.map(row => row.condition), ["movement_only", "rest_gate"]);
  for (const condition of report.conditions) {
    for (const split of ["validation", "test"]) {
      const audit = condition[split];
      assert.equal(audit.participants.length, 4);
      assert.equal(new Set(audit.sessions.map(row => row.participant)).size, 4);
      for (const session of audit.sessions) {
        const window = viewer.windowSlice(audit, viewer.sessionKey(session), 0, 120);
        assert(window.end > window.start);
        assert(window.events.length > 0);
        assert(window.commands.every(row => row.time_s >= window.start && row.time_s < window.end));
      }
    }
  }
});
