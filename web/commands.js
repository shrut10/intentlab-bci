"use strict";

const CommandAudit = (() => {
  const MAX_BYTES = 5 * 1024 * 1024;
  const STATES = new Set(["idle", "left", "right", "other", "unknown"]);
  const COUNT_FIELDS = ["prediction_rows", "command_count", "cues", "cues_with_correct", "cues_first_command_correct", "missed_cues", "cues_without_commands", "latency_observations", "correct_commands", "wrong_hand_commands", "repeated_correct_commands", "idle_commands", "other_commands", "unknown_commands"];
  const TIME_FIELDS = ["idle_seconds", "annotated_seconds", "observed_span_seconds", "evaluated_span_seconds", "gap_seconds", "idle_gap_seconds", "unobserved_edge_seconds"];
  const POLICY_FIELDS = ["enter_threshold", "release_threshold", "hold_seconds", "cooldown_seconds", "max_gap_seconds"];
  const fail = message => { throw new Error(message); };
  const object = value => value !== null && typeof value === "object" && !Array.isArray(value);
  const finite = value => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1e9;
  const count = value => Number.isSafeInteger(value) && value >= 0;
  const near = (a, b) => Math.abs(a - b) <= 1e-7 * Math.max(1, Math.abs(a), Math.abs(b));
  const key = row => JSON.stringify([row.participant, row.session]);
  const eventKey = row => JSON.stringify([row.participant, row.session, row.event_id]);

  function identifier(value) {
    if (typeof value !== "string" || value.length < 1 || value.length > 128 || value.trim() !== value || /[\u0000-\u001f\u007f]/.test(value)) fail("Invalid participant, session or event identifier");
  }

  function array(value, name, allowEmpty = true, limit = 100000) {
    if (!Array.isArray(value) || value.length > limit || (!allowEmpty && !value.length)) fail(`Invalid ${name} list`);
  }

  function validatePolicy(policy) {
    if (!object(policy) || Object.keys(policy).length !== POLICY_FIELDS.length || POLICY_FIELDS.some(field => !finite(policy[field]))) fail("Invalid operating rule");
    if (policy.enter_threshold < .5 || policy.enter_threshold > 1 || policy.release_threshold >= policy.enter_threshold || policy.max_gap_seconds <= 0) fail("Invalid confidence thresholds or gap limit");
    return policy;
  }

  function validateMetrics(metrics) {
    if (!object(metrics) || COUNT_FIELDS.some(field => !count(metrics[field])) || TIME_FIELDS.some(field => !finite(metrics[field]))) fail("Invalid command counts or exposure values");
    array(metrics.first_correct_latencies_seconds, "latency");
    if (metrics.first_correct_latencies_seconds.some(value => !finite(value))) fail("Invalid latency value");
    if (metrics.latency_observations !== metrics.first_correct_latencies_seconds.length || metrics.latency_observations !== metrics.cues_with_correct) fail("Latency counts do not match the contributing cues");
    if (metrics.cues_with_correct + metrics.missed_cues !== metrics.cues || metrics.cues_first_command_correct > metrics.cues_with_correct || metrics.cues_without_commands > metrics.missed_cues || metrics.repeated_correct_commands > metrics.correct_commands) fail("Inconsistent cue counts");
    if (["correct_commands", "wrong_hand_commands", "idle_commands", "other_commands", "unknown_commands"].reduce((sum, field) => sum + metrics[field], 0) !== metrics.command_count) fail("Command outcome counts do not add up");
    if (!metrics.latency_observations) {
      if (metrics.median_first_correct_latency_seconds !== null) fail("Latency must be undefined when no cue receives a correct command");
    } else {
      if (!finite(metrics.median_first_correct_latency_seconds)) fail("Invalid median latency");
      const sorted = metrics.first_correct_latencies_seconds.slice().sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      const median = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
      if (!near(metrics.median_first_correct_latency_seconds, median)) fail("Median latency does not match the supplied observations");
    }
    if (metrics.idle_seconds === 0) {
      if (metrics.idle_commands_per_minute !== null) fail("Idle command rate must be undefined without idle exposure");
    } else if (!finite(metrics.idle_commands_per_minute) || !near(metrics.idle_commands_per_minute, 60 * metrics.idle_commands / metrics.idle_seconds)) fail("Idle command rate does not match its exposure");
    if (metrics.idle_seconds > metrics.annotated_seconds + 1e-7 || metrics.idle_gap_seconds > metrics.gap_seconds + 1e-7 || metrics.gap_seconds > metrics.observed_span_seconds + 1e-7 || !near(metrics.evaluated_span_seconds, metrics.observed_span_seconds - metrics.gap_seconds) || !near(metrics.unobserved_edge_seconds, metrics.annotated_seconds - metrics.observed_span_seconds)) fail("Inconsistent recording exposure");
  }

  function verifyEvidence(data) {
    const group = (rows, keyFor) => {
      const groups = new Map();
      for (const row of rows) {
        const id = keyFor(row);
        if (!groups.has(id)) groups.set(id, []);
        groups.get(id).push(row);
      }
      return groups;
    };
    const commandsByEvent = group(data.commands, eventKey);
    const commandsBySession = group(data.commands, key);
    const eventsBySession = group(data.event_results, key);
    const gapsBySession = group(data.gaps, key);
    const sessionGroups = group(data.sessions, row => row.participant);
    const sameLatency = (actual, expected) => {
      const sortedActual = actual.slice().sort((a, b) => a - b);
      const sortedExpected = expected.slice().sort((a, b) => a - b);
      return sortedActual.length === sortedExpected.length && sortedActual.every((value, index) => near(value, sortedExpected[index]));
    };
    const compare = (actual, expected, name) => {
      for (const [field, value] of Object.entries(expected)) {
        const matches = typeof value === "number" ? typeof actual[field] === "number" && (COUNT_FIELDS.includes(field) ? actual[field] === value : near(actual[field], value)) : actual[field] === value;
        if (!matches) fail(`${name} ${field} does not match the recorded evidence`);
      }
    };

    for (const event of data.event_results) {
      const commands = commandsByEvent.get(eventKey(event)) || [];
      const correct = commands.filter(row => row.outcome === "correct");
      const isCue = event.state === "left" || event.state === "right";
      compare(event, {
        command_count: commands.length,
        correct_commands: correct.length,
        wrong_hand_commands: commands.filter(row => row.outcome === "wrong_hand").length,
        repeated_correct_commands: Math.max(0, correct.length - 1),
        first_command_correct: isCue && commands.length ? commands[0].outcome === "correct" : null,
        first_correct_latency_seconds: correct.length ? correct[0].time_s - event.start_s : null,
      }, "Event");
    }

    for (const session of data.sessions) {
      const id = key(session), events = eventsBySession.get(id), commands = commandsBySession.get(id) || [], gaps = gapsBySession.get(id) || [];
      const cues = events.filter(row => row.state === "left" || row.state === "right");
      const latencies = cues.filter(row => row.correct_commands).map(row => row.first_correct_latency_seconds);
      let eventIndex = 0, idleGapSeconds = 0;
      // The two ordered interval lists are swept once, avoiding quadratic work
      // when a local report contains many annotation intervals and outages
      for (const gap of gaps) {
        while (eventIndex < events.length && events[eventIndex].end_s <= gap.start_s) eventIndex++;
        for (let index = eventIndex; index < events.length && events[index].start_s < gap.end_s; index++) {
          const event = events[index];
          if (event.state === "idle") idleGapSeconds += Math.max(0, Math.min(event.end_s, gap.end_s) - Math.max(event.start_s, gap.start_s));
        }
      }
      compare(session, {
        command_count: commands.length,
        cues: cues.length,
        cues_with_correct: latencies.length,
        cues_first_command_correct: cues.filter(row => row.first_command_correct === true).length,
        missed_cues: cues.length - latencies.length,
        cues_without_commands: cues.filter(row => !row.command_count).length,
        correct_commands: commands.filter(row => row.outcome === "correct").length,
        wrong_hand_commands: commands.filter(row => row.outcome === "wrong_hand").length,
        repeated_correct_commands: cues.reduce((sum, row) => sum + row.repeated_correct_commands, 0),
        idle_commands: commands.filter(row => row.outcome === "idle").length,
        other_commands: commands.filter(row => row.outcome === "other").length,
        unknown_commands: commands.filter(row => row.outcome === "unknown").length,
        idle_seconds: events.filter(row => row.state === "idle").reduce((sum, row) => sum + row.end_s - row.start_s, 0),
        annotated_seconds: events[events.length - 1].end_s,
        gap_seconds: gaps.reduce((sum, row) => sum + row.duration_s, 0),
        idle_gap_seconds: idleGapSeconds,
      }, "Session");
      if (!sameLatency(session.first_correct_latencies_seconds, latencies)) fail("Session latencies do not match the event evidence");
      if (session.prediction_rows < 1 || session.prediction_rows < commands.length) fail("Session prediction count cannot support its commands");
    }

    const aggregate = (metrics, sessions, name) => {
      if (!sessions || !sessions.length) fail(`${name} has no sessions`);
      for (const field of [...COUNT_FIELDS, ...TIME_FIELDS]) {
        const expected = sessions.reduce((sum, row) => sum + row[field], 0);
        if (COUNT_FIELDS.includes(field) ? metrics[field] !== expected : !near(metrics[field], expected)) fail(`${name} ${field} does not match its sessions`);
      }
      if (!sameLatency(metrics.first_correct_latencies_seconds, sessions.flatMap(row => row.first_correct_latencies_seconds))) fail(`${name} latencies do not match its sessions`);
    };
    for (const participant of data.participants) aggregate(participant, sessionGroups.get(participant.participant), "Participant");
    aggregate(data.summary, data.sessions, "Summary");
  }

  function validateAudit(data) {
    if (!object(data) || data.schema_version !== "1.0" || data.kind !== "command_audit") fail("Expected a command audit with schema 1.0 from the local Python tool");
    validatePolicy(data.policy);
    validateMetrics(data.summary);
    for (const field of ["participants", "sessions", "event_results"]) array(data[field], field, false);
    for (const field of ["commands", "gaps"]) array(data[field], field);
    const participants = new Set();
    for (const row of data.participants) {
      if (!object(row)) fail("Invalid participant row");
      identifier(row.participant);
      validateMetrics(row);
      if (participants.has(row.participant)) fail("Duplicate participant");
      participants.add(row.participant);
    }
    const sessions = new Map();
    for (const row of data.sessions) {
      if (!object(row)) fail("Invalid session row");
      identifier(row.participant); identifier(row.session);
      validateMetrics(row);
      if (!participants.has(row.participant) || sessions.has(key(row))) fail("Unknown participant or duplicate session");
      sessions.set(key(row), row);
    }
    const events = new Map(), ends = new Map(), sessionCommands = new Map();
    for (const row of data.event_results) {
      if (!object(row)) fail("Invalid event interval");
      identifier(row.event_id);
      const session = key(row);
      if (!sessions.has(session) || !STATES.has(row.state) || !finite(row.start_s) || !finite(row.end_s) || row.end_s <= row.start_s || row.start_s !== (ends.get(session) || 0)) fail("Event intervals must be ordered and continuous from the start of each session");
      if (events.has(eventKey(row))) fail("Duplicate event interval");
      for (const field of ["command_count", "correct_commands", "wrong_hand_commands", "repeated_correct_commands"]) if (!count(row[field])) fail("Invalid event command counts");
      if (![true, false, null].includes(row.first_command_correct) || (row.first_correct_latency_seconds !== null && (!finite(row.first_correct_latency_seconds) || row.first_correct_latency_seconds >= row.end_s - row.start_s))) fail("Invalid event latency or first command outcome");
      events.set(eventKey(row), row); ends.set(session, row.end_s);
    }
    if (ends.size !== sessions.size) fail("Every session must have annotation intervals");
    const lastCommands = new Map();
    for (const row of data.commands) {
      if (!object(row)) fail("Invalid command row");
      const session = key(row), event = events.get(eventKey(row));
      if (!event || !["left", "right"].includes(row.direction) || !finite(row.time_s) || !finite(row.confidence) || row.confidence > 1 || row.time_s < event.start_s || row.time_s >= event.end_s || row.state !== event.state) fail("Invalid command timing, confidence or annotation");
      const expected = ["left", "right"].includes(row.state) ? (row.direction === row.state ? "correct" : "wrong_hand") : row.state;
      if (row.outcome !== expected || row.time_s <= (lastCommands.get(session) ?? -1)) fail("Invalid command outcome or order");
      lastCommands.set(session, row.time_s);
      sessionCommands.set(session, (sessionCommands.get(session) || 0) + 1);
    }
    const gapEnds = new Map();
    for (const row of data.gaps) {
      if (!object(row)) fail("Invalid prediction gap");
      const session = key(row);
      if (!sessions.has(session) || !finite(row.start_s) || !finite(row.end_s) || !finite(row.duration_s) || row.start_s < (gapEnds.get(session) || 0) || row.end_s <= row.start_s || row.end_s > ends.get(session) || !near(row.duration_s, row.end_s - row.start_s)) fail("Invalid prediction gap interval");
      gapEnds.set(session, row.end_s);
    }
    if (data.summary.command_count !== data.commands.length) fail("The summary does not match the command list");
    for (const [session, row] of sessions) if (row.command_count !== (sessionCommands.get(session) || 0) || !near(row.annotated_seconds, ends.get(session))) fail("Session counts or duration do not match the timeline");
    for (const field of COUNT_FIELDS) {
      if (data.sessions.reduce((sum, row) => sum + row[field], 0) !== data.summary[field] || data.participants.reduce((sum, row) => sum + row[field], 0) !== data.summary[field]) fail("Participant, session and summary counts do not agree");
    }
    verifyEvidence(data);
    return data;
  }

  function validateStudy(data) {
    if (!object(data) || data.schema_version !== "1.0" || data.kind !== "command_study") fail("The published study format was not recognised");
    array(data.conditions, "condition", false, 20);
    const names = new Set();
    for (const condition of data.conditions) {
      if (!object(condition) || typeof condition.condition !== "string" || typeof condition.label !== "string" || condition.label.length > 200 || typeof condition.validation_target_met !== "boolean" || names.has(condition.condition)) fail("Invalid study condition");
      names.add(condition.condition);
      validatePolicy(condition.policy);
      validateAudit(condition.validation); validateAudit(condition.test);
      if (POLICY_FIELDS.some(field => condition.policy[field] !== condition.validation.policy[field] || condition.policy[field] !== condition.test.policy[field])) fail("The test policy must match the validation policy");
      const validationSessions = new Set(condition.validation.sessions.map(key));
      if (condition.test.sessions.some(row => validationSessions.has(key(row)))) fail("Validation and test sessions must be separate");
    }
    return data;
  }

  function parseLocalReport(text) {
    if (typeof text !== "string" || new TextEncoder().encode(text).length > MAX_BYTES) fail("Choose an audit JSON smaller than 5 MB");
    let value;
    try { value = JSON.parse(text); } catch { fail("The selected file is not valid JSON"); }
    return validateAudit(value);
  }

  function timeLabel(seconds) {
    if (!finite(seconds)) return "—";
    const tenths = Math.round(seconds * 10);
    const minutes = Math.floor(tenths / 600);
    return `${minutes}:${((tenths % 600) / 10).toFixed(1).padStart(4, "0")}`;
  }

  function policyTimeLabel(seconds) {
    return finite(seconds) ? `${seconds.toLocaleString("en-GB", {maximumFractionDigits: 3})} s` : "—";
  }

  function windowSlice(audit, sessionKey, start, duration) {
    if (!finite(start) || !finite(duration) || duration <= 0) fail("Invalid time interval");
    const events = audit.event_results.filter(row => key(row) === sessionKey);
    if (!events.length) fail("The selected session has no annotation intervals");
    const end = events[events.length - 1].end_s;
    const from = Math.min(start, Math.max(0, end - duration));
    const to = Math.min(end, from + duration);
    return {
      start: from, end: to, sessionEnd: end,
      events: events.filter(row => row.start_s < to && row.end_s > from),
      commands: audit.commands.filter(row => key(row) === sessionKey && row.time_s >= from && row.time_s < to),
      gaps: audit.gaps.filter(row => key(row) === sessionKey && row.start_s < to && row.end_s > from),
    };
  }

  return {MAX_BYTES, validateAudit, validateStudy, parseLocalReport, timeLabel, policyTimeLabel, windowSlice, sessionKey: key};
})();

if (typeof module !== "undefined" && module.exports) module.exports = CommandAudit;

if (typeof document !== "undefined") {
  const $ = id => document.getElementById(id);
  const stateLabels = {idle: "Instructed rest", left: "Imagine left", right: "Imagine right", other: "Other task", unknown: "Unknown annotation"};
  const outcomeLabels = {correct: "Correct", wrong_hand: "Wrong hand", idle: "During instructed rest", other: "During another task", unknown: "Unknown annotation"};
  const phaseColours = {idle: "#284e36", left: "#73578e", right: "#326b76", other: "#6b5532", unknown: "#36343b"};
  let published = null, audit = null, local = false, loadToken = 0;
  const originalLead = document.querySelector(".lead").textContent;
  const format = (value, digits = 1) => value === null ? "—" : Number(value).toLocaleString("en-GB", {maximumFractionDigits: digits, minimumFractionDigits: digits});
  const seconds = value => value === null ? "—" : `${format(value)} s`;
  const text = (id, value) => { $(id).textContent = value; };

  function error(message) { text("command-error", message); $("command-error").hidden = false; }
  function clearError() { $("command-error").hidden = true; text("command-error", ""); }

  function setOptions(id, rows) {
    const previous = $(id).value;
    $(id).replaceChildren(...rows.map(([value, label]) => {
      const option = document.createElement("option"); option.value = value; option.textContent = label; return option;
    }));
    if (rows.some(row => row[0] === previous)) $(id).value = previous;
  }

  function table(id, rows, columns) {
    const body = $(id);
    if (!rows.length) {
      const row = document.createElement("tr"), cell = document.createElement("td");
      cell.colSpan = columns; cell.className = "empty-row"; cell.textContent = "No commands were emitted in this interval"; row.append(cell); body.replaceChildren(row); return;
    }
    body.replaceChildren(...rows.map(values => {
      const row = document.createElement("tr");
      values.forEach((value, index) => { const cell = document.createElement(index ? "td" : "th"); if (!index) cell.scope = "row"; cell.textContent = String(value); row.append(cell); });
      return row;
    }));
  }

  function renderMetrics() {
    const m = audit.summary;
    const items = [
      [format(m.idle_commands_per_minute, 2), "Commands per minute of instructed rest", `${m.idle_commands} commands across ${format(m.idle_seconds / 60, 2)} annotated minutes`],
      [`${m.cues_first_command_correct} / ${m.cues}`, "Cues whose first command was correct", "All left and right cues remain in the denominator"],
      [`${m.missed_cues} / ${m.cues}`, "Cues without a correct command", `${m.cues_without_commands} cues received no command at all`],
      [seconds(m.median_first_correct_latency_seconds), "Median time to the first correct command", `${m.latency_observations} contributing cues; cues without a correct command are excluded`],
      [String(m.repeated_correct_commands), "Repeated correct commands", "Additional correct commands during an already answered cue"],
      [String(m.unknown_commands), "Commands during unknown annotation", `${m.other_commands} further commands occurred during another task`],
    ];
    $("command-metrics").replaceChildren(...items.map(([value, label, note]) => {
      const block = document.createElement("div"); block.className = "metric";
      const number = document.createElement("strong"); number.className = "metric-value"; number.textContent = value;
      const name = document.createElement("span"); name.className = "metric-label"; name.textContent = label;
      const description = document.createElement("span"); description.className = "metric-note"; description.textContent = note;
      block.append(number, name, description); return block;
    }));
    text("exposure-note", `${m.command_count} commands in total, including ${m.wrong_hand_commands} wrong hand commands; the annotations cover ${format(m.annotated_seconds / 60, 2)} minutes; prediction spans cover ${format(m.observed_span_seconds / 60, 2)} minutes, with ${seconds(m.gap_seconds)} in flagged gaps and ${seconds(m.unobserved_edge_seconds)} outside those spans; flagged gaps overlap ${seconds(m.idle_gap_seconds)} of instructed rest; these exposure measures describe the supplied records and do not establish uninterrupted sensing`);
    table("command-participants", audit.participants.map(row => [row.participant, row.command_count, format(row.idle_commands_per_minute, 2), `${row.cues_first_command_correct} / ${row.cues}`, `${row.missed_cues} / ${row.cues}`, seconds(row.median_first_correct_latency_seconds), row.latency_observations]), 7);
  }

  function renderPolicy(condition) {
    const policy = audit.policy;
    const fields = [["Entry score", `${format(policy.enter_threshold * 100, 0)}%`], ["Release score", `${format(policy.release_threshold * 100, 0)}%`], ["Agreement time", CommandAudit.policyTimeLabel(policy.hold_seconds)], ["Cooldown", CommandAudit.policyTimeLabel(policy.cooldown_seconds)], ["Gap reset after", CommandAudit.policyTimeLabel(policy.max_gap_seconds)]];
    $("policy-details").replaceChildren(...fields.map(([label, value]) => {
      const item = document.createElement("div"), term = document.createElement("dt"), description = document.createElement("dd");
      term.textContent = label; description.textContent = value; item.append(term, description); return item;
    }));
    $("policy-status").classList.toggle("met", !local && condition.validation_target_met);
    text("policy-status", local ? "This report supplies its own fixed operating rule; the viewer has not selected it, checked the training split or verified whether a validation target was met" : condition.validation_target_met ? "The validation target was met: at most one command per minute of instructed rest and a correct first command for at least half the movement cues; the selected rule is kept fixed for the test session; meeting this study target does not establish reliable live control" : "The validation target was not met: no candidate reached both at most one command per minute of instructed rest and a correct first command for at least half the movement cues, so the study uses its declared fallback rule; test results are displayed with that rule unchanged and do not overturn the validation failure");
  }

  function updateAudit() {
    let condition = null;
    if (!local) {
      condition = published.conditions.find(row => row.condition === $("command-condition").value);
      audit = condition[$("command-split").value];
    }
    renderPolicy(condition); renderMetrics();
    setOptions("command-session", audit.sessions.map(row => [CommandAudit.sessionKey(row), `Participant ${row.participant} · Session ${row.session}`]));
    $("timeline-start").value = "0";
    $("inspect").hidden = false; $("timeline-section").hidden = false;
    renderTimeline();
    text("command-status", local ? "Showing a local report that has not been independently verified" : `Showing ${condition.label.toLowerCase()} · ${$("command-split").value} session · ${audit.participants.length} participants`);
  }

  function svg(tag, attributes, content) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attributes).forEach(([name, value]) => element.setAttribute(name, String(value)));
    if (content !== undefined) element.textContent = content;
    return element;
  }

  function renderTimeline() {
    if (!audit) return;
    const sessionKey = $("command-session").value, duration = Number($("timeline-duration").value);
    const slice = CommandAudit.windowSlice(audit, sessionKey, Number($("timeline-start").value), duration);
    const maximum = Math.max(0, slice.sessionEnd - duration);
    $("timeline-start").max = String(maximum); $("timeline-start").value = String(slice.start);
    $("timeline-start").disabled = maximum === 0;
    $("timeline-start").setAttribute("aria-valuetext", `${CommandAudit.timeLabel(slice.start)} into the recording`);
    $("timeline-back").disabled = slice.start <= 0; $("timeline-next").disabled = slice.start >= maximum;
    text("timeline-position-label", CommandAudit.timeLabel(slice.start));
    const session = audit.sessions.find(row => CommandAudit.sessionKey(row) === sessionKey);
    const title = `Participant ${session.participant}, session ${session.session}, ${CommandAudit.timeLabel(slice.start)} to ${CommandAudit.timeLabel(slice.end)}`;
    const canvas = $("command-timeline");
    canvas.replaceChildren(svg("title", {id: "timeline-svg-title"}, title), svg("desc", {id: "timeline-svg-description"}, `${slice.commands.length} emitted commands and ${slice.gaps.length} prediction gaps overlap this interval; command details are in the following table`));
    const width = Math.max(280, Math.min(1000, Math.round(canvas.clientWidth) || 1000));
    canvas.setAttribute("viewBox", `0 0 ${width} 220`);
    const x = time => 65 + (width - 85) * (time - slice.start) / (slice.end - slice.start);
    canvas.append(svg("text", {x: 8, y: 46, fill: "#d2d2d2", "font-size": 13}, "Task"), svg("text", {x: 8, y: 108, fill: "#d2d2d2", "font-size": 13}, "Left"), svg("text", {x: 8, y: 156, fill: "#d2d2d2", "font-size": 13}, "Right"));
    for (const y of [104, 152]) canvas.append(svg("line", {x1: 65, x2: width - 20, y1: y, y2: y, stroke: "#2c2c2c", "stroke-width": 1}));
    for (const event of slice.events) {
      const rect = svg("rect", {x: x(Math.max(slice.start, event.start_s)), y: 25, width: Math.max(0, x(Math.min(slice.end, event.end_s)) - x(Math.max(slice.start, event.start_s))), height: 34, fill: phaseColours[event.state], stroke: "#080808", "stroke-width": .7});
      rect.append(svg("title", {}, `${stateLabels[event.state]}, ${CommandAudit.timeLabel(event.start_s)} to ${CommandAudit.timeLabel(event.end_s)}`)); canvas.append(rect);
    }
    for (const gap of slice.gaps) {
      const from = Math.max(slice.start, gap.start_s), to = Math.min(slice.end, gap.end_s);
      const rect = svg("rect", {x: x(from), y: 76, width: x(to) - x(from), height: 93, fill: "#694b48", "fill-opacity": .5, stroke: "#c28f80", "stroke-dasharray": "4 3"});
      rect.append(svg("title", {}, `Prediction gap, ${CommandAudit.timeLabel(gap.start_s)} to ${CommandAudit.timeLabel(gap.end_s)}`)); canvas.append(rect);
    }
    for (const command of slice.commands) {
      const cx = x(command.time_s), cy = command.direction === "left" ? 104 : 152, correct = command.outcome === "correct";
      const marker = correct ? svg("circle", {cx, cy, r: 4.5, fill: "#67e685", stroke: "#080808", "stroke-width": 1}) : svg("path", {d: `M ${cx} ${cy - 5} L ${cx + 5} ${cy} L ${cx} ${cy + 5} L ${cx - 5} ${cy} Z`, fill: "#f09b80", stroke: "#080808", "stroke-width": 1});
      marker.append(svg("title", {}, `${CommandAudit.timeLabel(command.time_s)}, ${command.direction}, ${outcomeLabels[command.outcome]}`)); canvas.append(marker);
    }
    const ticks = width < 600 ? 2 : 4;
    for (let index = 0; index <= ticks; index++) {
      const tick = slice.start + (slice.end - slice.start) * index / ticks;
      canvas.append(svg("line", {x1: x(tick), x2: x(tick), y1: 177, y2: 183, stroke: "#777"}), svg("text", {x: x(tick), y: 205, "text-anchor": index === 0 ? "start" : index === ticks ? "end" : "middle", "font-size": 13, fill: "#b1b1b1"}, CommandAudit.timeLabel(tick)));
    }
    text("timeline-caption", `${title} · Time is measured from the start of the recording`);
    text("session-note", `This session contains ${session.command_count} commands, ${session.idle_commands} during instructed rest and ${session.missed_cues} cues without a correct command; annotated duration ${seconds(session.annotated_seconds)}; flagged prediction gaps ${seconds(session.gap_seconds)}; time outside the first and last prediction ${seconds(session.unobserved_edge_seconds)}`);
    text("command-table-caption", `${slice.commands.length} commands from ${CommandAudit.timeLabel(slice.start)} up to ${CommandAudit.timeLabel(slice.end)}`);
    table("command-events", slice.commands.slice(0, 500).map(row => [CommandAudit.timeLabel(row.time_s), row.direction === "left" ? "Left" : "Right", stateLabels[row.state], outcomeLabels[row.outcome], `${format(row.confidence * 100, 1)}%`]), 5);
    text("table-limit-note", slice.commands.length > 500 ? "The table shows the first 500 commands in this interval; shorten the visible duration or download the audit for the complete list" : "The interval includes its start and excludes its end; selecting a later interval does not count a boundary command twice");
    table("annotation-intervals", slice.events.slice(0, 500).map(row => [CommandAudit.timeLabel(row.start_s), CommandAudit.timeLabel(row.end_s), stateLabels[row.state], row.event_id]), 4);
    text("annotation-limit-note", slice.events.length > 500 ? "Showing the first 500 overlapping annotations; shorten the visible duration or download the audit for the complete list" : "The event boundaries describe the original annotation intervals, including any portion outside the visible window");
  }

  function showPublished() {
    if (!published) return;
    local = false;
    document.querySelector(".lead").textContent = originalLead;
    $("command-condition").disabled = false; $("command-split").disabled = false;
    setOptions("command-condition", published.conditions.map(row => [row.condition, row.label]));
    setOptions("command-split", [["test", "Test"], ["validation", "Validation"]]);
    $("command-split").value = "test";
    text("report-provenance", "Published Zhou2016 study using separate sessions from the same four participants; the original PhysioNet decoder remains unchanged; this comparison uses its own model and saved predictions");
    $("restore-study").disabled = true; clearError(); updateAudit();
  }

  async function loadStudy() {
    const token = ++loadToken;
    $("reload-study").hidden = true;
    try {
      const response = await fetch("/static/commands-study.json", {credentials: "omit"});
      if (!response.ok) throw new Error(`The published study could not be loaded (${response.status})`);
      const body = await response.text();
      if (new TextEncoder().encode(body).length > 20 * 1024 * 1024) throw new Error("The published study exceeds the viewer limit");
      const report = CommandAudit.validateStudy(JSON.parse(body));
      if (token !== loadToken) return;
      published = report;
      if (local) $("restore-study").disabled = false;
      else showPublished();
    } catch (problem) {
      if (token !== loadToken) return;
      error(`${problem.message}; you can retry or open a local command audit below`);
      $("reload-study").hidden = false;
      if (!local) text("command-status", "The published study is unavailable");
    }
  }

  $("command-condition").addEventListener("change", updateAudit);
  $("command-split").addEventListener("change", updateAudit);
  $("command-session").addEventListener("change", () => { $("timeline-start").value = "0"; renderTimeline(); });
  $("timeline-duration").addEventListener("change", renderTimeline);
  $("timeline-start").addEventListener("input", renderTimeline);
  for (const [id, direction] of [["timeline-back", -1], ["timeline-next", 1]]) $(id).addEventListener("click", () => { $("timeline-start").value = String(Math.max(0, Number($("timeline-start").value) + direction * Number($("timeline-duration").value))); renderTimeline(); });
  $("restore-study").addEventListener("click", () => { $("local-command-audit").value = ""; showPublished(); });
  $("reload-study").addEventListener("click", loadStudy);
  let resizePending = false;
  window.addEventListener("resize", () => {
    if (resizePending) return;
    resizePending = true;
    requestAnimationFrame(() => { resizePending = false; renderTimeline(); });
  });
  $("local-command-audit").addEventListener("change", async event => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      if (file.size > CommandAudit.MAX_BYTES) throw new Error("Choose an audit JSON smaller than 5 MB");
      const next = CommandAudit.parseLocalReport(await file.text());
      audit = next; local = true;
      document.querySelector(".lead").textContent = "Inspect commands, annotated intervals and waiting times from a locally generated audit; this viewer checks the report structure, while the correctness of its predictions, labels and evaluation design remains the responsibility of its author";
      setOptions("command-condition", [["local", "Local report"]]); $("command-condition").disabled = true;
      setOptions("command-split", [["local", "Supplied in the local report"]]); $("command-split").disabled = true;
      text("report-provenance", `Local file: ${file.name}; its source, model and evaluation split have not been independently verified; the file stays in this browser and does not replace the published study`);
      $("restore-study").disabled = !published; clearError(); updateAudit();
    } catch (problem) { error(`${problem.message}; the previous report has been kept`); }
  });
  $("download-audit").addEventListener("click", () => {
    if (!audit) return;
    const blob = new Blob([JSON.stringify(audit, null, 2) + "\n"], {type: "application/json"});
    const url = URL.createObjectURL(blob), link = document.createElement("a");
    link.href = url; link.download = local ? "intentlab-local-command-audit.json" : `intentlab-${$("command-condition").value}-${$("command-split").value}-commands.json`;
    link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  loadStudy();
}
