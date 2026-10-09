"use strict";

const $ = (id) => document.getElementById(id);
const names = { baseline: "Baseline", personal_temperature: "Personal temperature", alignment: "Euclidean alignment", alignment_temperature: "Alignment + personal temperature" };
const percent = (value) => value == null ? "Undefined" : `${(value * 100).toFixed(1)}%`;
const interval = (value) => value == null ? "not reported (fewer than two contributing participants)" : `${percent(value[0])}–${percent(value[1])}`;
let report;

function options(id, values, label = (x) => x) {
  const previous = $(id).value;
  $(id).replaceChildren(...values.map(value => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label(value);
    return option;
  }));
  if (values.map(String).includes(previous)) $(id).value = previous;
}

function cells(body, rows) {
  $(body).replaceChildren(...rows.map(values => {
    const row = document.createElement("tr");
    values.forEach((value, index) => {
      const cell = document.createElement(index === 0 ? "th" : "td");
      if (index === 0) cell.scope = "row";
      cell.textContent = value;
      row.append(cell);
    });
    return row;
  }));
}

function svgElement(tag, attrs, text) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
  if (text != null) element.textContent = text;
  return element;
}

function plot(points, selected) {
  const svg = $("curve");
  while (svg.children.length > 2) svg.lastElementChild.remove();
  const x = (value) => 70 + value * 540;
  const y = (value) => 280 - value * 240;
  for (let value = 0; value <= 1.001; value += 0.25) {
    svg.append(svgElement("line", {x1:70, y1:y(value), x2:610, y2:y(value), stroke:"#e2e7df"}));
    svg.append(svgElement("text", {x:55, y:y(value)+4, "text-anchor":"end"}, `${Math.round(value*100)}%`));
    svg.append(svgElement("text", {x:x(value), y:301, "text-anchor":"middle"}, `${Math.round(value*100)}%`));
  }
  svg.append(svgElement("text", {x:340, y:328, "text-anchor":"middle"}, "Trials receiving a command (coverage)"));
  svg.append(svgElement("text", {x:70, y:22}, "Errors among accepted predictions (selective risk)"));
  const valid = points.filter(p => p.selective_risk != null);
  svg.append(svgElement("polyline", {points:valid.map(p => `${x(p.coverage)},${y(p.selective_risk)}`).join(" "), fill:"none", stroke:"#346f5a", "stroke-width":2}));
  valid.forEach(point => {
    const dot = svgElement("circle", {cx:x(point.coverage), cy:y(point.selective_risk), r:point === selected ? 7 : 4, fill:point === selected ? "#9b482a" : "#346f5a", stroke:"white", "stroke-width":1});
    dot.append(svgElement("title", {}, `Threshold ${percent(point.threshold)}, coverage ${percent(point.coverage)}, error ${percent(point.selective_risk)}`));
    svg.append(dot);
  });
}

function update() {
  options("seed", [...new Set(report.groups.filter(g => g.condition === $("condition").value).map(g => g.seed))]);
  const group = report.groups.find(g => g.condition === $("condition").value && g.seed === $("seed").value && g.split === $("split").value);
  options("threshold", group.points.map(p => p.threshold), percent);
  const point = group.points.find(p => p.threshold === Number($("threshold").value));
  $("selection-note").textContent = `Validation target ${group.validation_target_met ? "met" : "not met"}; the validation rule ${group.validation_target_met ? "selected" : "fell back to"} ${percent(group.selected_threshold)}. ${point.threshold === group.selected_threshold ? "The displayed threshold matches that rule" : "This displayed threshold is exploratory"}. Model seeds are shown separately.`;
  $("summary").replaceChildren(...[
    [`${point.accepted} / ${point.trials}`, `trials accepted (${percent(point.coverage)})`],
    [`${point.accepted - point.correct}`, `wrong commands among ${point.accepted} accepted`],
    [`${point.participants_without_commands} / ${group.participant_count}`, "participants receiving no commands"],
  ].map(([value, label]) => {
    const block = document.createElement("div"); block.className = "audit-stat";
    const strong = document.createElement("strong"); strong.textContent = value;
    const span = document.createElement("span"); span.textContent = label;
    block.append(strong, span); return block;
  }));
  $("uncertainty").textContent = `Accepted accuracy: ${percent(point.accepted_accuracy)}; participant-bootstrap 95% interval: ${interval(point.accepted_accuracy_ci95)}. Coverage interval: ${interval(point.coverage_ci95)}. Accuracy is defined in ${point.accuracy_bootstrap_valid}/${report.bootstrap.repeats} resamples. Median participant coverage: ${percent(point.median_participant_coverage)}.`;
  plot(group.points, point);
  cells("operating-rows", group.points.map(p => [percent(p.threshold), `${p.accepted} / ${p.trials}`, p.accepted-p.correct, percent(p.coverage), percent(p.accepted_accuracy), `${p.participants_with_commands} / ${group.participant_count}`, `${p.accuracy_bootstrap_valid} / ${report.bootstrap.repeats}`]));
  cells("participant-rows", point.participants.map(p => [p.subject, `${p.accepted} / ${p.trials}`, p.accepted-p.correct, percent(p.coverage), percent(p.accepted_accuracy), percent(p.balanced_accuracy)]));
}

function validate(data) {
  if (data.schema_version !== "1.0" || !Number.isInteger(data.bootstrap?.repeats) || data.bootstrap.repeats < 100 || !Array.isArray(data.groups) || data.groups.length < 2 || data.groups.length > 1000) throw new Error("Expected an IntentLab schema 1.0 audit from the local CSV tool");
  const keys = new Set();
  for (const group of data.groups) {
    if (typeof group.condition !== "string" || typeof group.seed !== "string" || !["validation", "test"].includes(group.split) || !Array.isArray(group.points) || !group.points.length || group.points.length > 21 || !Number.isInteger(group.participant_count)) throw new Error("Invalid condition or operating points");
    const key = JSON.stringify([group.condition, group.seed, group.split]);
    if (keys.has(key)) throw new Error("Duplicate condition, seed and split");
    keys.add(key);
    for (const point of group.points) {
      for (const field of ["threshold", "coverage", "selective_risk", "accepted_accuracy", "median_participant_coverage"]) {
        if (!(point[field] === null && ["selective_risk", "accepted_accuracy"].includes(field)) && (!Number.isFinite(point[field]) || point[field] < 0 || point[field] > 1)) throw new Error("Invalid audit probability");
      }
      for (const field of ["accepted", "trials", "correct", "participants_with_commands", "participants_without_commands", "accuracy_bootstrap_valid"]) {
        if (!Number.isInteger(point[field]) || point[field] < 0) throw new Error("Invalid audit count");
      }
      if (!Array.isArray(point.participants) || point.participants.length !== group.participant_count) throw new Error("Invalid participant rows");
      for (const field of ["coverage_ci95", "accepted_accuracy_ci95"]) {
        const ci = point[field];
        if (ci !== null && (!Array.isArray(ci) || ci.length !== 2 || ci.some(v => !Number.isFinite(v) || v < 0 || v > 1))) throw new Error("Invalid audit interval");
      }
    }
  }
  for (const group of data.groups) {
    if (!keys.has(JSON.stringify([group.condition, group.seed, group.split === "test" ? "validation" : "test"]))) throw new Error("Every condition and seed needs both validation and test results");
  }
  return data;
}

function show(data, local) {
  report = validate(data);
  options("condition", [...new Set(report.groups.map(g => g.condition))], value => names[value] || value);
  $("condition").value = report.groups.some(g => g.condition === "baseline") ? "baseline" : report.groups[0].condition;
  $("threshold").replaceChildren();
  update();
  const thresholds = [...$("threshold").options].map(o => o.value);
  if (thresholds.includes(String(report.rule.fallback_threshold))) $("threshold").value = report.rule.fallback_threshold;
  update();
  $("provenance").textContent = local ? "You are viewing a local report, not the published IntentLab results. These values are supplied by the file and are not independently verified by this page; data remain in this browser. Returning to the published study restores the archived comparison." : "This is a descriptive reanalysis of the archived calibration and alignment experiment, with the trained models kept fixed. The same 21 test participants appear in each seed; seeds do not add independent participants, and no condition met the original validation reliability target.";
  $("explorer").hidden = false;
  $("loading").hidden = true;
}

async function loadPublished() {
  try {
    const response = await fetch("/static/reliability.json");
    if (!response.ok) throw new Error("The published audit could not be loaded");
    show(await response.json(), false);
    $("file-status").textContent = "";
    $("local-report").value = "";
  } catch (error) { $("loading").hidden = false; $("loading").textContent = `${error.message}. The local tool and downloadable report are linked above.`; }
}

for (const id of ["condition", "seed", "split", "threshold"]) $(id).addEventListener("change", update);
$("reset-report").addEventListener("click", loadPublished);
$("local-report").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    if (file.size > 5 * 1024 * 1024) throw new Error("Please use an audit smaller than 5 MB");
    const data = validate(JSON.parse(await file.text()));
    if (!data.rule || !Number.isFinite(data.rule.fallback_threshold)) throw new Error("Missing threshold rule");
    show(data, true);
    $("file-status").textContent = "Local report opened without uploading it";
  } catch (error) { $("file-status").textContent = `Could not open this report: ${error.message}`; }
});
loadPublished();
