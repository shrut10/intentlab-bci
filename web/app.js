/* Interface copy lives here and in index.html. See docs/EDITING.md. */
"use strict";

const $ = (id) => document.getElementById(id);
const pct = (value, digits = 1) =>
  value == null ? "No commands" : `${(100 * value).toFixed(digits)}%`;
const escapeHTML = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (ch) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        ch
      ],
  );
const state = {
  overview: null,
  report: null,
  trials: [],
  prediction: null,
  cursor: 50,
  request: 0,
  replay: 0,
  playing: false,
};
const svgText = (x, y, value, anchor = "start", extra = "") =>
  `<text x="${x}" y="${y}" text-anchor="${anchor}" class="svg-label" ${extra}>${escapeHTML(value)}</text>`;

async function api(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 25000);
  try {
    const response = await fetch(path, {
      ...options,
      signal: controller.signal,
    });
    if (!response.ok)
      throw new Error(
        `The server returned ${response.status}. Please try again.`,
      );
    return await response.json();
  } catch (error) {
    if (error.name === "AbortError")
      throw new Error("The server took too long to respond. Please try again.");
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

function showError(message) {
  const banner = $("error-banner");
  banner.textContent = message;
  banner.hidden = false;
}

function stopReplay() {
  if (state.playing) {
    state.request += 1;
    $("decode-button").disabled = false;
    $("decode-button").textContent = "Run decoder ↗";
  }
  state.replay += 1;
  state.playing = false;
  $("replay-button").textContent = "Replay 6 trials";
}

function selectedTrial() {
  return state.trials.find((trial) => trial.id === $("trial-select").value);
}

function fillTrials() {
  const subject = Number($("subject-select").value);
  const trials = state.trials.filter((trial) => trial.subject === subject);
  $("trial-select").replaceChildren(
    ...trials.map(
      (trial, i) =>
        new Option(
          `Trial ${i + 1} · ${trial.cue_seconds.toFixed(1)} s`,
          trial.id,
        ),
    ),
  );
}

function renderTrialMeta() {
  const trial = selectedTrial();
  $("cue-label").textContent =
    `Cued imagery: ${trial.label === 0 ? "left" : "right"} fist`;
  $("recording-id").textContent = trial.id;
  $("trial-badge").textContent =
    `S${String(trial.subject).padStart(3, "0")} · UNSEEN PERSON`;
  $("csv-link").href = `/api/trials/${encodeURIComponent(trial.id)}/csv`;
  $("csv-link").removeAttribute("aria-disabled");
}

async function decode(moveCursor = false) {
  const sequence = ++state.request;
  const trial = selectedTrial();
  if (!trial) return false;
  renderTrialMeta();
  $("decode-button").disabled = true;
  $("decode-button").textContent = "Decoding…";
  $("error-banner").hidden = true;
  try {
    const prediction = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        trial_id: trial.id,
        model: $("model-select").value,
        threshold: Number($("threshold").value) / 100,
        noise: Number($("noise-select").value),
        drop_channel: $("drop-select").value || null,
      }),
    });
    if (sequence !== state.request) return false;
    state.prediction = prediction;
    renderPrediction(prediction, moveCursor);
    return true;
  } catch (error) {
    if (sequence === state.request) {
      showError(`The recording could not be decoded. ${error.message}`);
      state.prediction = null;
      drawWaveform();
      $("left-prob").textContent = "—";
      $("right-prob").textContent = "—";
      $("occlusion-bars").replaceChildren();
      $("scalp").replaceChildren();
      $("spectrum").replaceChildren();
      $("decision-title").textContent = "Prediction unavailable";
      $("decision-explanation").textContent =
        "No command was sent. Run the decoder to retry.";
      stopReplay();
    }
    return false;
  } finally {
    if (sequence === state.request) {
      $("decode-button").disabled = false;
      $("decode-button").innerHTML =
        'Run decoder <span aria-hidden="true">↗</span>';
    }
  }
}

function renderPrediction(prediction, moveCursor) {
  $("left-prob").textContent = pct(prediction.left_probability, 0);
  $("right-prob").textContent = pct(prediction.right_probability, 0);
  $("left-fill").style.width = pct(prediction.left_probability, 3);
  $("latency").textContent = `${prediction.latency_ms.toFixed(1)} ms`;
  const label = prediction.predicted_label === 0 ? "left" : "right";
  const agrees = prediction.predicted_label === prediction.recorded_label;
  $("decision-title").textContent = prediction.accepted
    ? `${label === "left" ? "Left" : "Right"} command`
    : "Wait. No command.";
  $("decision-explanation").textContent = prediction.accepted
    ? `${pct(prediction.confidence, 0)} confidence · ${agrees ? "matches" : "does not match"} the recorded cue.`
    : `The model leans ${label} at ${pct(prediction.confidence, 0)} confidence, below your ${pct(prediction.threshold, 0)} threshold.`;
  $("command-state").textContent = prediction.accepted
    ? `${label} ${label === "left" ? "←" : "→"}`
    : "Abstained";
  $("cursor").classList.toggle("waiting", !prediction.accepted);
  if (moveCursor && prediction.accepted) {
    state.cursor = Math.max(
      15,
      Math.min(85, state.cursor + (label === "right" ? 10 : -10)),
    );
    $("cursor").style.left = `${state.cursor}%`;
  }
  drawWaveform();
  drawOcclusion(prediction.occlusion);
  drawSpectrum(prediction.spectrum);
}

function drawWaveform() {
  const canvas = $("waveform");
  const bounds = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(bounds.width * ratio);
  canvas.height = Math.round(bounds.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const w = bounds.width,
    h = bounds.height,
    left = 36,
    right = w - 8,
    top = 7,
    bottom = h - 25;
  ctx.font = '8px "SFMono-Regular", Consolas, monospace';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 6; i++) {
    const x = left + ((right - left) * i) / 6;
    ctx.strokeStyle = "#a6cbb51b";
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom + 4);
    ctx.stroke();
    ctx.fillStyle = "#8aaea1";
    ctx.textAlign = "center";
    ctx.fillText((1 + i / 2).toFixed(1), x, h - 9);
  }
  if (!state.prediction || !state.overview) {
    ctx.fillStyle = "#aec6ba";
    ctx.fillText("No signal to display yet.", w / 2, h / 2);
    return;
  }
  const signal = state.prediction.signal;
  const spacing = (bottom - top) / 9;
  signal.forEach((values, index) => {
    const y = top + spacing * (index + 0.5);
    const name = state.overview.channels[index];
    const dropped = state.prediction.drop_channel === name;
    ctx.fillStyle = dropped ? "#64746c" : "#a8c6b9";
    ctx.textAlign = "left";
    ctx.fillText(name, 2, y + 3);
    ctx.strokeStyle = "#a6cbb517";
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
    ctx.stroke();
    ctx.save();
    ctx.beginPath();
    ctx.rect(left, top, right - left, bottom - top);
    ctx.clip();
    ctx.strokeStyle = dropped
      ? "#687a70"
      : name === "C3" || name === "C4"
        ? "#e1ab88"
        : "#90cdb2";
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    values.forEach((value, i) => {
      const px = left + ((right - left) * i) / values.length;
      const py = y - value * spacing * 0.26;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
    ctx.restore();
  });
}

function drawOcclusion(rows) {
  const maximum = Math.max(
    ...rows.map((row) => Math.abs(row.right_probability_change)),
    0.01,
  );
  $("occlusion-bars").innerHTML = rows
    .map((row) => {
      const delta = row.right_probability_change;
      const width = (Math.abs(delta) / maximum) * 48;
      return `<div class="occlusion-row"><span>${escapeHTML(row.channel)}</span><div class="occlusion-track"><div class="occlusion-fill${delta < 0 ? " negative" : ""}" data-width="${width}" data-left="${delta < 0 ? 50 - width : 50}"></div></div><span>${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)} pp</span></div>`;
    })
    .join("");
  document.querySelectorAll(".occlusion-fill").forEach((bar) => {
    bar.style.width = `${bar.dataset.width}%`;
    bar.style.left = `${bar.dataset.left}%`;
  });
  let svg =
    '<path d="M111 31l14-17 14 17" fill="none" stroke="#a7b6a4"/><ellipse cx="125" cy="125" rx="85" ry="94" fill="#f0f2e9" stroke="#b9c5b4"/><path d="M39 102q-20 23 0 43M211 102q20 23 0 43" fill="none" stroke="#b9c5b4"/>';
  const positions = [
    [77, 72],
    [125, 66],
    [173, 72],
    [65, 124],
    [125, 124],
    [185, 124],
    [77, 176],
    [125, 185],
    [173, 176],
  ];
  rows.forEach((row, i) => {
    const [x, y] = positions[i];
    const strength = Math.abs(row.right_probability_change) / maximum;
    const radius = 8 + strength * 6;
    svg += `<circle cx="${x}" cy="${y}" r="${radius}" fill="${row.right_probability_change >= 0 ? "#73a78d" : "#d79f7b"}" fill-opacity="${0.3 + strength * 0.6}"/><circle cx="${x}" cy="${y}" r="3" fill="#426553"/>${svgText(x, y + 25, row.channel, "middle")}`;
  });
  $("scalp").innerHTML = svg;
}

function drawSpectrum(data) {
  const x = (hz) => 38 + ((hz - 4) / 36) * 380;
  const max = Math.max(...data.power, 1e-8);
  const y = (power) => 166 - (power / max) * 130;
  let svg = `<rect x="${x(8)}" y="23" width="${x(13) - x(8)}" height="143" fill="#c8ddcd" fill-opacity=".5"/><rect x="${x(13)}" y="23" width="${x(30) - x(13)}" height="143" fill="#eadfbd" fill-opacity=".5"/>`;
  [0, 0.5, 1].forEach((value) => {
    svg += `<path d="M38 ${y(value * max)}H418" class="svg-grid"/>${svgText(30, y(value * max) + 3, value === 1 ? "peak" : value, "end")}`;
  });
  [4, 8, 13, 20, 30, 40].forEach((hz) => {
    svg += svgText(x(hz), 186, hz, "middle");
  });
  svg += svgText(418, 203, "Hz · power scaled to this trial’s peak", "end");
  const points = data.hz
    .map((hz, i) => `${x(hz)},${y(data.power[i])}`)
    .join(" ");
  svg += `<polyline points="${points}" class="svg-line"/>`;
  $("spectrum").innerHTML = svg;
}

function lineChart(element, points, mode) {
  const calibration = mode === "calibration";
  const width = calibration ? 380 : 470,
    height = calibration ? 230 : 245;
  const left = 45,
    top = 16,
    right = width - 20,
    bottom = height - 43;
  const x = (value) => left + value * (right - left);
  const y = (value) => bottom - value * (bottom - top);
  let svg = "";
  [0, 0.25, 0.5, 0.75, 1].forEach((value) => {
    svg += `<path d="M${left} ${y(value)}H${right}" class="svg-grid"/>${svgText(left - 9, y(value) + 3, pct(value, 0), "end")}${svgText(x(value), bottom + 19, pct(value, 0), "middle")}`;
  });
  svg += `<path d="M${left} ${y(calibration ? 0 : 0.5)}L${right} ${y(calibration ? 1 : 0.5)}" class="svg-chance"/>`;
  const sorted = [...points].sort((a, b) => a.x - b.x);
  svg += `<polyline points="${sorted.map((p) => `${x(p.x)},${y(p.y)}`).join(" ")}" class="svg-line"/>`;
  sorted.forEach((point) => {
    svg += `<circle cx="${x(point.x)}" cy="${y(point.y)}" r="3" class="svg-point"><title>${escapeHTML(point.label)}</title></circle>`;
  });
  svg += svgText(
    (left + right) / 2,
    height - 5,
    calibration ? "Model confidence" : "Trials retained as commands",
    "middle",
  );
  svg += svgText(
    left,
    10,
    calibration ? "Observed accuracy" : "Accuracy on retained trials",
  );
  $(element).innerHTML = svg;
}

function renderEvidence() {
  const overview = state.overview,
    report = state.report,
    result = overview.headline;
  const counts = Object.values(overview.counts);
  $("people-count").textContent = counts.reduce(
    (sum, count) => sum + count.subjects,
    0,
  );
  $("trial-count").textContent = counts
    .reduce((sum, count) => sum + count.trials, 0)
    .toLocaleString("en-GB");
  $("test-count").textContent = overview.counts.test.subjects;
  $("metric-score").innerHTML =
    `${(result.participant_balanced_accuracy * 100).toFixed(1)}<span>%</span>`;
  $("metric-ci").textContent =
    `95% participant-bootstrap interval: ${pct(result.bootstrap_95_ci[0])}–${pct(result.bootstrap_95_ci[1])}`;
  $("metric-accuracy").innerHTML =
    overview.operating_point.accuracy == null
      ? "—"
      : `${(overview.operating_point.accuracy * 100).toFixed(1)}<span>%</span>`;
  $("metric-coverage").textContent =
    `${pct(overview.operating_point.coverage)} coverage · ${overview.operating_point.retained} of ${result.n} test trials · ${pct(overview.threshold, 0)} threshold`;
  $("operating-caveat").textContent = overview.threshold_target_met
    ? "The confidence threshold was chosen on validation participants. Test performance and retained-trial accuracy remain estimates; this is an offline research demo."
    : "The model did not meet the validation target of 75% accuracy at useful coverage. The 75% confidence threshold is a declared fallback, not a validated operating point. These results support an exploratory demo, not reliable device control.";
  $("models-table").innerHTML = Object.entries(report.models)
    .map(
      ([key, model]) =>
        `<tr><td>${escapeHTML(model.name)}${key === report.selected ? '<span class="selected-tag">SELECTED ON VALIDATION</span>' : ""}</td><td>${pct(model.validation_participant_balanced_accuracy)}</td><td>${pct(model.test.participant_balanced_accuracy)}</td></tr>`,
    )
    .join("");
  $("split-strip").innerHTML = Object.entries(overview.counts)
    .map(
      ([key, count]) =>
        `<span>${key === "train" ? "Train" : key === "test" ? "Test" : "Validation"} · ${count.subjects} people · ${count.trials.toLocaleString("en-GB")} trials</span>`,
    )
    .join("");
  lineChart(
    "coverage-chart",
    result.selective_curve
      .filter((point) => point.retained >= 1)
      .map((point) => ({
        x: point.coverage,
        y: point.accuracy,
        label: `Threshold ${pct(point.threshold, 0)}: ${pct(point.accuracy)} accuracy, ${point.retained} trials`,
      })),
    "coverage",
  );
  $("participant-chart").innerHTML = result.participants
    .map(
      (person) =>
        `<div class="person-bar" tabindex="0" data-height="${100 * person.balanced_accuracy}" aria-label="Participant ${person.subject}: ${pct(person.balanced_accuracy)} balanced accuracy" title="S${person.subject}: ${pct(person.balanced_accuracy)}, ${person.trials} trials"><span>${person.subject}</span></div>`,
    )
    .join("");
  document.querySelectorAll(".person-bar").forEach((bar) => {
    bar.style.height = `${bar.dataset.height}%`;
  });
  $("robustness-chart").innerHTML = report.robustness
    .map(
      (row) =>
        `<div class="robustness-row"><span>${escapeHTML(row.condition)}</span><div class="robustness-track"><div data-width="${row.participant_balanced_accuracy * 100}"></div></div><strong>${pct(row.participant_balanced_accuracy)}</strong></div>`,
    )
    .join("");
  document.querySelectorAll(".robustness-track>div").forEach((bar) => {
    bar.style.width = `${bar.dataset.width}%`;
  });
  $("calibration-summary").textContent =
    `Brier score ${result.brier.toFixed(3)}; expected calibration error ${pct(result.ece)}. Temperature scaling was fitted on validation only. Lower Brier is better; the curve compares confidence with observed correctness.`;
  lineChart(
    "calibration-chart",
    result.reliability.map((bin) => ({
      x: bin.confidence,
      y: bin.accuracy,
      label: `${bin.count} trials: ${pct(bin.confidence)} confidence, ${pct(bin.accuracy)} accuracy`,
    })),
    "calibration",
  );
  const matrix = result.confusion_matrix;
  $("confusion-matrix").innerHTML =
    `<div class="matrix"><div></div><div>Left</div><div>Right</div><div>Left</div><div class="cell correct">${matrix[0][0]}</div><div class="cell">${matrix[0][1]}</div><div>Right</div><div class="cell">${matrix[1][0]}</div><div class="cell correct">${matrix[1][1]}</div></div>`;
  const recordingExclusions = overview.quality_exclusions.filter(
    (row) => !row.trial,
  ).length;
  $("quality-summary").textContent =
    `All 327 source files were checksum-verified. ${recordingExclusions} recordings at a different sampling rate and ${overview.quality_exclusions.length - recordingExclusions} individual trials were excluded under the predefined rules. Raw recordings remain available from PhysioNet.`;
}

async function replay() {
  if (state.playing) {
    stopReplay();
    $("replay-status").textContent = "Replay stopped.";
    return;
  }
  const token = ++state.replay;
  state.playing = true;
  $("replay-button").textContent = "Stop replay";
  const trials = state.trials.filter(
    (trial) => trial.subject === Number($("subject-select").value),
  );
  for (let i = 0; i < trials.length; i++) {
    if (token !== state.replay) return;
    $("trial-select").value = trials[i].id;
    $("replay-status").textContent =
      `Trial ${i + 1} of ${trials.length} · each accepted prediction moves the simulated cursor.`;
    const success = await decode(true);
    if (!success || token !== state.replay) return;
    await new Promise((resolve) => setTimeout(resolve, 1200));
  }
  if (token === state.replay) {
    stopReplay();
    $("replay-status").textContent =
      "Replay complete. Accepted commands moved the cursor; abstentions left it still.";
  }
}

async function initialise() {
  try {
    const [overview, report, trials] = await Promise.all([
      api("/api/overview"),
      api("/api/evaluation"),
      api("/api/trials"),
    ]);
    state.overview = overview;
    state.report = report;
    state.trials = trials.trials;
    $("subject-select").replaceChildren(
      ...[...new Set(state.trials.map((trial) => trial.subject))].map(
        (subject) =>
          new Option(
            `Participant ${String(subject).padStart(3, "0")}`,
            subject,
          ),
      ),
    );
    $("model-select").replaceChildren(
      ...overview.models.map(
        (model) =>
          new Option(
            `${model.name}${model.id === overview.selected ? " · selected" : ""}`,
            model.id,
          ),
      ),
    );
    $("model-select").value = overview.selected;
    overview.channels.forEach((channel) =>
      $("drop-select").add(new Option(channel, channel)),
    );
    $("threshold").value = Math.round(overview.threshold * 100);
    $("threshold-value").textContent = pct(overview.threshold, 0);
    $("threshold-note").textContent =
      "Below this level, the interface sends no command. Changing it here is exploratory.";
    [
      "subject-select",
      "trial-select",
      "model-select",
      "noise-select",
      "drop-select",
      "threshold",
      "replay-button",
      "decode-button",
    ].forEach((id) => {
      $(id).disabled = false;
    });
    fillTrials();
    renderEvidence();
    await decode(false);
  } catch (error) {
    showError(
      `The experiment could not load. ${error.message} Refresh this page to retry.`,
    );
  }
}

$("subject-select").addEventListener("change", () => {
  stopReplay();
  fillTrials();
  decode(false);
});
["trial-select", "model-select", "noise-select", "drop-select"].forEach((id) =>
  $(id).addEventListener("change", () => {
    stopReplay();
    decode(false);
  }),
);
$("threshold").addEventListener("input", () => {
  $("threshold-value").textContent = `${$("threshold").value}%`;
});
$("threshold").addEventListener("change", () => {
  stopReplay();
  decode(false);
});
$("decode-button").addEventListener("click", () => {
  stopReplay();
  decode(true);
});
$("replay-button").addEventListener("click", replay);
$("reset-button").addEventListener("click", () => {
  stopReplay();
  state.request++;
  state.cursor = 50;
  $("cursor").style.left = "50%";
  $("replay-status").textContent =
    "Cursor reset. This is a recorded-data simulation.";
  $("decode-button").disabled = false;
  $("decode-button").textContent = "Run decoder ↗";
});
new ResizeObserver(drawWaveform).observe($("waveform"));
initialise();
