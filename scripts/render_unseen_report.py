"""Generate the independent participant report from archived numerical evidence."""

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
from statistics import mean
from string import Template

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/unseen-v1"
ARTIFACTS = ROOT / "artifacts/unseen-v1"
HTML_OUTPUT = ROOT / "web/unseen.html"
JSON_OUTPUT = ROOT / "web/unseen-study.json"
MODEL_NAMES = ("bandpower", "covariance_tangent")
LABELS = {"bandpower": "Band power", "covariance_tangent": "Covariance tangent space"}
PARTITION_COUNTS = {"train": 45, "validation": 15, "test": 27}
COHORT_COUNTS = {
    "train": {"A": 31, "B": 11, "C": 3},
    "validation": {"A": 10, "B": 4, "C": 1},
    "test": {"A": 19, "B": 6, "C": 2},
}


def number(value, name, low=0, high=1):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if not low <= value <= high:
        raise ValueError(f"{name} is outside its permitted range")
    return value


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def same(actual, expected, name):
    if expected is None:
        if actual is not None:
            raise ValueError(f"{name} must be undefined when there are no contributing trials")
    elif (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10)
    ):
        raise ValueError(f"{name} does not agree with its participant records")


def validate_metrics(metrics, identities, cohorts, name):
    people = metrics["participants"]
    if not isinstance(people, list) or len(people) != len(identities):
        raise ValueError(f"{name} has incomplete participant records")
    if {person["participant"] for person in people} != set(identities):
        raise ValueError(f"{name} participant membership differs from the declared partition")
    for person in people:
        if person["cohort"] != cohorts[person["participant"]]:
            raise ValueError(f"{name} participant cohort differs from the source manifest")
        count = integer(person["trials"], "trials", 2)
        left = integer(person["left_trials"], "left trials", 1)
        right = integer(person["right_trials"], "right trials", 1)
        if left + right != count:
            raise ValueError("Left and right trial counts do not add up")
        for field in ("balanced_accuracy", "accuracy", "roc_auc", "brier", "coverage"):
            number(person[field], field)
        accepted = integer(person["accepted"], "accepted trials")
        correct = integer(person["correct_accepted"], "correct accepted trials")
        if not correct <= accepted <= count:
            raise ValueError("Accepted counts exceed their denominators")
        same(person["coverage"], accepted / count, "Participant coverage")
        same(person["accepted_accuracy"], correct / accepted if accepted else None, "Accepted accuracy")
    same(metrics["participant_count"], len(people), "Participant count")
    total = sum(person["trials"] for person in people)
    same(metrics["trials"], total, "Trial count")
    for aggregate, field in (
        ("participant_macro_balanced_accuracy", "balanced_accuracy"),
        ("participant_macro_accuracy", "accuracy"),
        ("participant_macro_auc", "roc_auc"),
        ("participant_macro_brier", "brier"),
    ):
        same(metrics[aggregate], mean(person[field] for person in people), aggregate)
    for field in ("balanced_accuracy", "accuracy", "roc_auc", "brier"):
        number(metrics["pooled"][field], f"Pooled {field}")
    matrix = metrics["pooled"]["confusion_matrix"]
    if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        raise ValueError("Expected a two class confusion matrix")
    for value in sum(matrix, []):
        integer(value, "Confusion matrix count")
    if sum(sum(row) for row in matrix) != total:
        raise ValueError("Confusion matrix does not cover all trials")
    if sum(matrix[0]) != sum(person["left_trials"] for person in people) or sum(matrix[1]) != sum(
        person["right_trials"] for person in people
    ):
        raise ValueError("Confusion matrix class counts disagree")
    same(metrics["pooled"]["accuracy"], (matrix[0][0] + matrix[1][1]) / total, "Pooled accuracy")
    same(
        metrics["pooled"]["balanced_accuracy"],
        (matrix[0][0] / sum(matrix[0]) + matrix[1][1] / sum(matrix[1])) / 2,
        "Pooled balanced accuracy",
    )
    observed_cohorts = {row["cohort"]: row for row in metrics["cohorts"]}
    if len(observed_cohorts) != len(metrics["cohorts"]) or set(observed_cohorts) != {
        p["cohort"] for p in people
    }:
        raise ValueError("Cohort summaries are incomplete or duplicated")
    for cohort, summary in observed_cohorts.items():
        subset = [person for person in people if person["cohort"] == cohort]
        same(summary["participant_count"], len(subset), "Cohort participant count")
        same(
            summary["participant_macro_balanced_accuracy"],
            mean(person["balanced_accuracy"] for person in subset),
            "Cohort balanced accuracy",
        )
    selective = metrics["selective"]
    number(selective["threshold"], "Threshold", 0.5, 1)
    accepted = sum(person["accepted"] for person in people)
    correct = sum(person["correct_accepted"] for person in people)
    defined = [person["accepted_accuracy"] for person in people if person["accepted"]]
    expected = {
        "accepted": accepted,
        "correct_accepted": correct,
        "pooled_accuracy": correct / accepted if accepted else None,
        "pooled_coverage": accepted / total,
        "participant_macro_accuracy": mean(defined) if defined else None,
        "participant_macro_coverage": mean(person["coverage"] for person in people),
        "participants_with_acceptance": len(defined),
        "participants_without_acceptance": len(people) - len(defined),
        "minimum_accepted_per_participant": min(person["accepted"] for person in people),
    }
    for field, value in expected.items():
        same(selective[field], value, f"Selective {field}")


def validate_report(report, split_manifest=None, source_manifest=None):
    if report["schema_version"] != "1.0" or report["kind"] != "unseen_participants":
        raise ValueError("Unexpected independent participant report schema")
    if report["experiment"] != "unseen-v1" or report["dataset"] != "Dreyer2023":
        raise ValueError("This renderer is specific to the declared Dreyer2023 experiment")
    if report["production_model_changed"] is not False:
        raise ValueError("The independent study must not change the deployed model")
    split_manifest = split_manifest or json.loads((EXPERIMENT / "split_manifest.json").read_text())
    source_manifest = source_manifest or json.loads((EXPERIMENT / "source_manifest.json").read_text())
    source_cohorts = {row["participant"]: row["cohort"] for row in source_manifest["participants"]}
    declared = split_manifest["participants"]
    if (
        len(declared) != 87
        or len(source_cohorts) != 87
        or {row["participant"] for row in declared} != set(source_cohorts)
    ):
        raise ValueError("The source and split manifests must cover all 87 participants")
    if set(report["splits"]) != set(PARTITION_COUNTS) or set(report["models"]) != set(MODEL_NAMES):
        raise ValueError("Every declared partition and model must be present")
    combined = []
    for split, expected_count in PARTITION_COUNTS.items():
        members = report["splits"][split]
        planned = {row["participant"] for row in declared if row["split"] == split}
        if len(members) != expected_count or set(members) != planned:
            raise ValueError(f"{split} participant identities differ from the frozen manifest")
        if Counter(source_cohorts[person] for person in members) != COHORT_COUNTS[split]:
            raise ValueError("Source cohort counts differ from the protocol")
        combined.extend(members)
        if report["counts"][split]["participants"] != expected_count:
            raise ValueError("Reported split counts differ from the participant identifiers")
        integer(report["counts"][split]["trials"], "Partition trials", 2 * expected_count)
    if len(set(combined)) != 87:
        raise ValueError("Participants overlap between partitions")
    for model in MODEL_NAMES:
        result = report["models"][model]
        decision = result["threshold_selection"]
        threshold = decision["threshold"]
        number(threshold, "Selected threshold", 0.5, 1)
        if not isinstance(decision["validation_target_met"], bool):
            raise ValueError("Threshold selection must record its validation outcome")
        rule = decision["rule"]
        expected_rule = {
            "thresholds": [value / 100 for value in range(50, 100, 5)],
            "target_participant_macro_accuracy": 0.75,
            "target_participant_macro_coverage": 0.20,
            "minimum_retained_trials": 100,
            "minimum_retained_per_participant": 5,
            "fallback_threshold": 0.75,
        }
        if rule != expected_rule:
            raise ValueError("Threshold rule differs from the frozen study protocol")
        candidates = decision["candidates"]
        if [row["threshold"] for row in candidates] != rule["thresholds"]:
            raise ValueError("Threshold candidates differ from the frozen ascending grid")
        qualifying = [
            row
            for row in candidates
            if row["participant_macro_accuracy"] is not None
            and row["participant_macro_accuracy"] >= rule["target_participant_macro_accuracy"]
            and row["participant_macro_coverage"] >= rule["target_participant_macro_coverage"]
            and row["accepted"] >= rule["minimum_retained_trials"]
            and row["minimum_accepted_per_participant"] >= rule["minimum_retained_per_participant"]
        ]
        expected_threshold = qualifying[0]["threshold"] if qualifying else rule["fallback_threshold"]
        if threshold != expected_threshold or decision["validation_target_met"] != bool(qualifying):
            raise ValueError("Threshold selection does not follow its recorded validation candidates")
        for split in ("validation", "test"):
            metrics = result[split]
            validate_metrics(metrics, report["splits"][split], source_cohorts, f"{model} {split}")
            if metrics["trials"] != report["counts"][split]["trials"]:
                raise ValueError("Model trial counts differ from the declared partition")
            if metrics["selective"]["threshold"] != threshold:
                raise ValueError("The selected threshold must stay fixed between validation and test")
        selected_candidate = next(row for row in candidates if row["threshold"] == threshold)
        for field, value in selected_candidate.items():
            same(result["validation"]["selective"][field], value, f"Selected threshold {field}")
    expected_selected = max(
        MODEL_NAMES,
        key=lambda name: report["models"][name]["validation"]["participant_macro_balanced_accuracy"],
    )
    selected = report["selection"]["model"]
    if selected != expected_selected:
        raise ValueError("Model selection does not follow validation balanced accuracy")
    bootstrap = report["test_bootstrap"]
    if (
        bootstrap["unit"] != "participant"
        or set(bootstrap["participant_ids"]) != set(report["splits"]["test"])
        or len(bootstrap["participant_ids"]) != 27
    ):
        raise ValueError("Bootstrap must resample the 27 complete test participants")
    if bootstrap["strata_counts"] != COHORT_COUNTS["test"]:
        raise ValueError("Bootstrap cohort counts differ from the frozen split")
    integer(bootstrap["repeats"], "Bootstrap repeats", 1)
    integer(bootstrap["seed"], "Bootstrap seed", 0)
    if bootstrap["repeats"] != 5000 or bootstrap["seed"] != 2027:
        raise ValueError("Bootstrap must use the frozen 5000 repetitions and seed 2027")
    if set(bootstrap["model_intervals"]) != set(MODEL_NAMES):
        raise ValueError("Bootstrap intervals must include both models")
    for interval in bootstrap["model_intervals"].values():
        if len(interval) != 2:
            raise ValueError("Expected two bootstrap interval bounds")
        low, high = (number(value, "Bootstrap bound") for value in interval)
        if low > high:
            raise ValueError("Bootstrap bounds are reversed")
    primary = report["selection"]["primary_criterion_met"]
    if not isinstance(primary, bool) or primary != (bootstrap["model_intervals"][selected][0] > 0.5):
        raise ValueError("Primary decision must use the selected model's lower test bound")
    difference = bootstrap["paired_difference"]
    if difference["direction"] != "covariance_tangent minus bandpower":
        raise ValueError("Unexpected paired comparison direction")
    same(
        difference["estimate"],
        report["models"]["covariance_tangent"]["test"]["participant_macro_balanced_accuracy"]
        - report["models"]["bandpower"]["test"]["participant_macro_balanced_accuracy"],
        "Paired difference",
    )
    bounds = difference["ci95"]
    if len(bounds) != 2 or not -1 <= bounds[0] <= bounds[1] <= 1:
        raise ValueError("Invalid paired difference interval")
    for field in ("protocol_commit", "selection_commit"):
        if not re.fullmatch(r"[0-9a-f]{7,40}", report[field]):
            raise ValueError(f"{field} must identify a saved git revision")
    datetime.fromisoformat(report["completed_utc"].replace("Z", "+00:00"))
    return report


def pct(value, digits=2):
    return "—" if value is None else f"{100 * value:.{digits}f}%"


def table(caption, headings, rows, identifier):
    heading = "".join(f'<th scope="col">{escape(str(value))}</th>' for value in headings)
    contents = []
    for row in rows:
        cells = [f'<th scope="row">{escape(str(row[0]))}</th>']
        cells.extend(f"<td>{escape(str(value))}</td>" for value in row[1:])
        contents.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f'<div class="report-table-wrap" tabindex="0" role="region" aria-label="{escape(caption, quote=True)}">'
        f'<table class="report-table" id="{identifier}"><caption>{escape(caption)}</caption>'
        f"<thead><tr>{heading}</tr></thead><tbody>{''.join(contents)}</tbody></table></div>"
    )


def participant_table(metrics, caption, identifier):
    return table(
        caption,
        [
            "Participant",
            "Cohort",
            "Left / right trials",
            "Balanced accuracy",
            "Brier",
            "Accepted / all",
            "Coverage",
            "Accepted accuracy",
        ],
        [
            [
                person["participant"],
                person["cohort"],
                f"{person['left_trials']} / {person['right_trials']}",
                pct(person["balanced_accuracy"]),
                f"{person['brier']:.4f}",
                f"{person['accepted']} / {person['trials']}",
                pct(person["coverage"]),
                pct(person["accepted_accuracy"]),
            ]
            for person in sorted(metrics["participants"], key=lambda row: row["participant"])
        ],
        identifier,
    )


def participant_plot(metrics):
    """Use percentage positions with fixed text sizes for a readable mobile SVG."""
    people = sorted(metrics["participants"], key=lambda row: row["participant"])
    height = 88 + len(people) * 20
    bottom = 60 + (len(people) - 1) * 20
    elements = [
        '<figure class="report-table-wrap">',
        f'<svg width="100%" height="{height}" role="img" aria-labelledby="unseen-plot-title unseen-plot-description">',
        '<title id="unseen-plot-title">Balanced accuracy for every reserved test participant</title>',
        '<desc id="unseen-plot-description">Each point represents one participant under the selected model, ordered by participant identifier; the dashed line marks the 50 percent chance reference, and the following table provides the same values and trial counts</desc>',
        '<text x="2%" y="18" fill="#b1b1b1" font-size="13">Person · Cohort</text>',
        '<text x="98%" y="18" text-anchor="end" fill="#b1b1b1" font-size="13">Accuracy</text>',
    ]
    for value in (0, 50, 100):
        position = 26 + 52 * value / 100
        elements.append(
            f'<text x="{position}%" y="38" text-anchor="middle" fill="#b1b1b1" font-size="13">{value}%</text>'
        )
    elements.append(
        f'<line x1="52%" x2="52%" y1="46" y2="{bottom + 10}" stroke="#b1b1b1" stroke-width="1" stroke-dasharray="4 4" />'
    )
    for index, person in enumerate(people):
        y = 60 + 20 * index
        participant = escape(person["participant"], quote=True)
        cohort = escape(person["cohort"])
        score = pct(person["balanced_accuracy"])
        position = 26 + 52 * person["balanced_accuracy"]
        elements.extend(
            [
                f'<line x1="26%" x2="78%" y1="{y}" y2="{y}" stroke="#262626" stroke-width="1" />',
                f'<text x="2%" y="{y + 4}" fill="#ededed" font-size="13">{participant} · {cohort}</text>',
                f'<circle data-participant="{participant}" data-score="{person["balanced_accuracy"]}" cx="{position:.6f}%" cy="{y}" r="4" fill="#39ff14"><title>{participant}, cohort {cohort}, balanced accuracy {score}</title></circle>',
                f'<text x="98%" y="{y + 4}" text-anchor="end" fill="#ededed" font-size="13">{score}</text>',
            ]
        )
    elements.extend(
        [
            "</svg>",
            '<figcaption class="report-status">Each participant has equal weight in the primary mean; the dashed line is the 50% reference, and individual points are descriptive rather than tests of significance</figcaption>',
            "</figure>",
        ]
    )
    return "".join(elements)


def validation_threshold_note(selection, metrics):
    """Explain the frozen rule against its actual validation operating point."""
    rule, values = selection["rule"], metrics["selective"]
    failures = []
    if values["participant_macro_accuracy"] is None:
        failures.append("conditional mean participant accuracy is undefined without accepted predictions")
    elif values["participant_macro_accuracy"] < rule["target_participant_macro_accuracy"]:
        failures.append(
            f"conditional mean participant accuracy is {pct(values['participant_macro_accuracy'])}, below {pct(rule['target_participant_macro_accuracy'], 0)}"
        )
    if values["participant_macro_coverage"] < rule["target_participant_macro_coverage"]:
        failures.append(
            f"mean participant coverage is {pct(values['participant_macro_coverage'])}, below {pct(rule['target_participant_macro_coverage'], 0)}"
        )
    if values["accepted"] < rule["minimum_retained_trials"]:
        failures.append(
            f"{values['accepted']} accepted trials fall below the minimum of {rule['minimum_retained_trials']}"
        )
    below_minimum = sum(
        person["accepted"] < rule["minimum_retained_per_participant"] for person in metrics["participants"]
    )
    if below_minimum:
        failures.append(
            f"{below_minimum} of {metrics['participant_count']} validation participants {'has' if below_minimum == 1 else 'have'} fewer than "
            f"{rule['minimum_retained_per_participant']} accepted trials, with a lowest count of {values['minimum_accepted_per_participant']}"
        )
    point = "selected threshold" if selection["validation_target_met"] else "fallback threshold"
    opening = f"At the {point}, pooled accuracy among accepted validation predictions is {pct(values['pooled_accuracy'])}; "
    if failures:
        return opening + "the unmet requirements are: " + "; ".join(failures)
    return (
        opening
        + f"all {metrics['participant_count']} validation participants contribute at least "
        + f"{values['minimum_accepted_per_participant']} accepted trials, meeting the declared minimum alongside the accuracy and coverage requirements"
    )


def render(report):
    validate_report(report)
    selected = report["selection"]["model"]
    result = report["models"][selected]
    metrics, selective = result["test"], result["test"]["selective"]
    lower, upper = report["test_bootstrap"]["model_intervals"][selected]
    criterion = report["selection"]["primary_criterion_met"]
    selection = result["threshold_selection"]
    rule = selection["rule"]
    primary_finding = (
        f"{LABELS[selected]} was selected using validation participants and reached "
        f"{pct(metrics['participant_macro_balanced_accuracy'])} mean balanced accuracy across the 27 reserved test participants, "
        f"with a 95% participant bootstrap interval of {pct(lower)} to {pct(upper)}; "
        f"the declared primary criterion {'was met because the lower bound exceeds' if criterion else 'was not met because the lower bound does not exceed'} the 50% chance reference"
    )
    primary_scope = (
        "This supports above chance classification under this recording protocol, while visual cues and other possible confounds limit any claim about intention itself"
        if criterion
        else "This study does not meet its declared evidence criterion for above chance performance in the reserved participants; the saved result remains useful for inspecting where performance varies"
    )
    validation_target = (
        "met" if selection["validation_target_met"] else "not met, so the declared fallback was used"
    )
    selective_finding = (
        f"The separate threshold target was {validation_target} on validation data; at the fixed threshold of "
        f"{selection['threshold']:.2f}, the test model accepted {selective['accepted']} of {metrics['trials']} trials, "
        f"with {pct(selective['participant_macro_coverage'])} mean participant coverage and "
        f"{selective['participants_without_acceptance']} participants receiving no accepted predictions"
    )
    split_rows = [
        [name.capitalize(), report["counts"][name]["participants"], report["counts"][name]["trials"], role]
        for name, role in (
            ("train", "Fit transforms and both models"),
            ("validation", "Choose the model and its threshold"),
            ("test", "Evaluate once with those choices fixed"),
        )
    ]
    cohort_rows = [
        [cohort, *[COHORT_COUNTS[split][cohort] for split in PARTITION_COUNTS]] for cohort in ("A", "B", "C")
    ]
    model_rows, selective_rows = [], []
    for name in MODEL_NAMES:
        entry = report["models"][name]
        interval = report["test_bootstrap"]["model_intervals"][name]
        label = LABELS[name] + (" · Selected" if name == selected else " · Descriptive comparison")
        model_rows.append(
            [
                label,
                pct(entry["validation"]["participant_macro_balanced_accuracy"]),
                pct(entry["test"]["participant_macro_balanced_accuracy"]),
                f"{pct(interval[0])} to {pct(interval[1])}",
                f"{entry['test']['participant_macro_brier']:.4f}",
            ]
        )
    for split in ("validation", "test"):
        values = result[split]["selective"]
        selective_rows.append(
            [
                split.capitalize(),
                f"{values['accepted']} / {result[split]['trials']}",
                pct(values["participant_macro_coverage"]),
                pct(values["participant_macro_accuracy"]),
                pct(values["pooled_accuracy"]),
                f"{values['participants_with_acceptance']} / {result[split]['participant_count']}",
            ]
        )
    difference = report["test_bootstrap"]["paired_difference"]
    paired_finding = (
        f"Covariance tangent space minus band power gives a paired difference of {difference['estimate'] * 100:+.2f} "
        f"percentage points on the test participants, with a 95% participant bootstrap interval from "
        f"{difference['ci95'][0] * 100:+.2f} to {difference['ci95'][1] * 100:+.2f} percentage points; this secondary comparison does not revise model selection"
    )
    test_cohorts = metrics["cohorts"]
    actual_trials = sum(report["counts"][split]["trials"] for split in PARTITION_COUNTS)
    nominal_trials = 87 * 2 * 40
    count_comparison = (
        "matches the nominal total"
        if actual_trials == nominal_trials
        else f"is {abs(actual_trials - nominal_trials):,} {'below' if actual_trials < nominal_trials else 'above'} the nominal total"
    )
    values = {
        "report_date": datetime.fromisoformat(report["completed_utc"].replace("Z", "+00:00"))
        .strftime("%d %B %Y")
        .lstrip("0"),
        "primary_finding": escape(primary_finding),
        "primary_scope": escape(primary_scope),
        "selective_finding": escape(selective_finding),
        "validation_failure_note": escape(validation_threshold_note(selection, result["validation"])),
        "split_table": table(
            "The three partitions have no participants in common",
            ["Partition", "Participants", "Trials", "Permitted use"],
            split_rows,
            "unseen-splits",
        ),
        "trial_count_note": escape(
            f"The source protocol describes two acquisition runs of 40 trials per participant; the saved analysis contains {actual_trials:,} eligible trials, "
            f"which {count_comparison} of {nominal_trials:,}; all tables retain actual recorded counts rather than assigning 80 trials to every person, "
            "and the provenance records the inspected files and trial eligibility"
        ),
        "cohort_split_table": table(
            "Participant counts within each source cohort",
            ["Source cohort", "Training", "Validation", "Test"],
            cohort_rows,
            "unseen-cohort-splits",
        ),
        "split_membership": "".join(
            f"<p><strong>{split.capitalize()}</strong><br />{escape(', '.join(report['splits'][split]))}</p>"
            for split in PARTITION_COUNTS
        ),
        "preprocessing_details": "All 27 named EEG channels form the common average reference before the same nine central channels are retained; each completed epoch is demeaned, filtered from 8 to 30 Hz in both directions and divided by one root mean square value across its channels and samples; the models use either log power in three fixed frequency bands or trace normalised covariance matrices",
        "threshold_rule": escape(
            f"The threshold rule requires at least {pct(rule['target_participant_macro_accuracy'], 0)} conditional mean participant accuracy, "
            f"{pct(rule['target_participant_macro_coverage'], 0)} mean participant coverage, {rule['minimum_retained_trials']} accepted validation trials "
            f"and {rule['minimum_retained_per_participant']} accepted trials for every validation participant; the first qualifying threshold in the fixed ascending grid is used, "
            f"otherwise the fallback is {rule['fallback_threshold']:.2f}; probabilities have not been independently calibrated"
        ),
        "bootstrap_details": escape(
            f"The 95% intervals resample complete test participants {report['test_bootstrap']['repeats']:,} times within each source cohort, "
            "preserving the cohort counts and pairing the two models on the same resampled people; trials are not treated as independent people, "
            "and the interval is conditional on these fitted models and the observed cohort composition"
        ),
        "model_table": table(
            "Equal participant weighting; test intervals use participant resampling",
            [
                "Model",
                "Validation balanced accuracy",
                "Test balanced accuracy",
                "Test 95% interval",
                "Test mean Brier",
            ],
            model_rows,
            "unseen-models",
        ),
        "selective_table": table(
            f"Selected model: {LABELS[selected]} at threshold {selection['threshold']:.2f}",
            [
                "Partition",
                "Accepted / all trials",
                "Mean participant coverage",
                "Conditional mean accuracy",
                "Pooled accepted accuracy",
                "People with acceptance / all",
            ],
            selective_rows,
            "unseen-selective",
        ),
        "paired_finding": escape(paired_finding),
        "participant_table": participant_table(
            metrics, f"Every test participant under {LABELS[selected]}", "unseen-test-participants"
        ),
        "participant_plot": participant_plot(metrics),
        "validation_participant_table": participant_table(
            result["validation"],
            f"Validation participants under {LABELS[selected]}",
            "unseen-validation-participants",
        ),
        "cohort_result_table": table(
            "Descriptive cohort results for the selected model",
            ["Source cohort", "Test participants", "Mean balanced accuracy"],
            [
                [row["cohort"], row["participant_count"], pct(row["participant_macro_balanced_accuracy"])]
                for row in test_cohorts
            ],
            "unseen-cohort-results",
        ),
        "protocol_commit": report["protocol_commit"][:7],
        "selection_commit": report["selection_commit"][:7],
        "protocol_url": f"https://github.com/shrut10/intentlab-bci/commit/{report['protocol_commit']}",
        "selection_url": f"https://github.com/shrut10/intentlab-bci/commit/{report['selection_commit']}",
    }
    return Template((EXPERIMENT / "report.template.html").read_text()).substitute(values)


def numerical_export(report):
    validate_report(report)
    return json.dumps(report, allow_nan=False, separators=(",", ":")) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = json.loads((ARTIFACTS / "evaluation.json").read_text())
    expected = {HTML_OUTPUT: render(report), JSON_OUTPUT: numerical_export(report)}
    for destination, content in expected.items():
        if args.check:
            if not destination.exists() or destination.read_text() != content:
                raise SystemExit(f"Published independent participant report is stale: {destination.name}")
        else:
            destination.write_text(content)
    print(
        "Published independent participant report matches the archived evaluation"
        if args.check
        else "Wrote independent participant page and numerical report"
    )


if __name__ == "__main__":
    main()
