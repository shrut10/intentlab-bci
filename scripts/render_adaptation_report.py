"""Render publication tables from the archived study, never from hand-entered scores."""

import argparse
import json
from datetime import datetime
from html import escape
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_COMMIT = "e78292e"


def pct(value, digits=2):
    return f"{100 * value:.{digits}f}%" if value is not None else "No accepted predictions"


def render(report, interpretation):
    summary, results = report["summary"], report["results"]
    baseline, alignment = summary["baseline"], summary["alignment"]
    delta = alignment["paired_change_vs_baseline"]["balanced_accuracy"]
    low, high = delta["bootstrap_95_ci"]
    evidence = (
        "includes no change" if low <= 0 <= high else "lies above zero" if low > 0 else "lies below zero"
    )
    overall = (
        f"Mean participant-macro balanced accuracy was {pct(baseline['mean_participant_balanced_accuracy'])} "
        f"for the baseline and {pct(alignment['mean_participant_balanced_accuracy'])} for Euclidean alignment, "
        f"a paired change of {100 * delta['mean_difference']:+.2f} percentage points "
        f"(95% participant-bootstrap interval {100 * low:+.2f} to {100 * high:+.2f}). "
        f"The interval {evidence}, conditional on this split and these three fitted model pairs."
    )
    target_count = sum(s["validation_target_met_seeds"] for s in summary.values())
    reliability = (
        "None of the twelve condition–seed combinations met the predefined validation reliability target, "
        "so every operating point uses the declared fallback; this comparison does not establish reliable control."
        if not target_count
        else f"The predefined validation reliability target was met in {target_count} of twelve condition–seed combinations; "
        "the remaining combinations use the declared fallback. Passing that criterion is an offline result, not a validation of reliable device control."
    )
    personal = summary["personal_temperature"]
    combined = summary["alignment_temperature"]
    brier_change = personal["paired_change_vs_baseline"]["brier"]
    brier_low, brier_high = brier_change["bootstrap_95_ci"]
    calibration = (
        "Personal temperature scaling leaves each model’s classifications unchanged, as expected. "
        f"Its mean test Brier score was {personal['mean_brier']:.4f}, compared with {baseline['mean_brier']:.4f} "
        f"for the global-temperature baseline. With alignment, the corresponding values were {combined['mean_brier']:.4f} "
        f"for personal temperature and {alignment['mean_brier']:.4f} for global temperature. "
        "These comparisons describe probability quality rather than an improvement in discrimination. "
        f"For personal temperature versus baseline, the paired Brier change was {brier_change['mean_difference']:+.4f} "
        f"(95% participant-bootstrap interval {brier_low:+.4f} to {brier_high:+.4f}); "
        f"this interval {'includes no change' if brier_low <= 0 <= brier_high else 'excludes zero'} "
        "under the same conditional interpretation as the accuracy comparison."
    )
    comparison_rows, coverage_rows, seed_rows, threshold_rows = [], [], [], []
    for key, entry in summary.items():
        difference = entry["paired_change_vs_baseline"]["balanced_accuracy"]
        lo, hi = difference["bootstrap_95_ci"]
        comparison = (
            "Reference"
            if key == "baseline"
            else f"{100 * difference['mean_difference']:+.2f} pp ({100 * lo:+.2f}, {100 * hi:+.2f})"
        )
        comparison_rows.append(
            f'<tr data-condition="{key}"><th scope="row">{escape(entry["name"])}</th><td>{pct(entry["mean_participant_balanced_accuracy"])}</td><td>{comparison}</td><td>{entry["mean_brier"]:.4f}</td><td>{pct(entry["mean_ece"])}</td></tr>'
        )
        operating = entry["operating_points"]["fixed_threshold"]
        coverage_rows.append(
            f'<tr data-coverage="{key}"><th scope="row">{escape(entry["name"])}</th><td>{pct(operating["mean_coverage"])}</td><td>{operating["mean_retained_per_seed"]:.1f}</td><td>{pct(operating["pooled_repeated_prediction_accuracy"])}</td><td>{entry["validation_target_met_seeds"]}/3</td></tr>'
        )
    for record in results:
        metrics = record["test"]["metrics"]
        name = escape(record["name"])
        seed_rows.append(
            f'<tr><th scope="row">{name}</th><td>{record["seed"]}</td><td>{pct(metrics["participant_balanced_accuracy"])}</td><td>{metrics["brier"]:.4f}</td><td>{pct(metrics["ece"])}</td></tr>'
        )
        selected, point = record["validation"]["threshold_rule"], record["test"]["validation_threshold"]
        threshold_rows.append(
            f'<tr><th scope="row">{name} / {record["seed"]}</th><td>{selected["threshold"]:.2f}</td><td>{"Yes" if selected["target_met"] else "No — fallback"}</td><td>{point["retained"]}/{point["n"]}</td><td>{point["correct"]}</td><td>{pct(point["accuracy"])}</td></tr>'
        )
    counts = report["counts"]
    audit = report["personal_temperature_audit"]["test"]
    revision = report["run"]["code_revision"]
    values = {
        "report_date": datetime.fromisoformat(report["completed_utc"]).strftime("%d %B %Y").lstrip("0"),
        "test_trials": counts["test"]["evaluation_trials"],
        "test_people": counts["test"]["participants"],
        "validation_trials": counts["validation"]["evaluation_trials"],
        "validation_people": counts["validation"]["participants"],
        "train_people": counts["train"]["participants"],
        "repeated_trials": counts["test"]["evaluation_trials"] * len(report["config"]["seeds"]),
        "overall_finding": overall,
        "reliability_finding": reliability,
        "calibration_finding": calibration,
        "comparison_rows": "\n".join(comparison_rows),
        "coverage_rows": "\n".join(coverage_rows),
        "seed_rows": "\n".join(seed_rows),
        "threshold_rows": "\n".join(threshold_rows),
        "unaligned_bounds": audit["unaligned"]["at_bound"],
        "aligned_bounds": audit["aligned"]["at_bound"],
        "personal_fits": audit["unaligned"]["fits"],
        "personal_fallbacks": sum(v["fallbacks"] for v in audit.values()),
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_commit_url": f"https://github.com/shrut10/intentlab-bci/commit/{PROTOCOL_COMMIT}",
        "protocol_url": f"https://github.com/shrut10/intentlab-bci/blob/{PROTOCOL_COMMIT}/experiments/adaptation-v1/PROTOCOL.md",
        "code_commit": revision[:7],
        "code_commit_url": f"https://github.com/shrut10/intentlab-bci/commit/{revision}",
        "parity": f"{max(m['onnx_max_absolute_logit_error'] for m in report['models']):.2e}",
        "interpretation": escape(interpretation.strip()),
    }
    template = (ROOT / "experiments/adaptation-v1/report.template.html").read_text()
    return Template(template).substitute(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = json.loads((ROOT / "artifacts/adaptation-v1/evaluation.json").read_text())
    interpretation = (ROOT / "experiments/adaptation-v1/interpretation.txt").read_text()
    html = render(report, interpretation)
    output = ROOT / "web/adaptation.html"
    if args.check:
        if not output.exists() or output.read_text() != html:
            raise SystemExit("Published report differs from archived results/template; regenerate it")
        print("Published adaptation report matches archived results")
    else:
        output.write_text(html)
        print(f"Rendered {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
