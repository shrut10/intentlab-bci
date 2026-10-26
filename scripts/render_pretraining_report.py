"""Render the validation-only report from archived evidence; CI uses --check."""

import argparse
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "experiments/pretraining-v1/report.template.html"
REPORT = ROOT / "artifacts/pretraining-v1/evaluation.json"
OUTPUT = ROOT / "web/pretraining.html"


def budget_label(fraction):
    return "10% per participant/class" if fraction == 0.1 else "All source labels"


def table(caption, headings, rows):
    header = "".join(f'<th scope="col">{heading}</th>' for heading in headings)
    body = "".join("<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>" for row in rows)
    return (
        '<div class="report-table-wrap"><table class="report-table">'
        f"<caption>{caption}</caption><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def render(report):
    if report["production_model_changed"] or report["test_evaluated"]:
        raise ValueError("This page describes only the isolated validation experiment")
    if len(report["results"]) != 12 or len(report["summary"]) != 2:
        raise ValueError("All seeds, budgets and arms must be present before publication")
    result_rows, paired_rows, interpretations = [], [], []
    for row in sorted(report["summary"], key=lambda item: item["label_fraction"]):
        fraction = row["label_fraction"]
        counts = sorted(set(row["labelled_trials_per_seed"].values()))
        labelled = ", ".join(str(n) for n in counts)
        result_rows.append(
            [
                f"{budget_label(fraction)}<br />{labelled} labelled trials",
                f"{row['scratch_mean_balanced_accuracy']:.2%}",
                f"{row['pretrained_mean_balanced_accuracy']:.2%}",
                *[
                    f"{mean(r['validation']['participant_brier'] for r in report['results'] if r['label_fraction'] == fraction and r['arm'] == arm):.4f}"
                    for arm in ("scratch", "masked_pretraining")
                ],
            ]
        )
        low, high = row["paired_difference_95_ci"]
        difference = row["paired_difference"]
        paired_rows.append(
            [
                budget_label(fraction),
                f"{difference * 100:+.2f} pp",
                f"{low * 100:+.2f} to {high * 100:+.2f} pp",
            ]
        )
        direction = "higher" if difference > 0 else "lower"
        interval_note = (
            "the participant-bootstrap interval includes zero"
            if low <= 0 <= high
            else "the participant-bootstrap interval excludes zero on this known cohort"
        )
        interpretations.append(
            f"With {labelled} supervised labels, pretrained balanced accuracy was "
            f"{abs(difference) * 100:.2f} percentage points {direction} than scratch; {interval_note}."
        )
    seed_rows = [
        [
            str(row["seed"]),
            str(row["labelled_trials"]),
            "Masked pretraining" if row["arm"] == "masked_pretraining" else "Scratch",
            f"{row['validation']['participant_balanced_accuracy']:.2%}",
            f"{row['validation']['participant_brier']:.4f}",
        ]
        for row in report["results"]
    ]
    replacements = {
        "INTERPRETATION": " ".join(interpretations)
        + " These findings are exploratory and do not alter the live decoder.",
        "RESULTS": table(
            "Validation only: 22 people, 989 trials; means across three seeds. BA = participant-macro balanced accuracy.",
            ["Source label budget", "Scratch BA", "Pretrained BA", "Scratch Brier", "Pretrained Brier"],
            result_rows,
        ),
        "PAIRED_RESULTS": table(
            "Pretraining minus scratch; seed-averaged paired participant bootstrap, 2,000 resamples. pp = percentage points.",
            ["Source label budget", "Difference", "95% interval"],
            paired_rows,
        ),
        "SEED_RESULTS": table(
            "Every completed run; no best-seed selection.",
            ["Seed", "Labels", "Training", "Validation BA", "Brier"],
            seed_rows,
        ),
    }
    html = TEMPLATE.read_text()
    for name, value in replacements.items():
        html = html.replace("{{" + name + "}}", value)
    if "{{" in html:
        raise ValueError("Unfilled report template field")
    return html


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render(json.loads(REPORT.read_text()))
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != expected:
            raise SystemExit("Pretraining report is stale; run python scripts/render_pretraining_report.py")
        print("Pretraining report matches the archived study")
    else:
        OUTPUT.write_text(expected)
        print(OUTPUT)


if __name__ == "__main__":
    main()
