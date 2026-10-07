# Calibration and Euclidean alignment follow-up

This experiment addresses [IntentLab issue #1](https://github.com/shrut10/intentlab-bci/issues/1) by comparing a matched CNN baseline, participant-specific confidence calibration, Euclidean alignment, and their combination. The production model is unchanged.

[Published report](https://intentlab-bci.vercel.app/research/adaptation) · [Frozen protocol](PROTOCOL.md) · [Configuration](protocol.json) · [Archived numerical results](../../artifacts/adaptation-v1/evaluation.json)

## What was implemented

- Participant-specific alignment matrices fitted to the first ten eligible run-04 epochs, without using labels or later-run recordings.
- Positive bounded temperatures fitted on the same ten early trials with known labels; single-class blocks fall back to the global temperature.
- Consistently trained unaligned/aligned model pairs at seeds 2027, 2028 and 2029, with validation-based checkpoint selection.
- Evaluation on the same 630 later-run epochs from 21 held-out participants, plus paired participant-bootstrap comparisons.
- Validation-only rejection-threshold selection and a fixed 0.75 comparison, reporting accuracy and coverage together.
- Six ONNX checkpoints, full validation histories, calibration matrices, temperature audits, trial-level probabilities and SHA-256 manifests.

This is exploratory reanalysis of a previously inspected benchmark. Ten-trial adaptation changes the deployment assumption from zero-calibration transfer; it does not validate a live headset, an independent dataset or a smaller montage. The implementation is regularised Euclidean alignment, not Riemannian alignment.

## Files

| Purpose | Location |
| --- | --- |
| Fixed study design | `experiments/adaptation-v1/PROTOCOL.md` and `protocol.json` |
| Alignment, calibration and metrics | `src/intentlab/adaptation.py` |
| Six-model training/evaluation runner | `scripts/run_adaptation.py` |
| Completed study, predictions and ONNX models | `artifacts/adaptation-v1/` |
| Academic page wording and layout | `experiments/adaptation-v1/report.template.html` |
| Interpretation paragraph | `experiments/adaptation-v1/interpretation.txt` |
| Generated public page | `web/adaptation.html` |
| Deterministic page renderer | `scripts/render_adaptation_report.py` |
| Isolation and numerical checks | `tests/test_adaptation.py`, `tests/test_adaptation_results.py` |

## Reproduce in a separate checkout

Use the pinned Python 3.12 training environment. Obtain the original source data through `scripts/prepare_data.py`; it verifies the publisher's checksums and reproduces the epoch archive. The supplied `artifacts/adaptation-v1` directory contains a completed run, so move it to a separate archive in the reproduction checkout before running the experiment. The runner intentionally refuses to overwrite completed results.

```sh
pip install -r requirements-training.txt
python scripts/prepare_data.py
python scripts/run_adaptation.py
python scripts/render_adaptation_report.py
pytest -q
```

For an interrupted run only, `python scripts/run_adaptation.py --resume` checks the recorded protocol, implementation and input hashes before reusing complete checkpoints. It refuses changed inputs or code; do not use it to revise the method after inspecting test results.

The original train/validation/test participant assignment is preserved. Later runs 08 and 12 are separate recording blocks within this dataset, not separate-day sessions. The ten early target trials are part of the declared adaptation budget, not additional supervised training subjects.

## Edit the report

Change prose in `report.template.html` and the interpretation in `interpretation.txt`, then run `python scripts/render_adaptation_report.py`. Table values and quantitative findings are generated directly from the immutable study JSON. CI checks the generated page and independently reconstructs the reported scores and validation threshold choices from the saved probabilities.

The public API exposes `/api/adaptation` and `/api/adaptation/predictions` for archived results and a Parquet prediction download. These routes do not train a model or accept private EEG uploads. The original `/api/predict` continues using the original model and threshold.

## Result interpretation

The public report includes every predeclared condition and seed, including negative results. A favourable mean with a paired interval containing zero is inconclusive for this study. Changes in accepted accuracy must be interpreted together with coverage, and the validation target can remain unmet even when the accepted test subset appears accurate. Three repeated predictions per trial do not constitute three independent observations.
