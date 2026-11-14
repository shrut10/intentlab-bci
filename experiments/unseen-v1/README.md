# Testing new participants

This study asks whether a fixed method can distinguish cued left and right hand imagery in people who have contributed no recordings to fitting or selection

It uses the public Dreyer2023 dataset with all 87 participants assigned in advance: 45 for training, 15 for choosing the model and confidence threshold, and 27 for the final test; the original deployed PhysioNet model remains unchanged

## What the comparison means

The bandpower model describes how much activity each channel contains in three frequency bands; the covariance model also describes how the channels vary together, then represents those relationships relative to a reference learned from training participants

Both descriptions feed a fixed logistic regression, so the comparison tests two established signal representations rather than giving one model a larger tuning budget; the selected model is chosen using validation people before the final people are read

Training gives each person and each hand class equal total weight, and those same weights fit the covariance reference, feature scaling and logistic regression; the confidence threshold is chosen separately for each method using its own validation predictions

Balanced accuracy gives equal importance to left and right imagery, then the report averages that score across people so a participant with more trials cannot dominate the result; a score above 50% indicates better discrimination than a constant class guess in this two class task

The confidence threshold asks a separate question: how many predictions would the model retain, and how often would those retained predictions be correct; people receiving no predictions remain in the coverage calculation, and their accepted accuracy is left undefined

## Inspect the evidence

* [Declared protocol](PROTOCOL.md) and [machine readable settings](protocol.json)
* [Exact participant membership](split_manifest.json)
* [Pinned public sources and publisher checksums](source_manifest.json)
* [Saved validation selection, predictions and final evaluation](../../artifacts/unseen-v1)
* [Public research report](https://intentlab-bci.vercel.app/research/unseen)

## Reproduce the study

Use Python 3.12 and install the separate research requirements from the repository root; for an exact reproduction, use a separate checkout of the [executed methods revision](https://github.com/shrut10/intentlab-bci/tree/dbdb045e860c22827f436e24241030d8bf2c5387), then verify its source hashes against the published provenance

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-unseen-study.txt
.venv/bin/python -m pytest -q tests/test_dreyer_source.py tests/test_unseen.py tests/test_unseen_study.py
```

The first stage reads only the training and validation people, downloads verified source archives when requested, computes deterministic features and fits the two fixed models

```bash
.venv/bin/python scripts/run_unseen_study.py --stage fit \
  --protocol-commit 72752d9ff1242060bce8b5afcc76284b03c8f05b \
  --output artifacts/unseen-reproduction --cache data/derived/unseen-v1/reproduction \
  --download
```

Commit the resulting `artifacts/unseen-reproduction/selection.json` and `artifacts/unseen-reproduction/validation_predictions.csv` in that checkout before proceeding, then replace `YOUR_SELECTION_COMMIT` below with the resulting commit identifier; this must be the commit from your own reproduction, because the selection records the models and predictions produced by your first stage

```bash
.venv/bin/python scripts/run_unseen_study.py --stage test \
  --protocol-commit 72752d9ff1242060bce8b5afcc76284b03c8f05b \
  --selection-commit YOUR_SELECTION_COMMIT \
  --output artifacts/unseen-reproduction --cache data/derived/unseen-v1/reproduction \
  --download
```

The script checks the committed protocol, selection, executed source, package versions and model hashes before reading a final participant, then checks the selection again before producing final predictions; keep those source files and dependencies unchanged throughout both stages

The archived study directory is protected against overwriting, so use a separate output directory inside your checkout for a reproduction and keep its saved models in the ignored cache directory; `provenance.json` records the exact executed source hashes and package versions

All 87 compressed archives require about 5.46 GB, while each participant's two acquisition recordings are extracted temporarily and removed after feature calculation; the original recordings and locally fitted models are excluded from Git and deployment

The saved CSV probabilities are sufficient to recompute the reported metrics without downloading EEG or fitting another model; automated evidence checks exercise that independent route

## Scope and attribution

The source is the EDF conversion distributed through MOABB's OSF storage, mirrored byte for byte on Hugging Face; it is not the original GDF archive on Zenodo, and the source manifest preserves both the conversion identity and the original deposit attribution

Dreyer, Roc, Pillette, Rimbert and Lotte published the recordings under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with [dataset DOI 10.5281/zenodo.8089820](https://doi.org/10.5281/zenodo.8089820) and [paper DOI 10.1038/s41597-023-02445-z](https://doi.org/10.1038/s41597-023-02445-z); the public derived outputs contain prediction probabilities, trial identities, provenance and aggregated metrics, while transformed features and fitted models remain in the local ignored cache

The first two runs contain sham visual feedback and the trial labels can be associated with visual, eye or muscle activity; a delayed window does not prove those effects are absent, and this study cannot establish spontaneous intention decoding, clinical usefulness or reliable live control

These are separately fitted models in a new recording corpus, rather than an external validation of the deployed PhysioNet CNN
