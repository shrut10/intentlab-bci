# Ten early calibration trials, followed by later-run evaluation

This runnable example shows how to express an **absolute chronological calibration budget** through MOABB's `CrossSubjectEvaluation(splitter=...)` interface. It was prepared following the discussion in [MOABB issue #1077](https://github.com/NeuroTechX/moabb/issues/1077#issuecomment-6127663242).

For each held-out participant:

1. Train a classifier using the other participants
2. Use the first ten eligible epochs from the held-out participant's first run for adaptation
3. Exclude the remaining epochs from that first run
4. Score only the second and third runs, without updating the fitted model

There are **three generated participants**, each with 14, 12 and 12 trials across three runs. Each fold therefore contains **76 source training trials, 10 target calibration trials, 4 discarded trials and 24 target test trials**. These counts are checked by the tests, rather than selected according to class labels.

## Run it

From the IntentLab repository root, using Python 3.12:

```sh
python3.12 -m venv /tmp/intentlab-moabb-venv
source /tmp/intentlab-moabb-venv/bin/activate
python -m pip install -r examples/moabb_chronological/requirements.txt
python -m pytest -q examples/moabb_chronological/test_example.py
python examples/moabb_chronological/example.py
```

This uses MOABB's `FakeDataset`; no EEG recordings, credentials or model weights are downloaded. Dependencies are installed on the first run. The example pins [MOABB commit ed1aacd](https://github.com/NeuroTechX/moabb/commit/ed1aacdb39a445efe13a4f026673b14cc2913b85), which includes the top-level splitter interface; an older installed release may not support it. Use this separate environment rather than adding MOABB to IntentLab's production dependencies.

The command prints six result rows: three held-out participants in each of two modes. Temporary results are removed on exit. To retain the split manifest, CSV results and MOABB cache, run from the repository root:

```sh
python -c 'from examples.moabb_chronological.example import run_demo; run_demo("/tmp/intentlab-moabb-results")'
```

## What the two modes demonstrate

| Mode | Target recordings supplied to `fit` | Target labels supplied to `fit` | Adaptation |
|---|---|---|---|
| Alignment only | The ten early epochs | None | Estimate and freeze a target alignment reference |
| Alignment and temperature | The same ten early epochs | Their ten labels | Freeze the alignment reference, then fit a confidence temperature |

In the combined mode, there are **ten calibration trials in total**, not ten labelled trials plus another ten unlabelled trials. Alignment uses only their signals; temperature scaling uses their labels. No later-run samples or labels enter `fit`.

The small teaching classifier performs regularised Euclidean alignment, extracts log mean-square amplitude per channel, and trains logistic regression. Each source participant is aligned using that participant's source training epochs. Target alignment is estimated exclusively from the ten calibration epochs. A positive temperature in [0.5, 5] minimises binary cross-entropy on those calibration labels. If the slice contains only one class, the temperature remains 1 and the fitted estimator records that fallback.

MOABB's default `LeftRightImagery` score is **ROC AUC**. Positive temperature scaling preserves the ordering of binary predictions, so the two modes should have the same ROC AUC here. These scores do not evaluate calibration quality or an abstention policy, and generated noise cannot demonstrate useful EEG decoding.

## Interface details

- `EarlyRunCalibration` is a `BaseCrossValidator`, with MOABB's metadata-aware `split(y, metadata)` signature
- It yields three arrays of **integer row positions**, not DataFrame index labels: `(train, calibration, test)`
- Run order is explicit: `("0", "1", "2")` for this generated dataset, with rows already in acquisition order within each run
- `get_metadata()` includes `calibration_size`, which enables target-data routing in this MOABB revision, and `calibration_labeled`, which selects the labelled or unlabelled route
- The reported `calibration_size` is the realised fraction of all target epochs (10/38); it does not select the slice. The splitter always selects the absolute count of ten
- The estimator requests `subjects`, `X_target_unlabeled`, `X_target_labeled` and `y_target_labeled` through scikit-learn metadata routing. Only the route selected by the split is supplied
- Alignment and feature extraction are inside the estimator, so training and raw calibration epochs follow the intended transformations
- Prediction uses the frozen reference and temperature. Batch prediction and one-epoch-at-a-time prediction give the same outputs

## Tests and boundaries

The tests check exact trial counts, held-out participant separation, omitted early trials, positional indices with duplicate DataFrame index labels, independence from label values, explicit run order, and errors for insufficient calibration data or ambiguous metadata. They also perturb later and discarded **preprocessed epochs**, check that fitted parameters stay unchanged, and check that calibration labels cannot change alignment or the source classifier.

The integration test calls the actual `CrossSubjectEvaluation.process(...)` in both modes, intercepts every estimator fit, and compares the exact source and target arrays and labels against the expected split. A separate [GitHub Actions workflow](../../.github/workflows/moabb-example.yml) runs these tests and the example without changing the deployed decoder.

This is a **synthetic interface example, not a port of IntentLab's CNN experiment or a real-data MOABB benchmark**. The source alignment policy and classifier are deliberately simpler than [the original adaptation study](../../experiments/adaptation-v1/README.md). It adds no new evidence that alignment or temperature scaling improves decoding or reliability.

Chronology here means first eligible epochs in metadata row order within explicitly ordered runs. The example rejects multiple sessions, missing runs and too few early epochs rather than silently changing the budget. For a real dataset, verify the session/run mapping and within-run acquisition order after preprocessing. MOABB's standard offline filtering is not a causal streaming pipeline; the tests establish isolation at the preprocessed-epoch/model boundary, not sample-level causality during signal filtering. Ten trials also provide a very small temperature-fitting sample and do not guarantee well-calibrated confidence on future runs.
