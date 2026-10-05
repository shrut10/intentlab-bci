# IntentLab · experiment protocol v1

Written before training or examining test performance, 5 October 2026.

## Question

Can a model trained on some people distinguish cued left- and right-fist motor imagery in previously unseen people? How does declining uncertain predictions change the error rate and the number of commands available?

This is an offline research experiment. It does not read arbitrary thoughts, decode dreams, establish consciousness, or control a physical device. The interactive cursor replays recorded trials. Visual cues and other artefacts may contribute to decoding; the experiment cannot prove a purely motor-imagery mechanism.

## Data and fixed preprocessing

- PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0, DOI [10.13026/C28G6P](https://doi.org/10.13026/C28G6P), all 109 available participant IDs, runs **04, 08, 12** only. These are imagined unilateral fist movements. T1 = left; T2 = right; rest excluded.
- Verify downloads against the publisher's SHA256SUMS file. Do not publish raw EDFs in Git. Preserve a provenance manifest.
- Common average reference using all 64 original EEG channels; retain FC3, FCz, FC4, C3, Cz, C4, CP3, CPz, CP4. Nine prespecified sensorimotor channels keep the application small and interpretable.
- Extract exactly 1–4 seconds after each cue (480 samples at 160 Hz). Remove each epoch's channel means, then apply a fourth-order 8–30 Hz Butterworth bandpass independently to each epoch with zero-phase filtering. This uses the entire completed epoch and is **not streaming/causal**. Never filter across train/test boundaries.
- Exclude recordings with sampling frequency other than 160 Hz, missing required channels, or unparseable files. Exclude epochs without a full 4-second labelled interval, nonfinite values, flat selected channels (standard deviation < 0.01 microvolt), or peak-to-peak amplitude > 1000 microvolts before filtering. Report every exclusion. No subject exclusion based on model performance.
- Scale each filtered epoch by a single RMS across channels and time; this preserves relative channel power. Do not fit preprocessing on the test set.

## Splits and models

Shuffle the integer IDs 1…109 with NumPy seed 2027 **before exclusions**. First 65 train, next 22 validation, remaining 22 test. Every trial from a person stays in one partition. Fix the split in the manifest. No test-driven selection or retraining after seeing results.

Compare a constant majority baseline, log-bandpower logistic regression, four-component CSP with shrinkage LDA, and a compact temporal/depthwise convolutional network inspired by EEGNet (an adaptation, not an exact reproduction). Logistic regression C=1.0. CNN: 8 temporal filters, spatial depth multiplier 2, dropout 0.35, AdamW learning rate 0.001, weight decay 0.01, batch size 64, at most 40 epochs, patience 8. Use a single seed (2027); report that this is not a multi-seed benchmark.

Select the CNN checkpoint and the serving model by **validation participant-macro balanced accuracy**, with simpler model winning exact ties. Keep all models' test results visible. Fit one temperature per candidate on validation logits, bound 0.5–5.0. Reusing validation for selection and calibration can bias validation estimates; only the untouched test provides the final estimate.

Select an abstention threshold from 0.50, 0.55, …, 0.95 using validation only: choose the lowest threshold achieving at least 75% retained-trial accuracy at at least 20% coverage and at least 100 retained trials. If none qualifies, use 0.75 and explicitly state that the target was not met. A confidence score is a model estimate, not a guarantee. Changing the UI threshold is exploratory.

## Evaluation and explanations

Report participant-macro balanced accuracy (primary), pooled balanced accuracy, accuracy, ROC AUC, Brier score, a ten-bin reliability curve, confusion matrix, and per-person results. Bootstrap held-out participants 2,000 times for the primary metric's 95% interval. A 50% balanced-accuracy baseline applies to this binary task; do not compare results with papers using other splits/tasks.

Report fixed-threshold accuracy and coverage, plus a risk/coverage curve. Stress-test the frozen models with reproducible added Gaussian noise (0.5 and 1.0 times epoch RMS) and zeroed C3/C4 channels; these synthetic disturbances are not a clinical robustness validation. Do not tune on these test results.

Explain the selected model with single-channel occlusion: recompute predictions after zeroing each channel, without re-normalisation. Show the change in right-imagery probability; this measures model sensitivity, not causal brain importance. Report dataset-level changes in accuracy as a separate analysis.

## Delivery

Publish reproducible acquisition/training scripts, Parquet trial metadata and bandpower features, actual model artifacts, a deterministic sample of the first six valid test trials per participant, and versioned metrics. Serve live inference through FastAPI and a browser-based signal viewer/control replay. Docker, automated tests, CI, health checks, source attribution, a model card and an interview walkthrough are part of the deliverable. No paid language-model dependency is necessary.
