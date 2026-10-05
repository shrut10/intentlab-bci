# Model card · IntentLab 1.0.0

## Intended use

Offline exploration of cued left- versus right-fist motor imagery in the PhysioNet EEGMMIDB dataset. Portfolio research demonstration of data provenance, subject-level evaluation, uncertainty, explanations and deployment. No clinical, diagnostic, identity, consciousness, employment or safety-critical use. No live hardware control.

## Data and input

Nine sensorimotor electrodes, 480 samples at 160 Hz, from 1–4 seconds after an imagery cue. Common average reference over the original 64 channels; 8–30 Hz fourth-order Butterworth zero-phase filtering within each completed epoch; one RMS normalisation value per epoch. No participant identifiers, labels or trial indices are model inputs.

The original source has 109 participant IDs. Three participants' selected recordings used 128 Hz and were excluded by the fixed 160 Hz rule. One high-amplitude epoch and one incomplete interval were excluded. The result is 4,766 epochs from 106 participants: 2,832 train / 989 validation / 945 test. See the data card and manifest.

## Models and selection

The selected model is a 1,289-parameter EEGNet-inspired CNN with temporal, depthwise spatial and separable temporal convolutions, batch normalisation, ELU activations, pooling, dropout and one binary logit. It uses adaptive pooling and different kernels from the EEGNet paper. The serving artifact is ONNX. Training used CPU PyTorch, seed 2027; early stopping selected epoch 9 from 17 completed epochs. Selection metric: validation participant-macro balanced accuracy.

Baselines are training-majority prevalence, 27 log-bandpower features with logistic regression, and four train-fitted CSP components with shrinkage LDA. CSP uses trace-normalised trial covariances, 10% covariance shrinkage, and two extreme filters from each end of the generalised eigenspectrum. The LDA uses scikit-learn's automatic shrinkage; all feature scaling is fit on training only.

Validation-fitted temperature scaling: CNN temperature approximately 2.883, bounded to 0.5–5.0. Validation is reused for model selection, calibration and threshold choice, so its estimates are optimistic. Test participants remain untouched until those choices are frozen.

## Outcomes

- Selected CNN: **61.073%** test participant-macro balanced accuracy; bootstrap 95% interval **55.849–66.645%**, resampling the 21 test people 2,000 times.
- CSP + LDA: 61.5%; bandpower logistic regression: 58.7%; majority: 50%. Exact unrounded values are in `artifacts/evaluation.json`.
- The validation criterion of 75% retained accuracy with at least 20% coverage and 100 retained examples was **not achieved**. The protocol's fallback threshold of 0.75 accepted seven validation epochs (57.1% accuracy), and 27 test epochs (88.9% accuracy; 2.86% test coverage). The sharp difference and tiny samples make this unsuitable as an operational reliability claim.
- Zeroing C3 or C4 lowers test participant-macro balanced accuracy to approximately 52.8% and 52.9%. Noise equal to one epoch RMS lowers it to approximately 59.5%.

The test result is shown for all candidates, but test performance does not alter the serving-model choice. Threshold changes in the public demo and the risk/coverage curve are exploratory.

## Explanations and limitations

Single-channel occlusion recomputes the selected model after zeroing one processed channel, without re-normalising. Per-trial bars show change in right-class probability; aggregate evaluation reports change in balanced accuracy. Channels are correlated, and setting a channel to zero is an artificial intervention. These values are model sensitivity, not causal importance or a map of intentions.

The dataset is cue-driven. Cue-evoked responses and artefacts may be predictive even after skipping the first second and selecting motor-region electrodes. No independent EMG/eye-movement control was implemented. The quality checks are simple and do not guarantee clean recordings. Demographic representativeness was not assessed; dataset subject IDs are not diagnoses.

Zero-phase filtering uses future samples within the completed epoch. The system requires a three-second window and does not provide causal, sample-by-sample online predictions. API latency excludes recording acquisition, preprocessing and network travel; the displayed timing includes inference, explanation variants and spectrum computation.

Only one dataset, one subject split and one CNN training seed were evaluated. Per-person performance varies substantially. There is no external-dataset test, personal calibration, drift monitor, headset integration or treatment claim. Future experiments must use a new untouched holdout or a separately designed nested evaluation.

## Reproducibility and serving

Acquisition validates publisher checksums. Source manifests, frozen reports and split membership are versioned. Artifacts are loaded without pickle and verified at app startup. PyTorch-to-ONNX numerical parity was checked on validation examples before test evaluation; the maximum logit error is recorded in the report. Demo epochs are the first six eligible test trials per participant, selected without looking at correctness.

See `docs/PROTOCOL.md`, `scripts/prepare_data.py`, `scripts/train.py`, `artifacts/checksums.json`, and `docs/OPERATIONS.md`.
