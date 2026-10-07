# Adaptation follow-up to issue #1

Protocol specified on 7 October 2026 before fitting the follow-up models. This is an exploratory reanalysis of an already inspected benchmark, not an externally preregistered study or a new independent test. The v1.0.0 production model and its original evaluation remain frozen.

## Question and scope

Does a short participant-specific alignment step, confidence calibration, or their combination improve later-run motor-imagery decoding relative to a matched unaligned CNN? This implements the comparison proposed in [issue #1](https://github.com/shrut10/intentlab-bci/issues/1), without assuming that it will improve reliability. The scope is Euclidean alignment, not Riemannian alignment or hardware validation.

## Data and chronology

Reuse the checksum-verified, preprocessed nine-channel epochs and the original participant-disjoint train/validation/test partition. Keep all original quality exclusions. Within every participant, identify the first **ten eligible run-04 trials**, ordered by cue onset and event, without selecting by class, confidence or correctness. Fit participant-specific quantities only to those initial trials. Evaluate validation/test participants on **all eligible trials in runs 08 and 12**. Unused run-04 trials are never used for validation/test adaptation or scoring. Later runs are chronological blocks within this dataset, not separate-day sessions or prospective live use.

Assert that each participant has the required calibration budget and later-run examples of both classes. Fail rather than silently changing the population if this is not true. The original 63 training participants may contribute all their training trials (including run 04); their class labels are available during supervised training. Validation/test participants remain absent from gradient training.

The baseline uses zero target-person recordings or labels. Alignment uses ten unlabelled recordings; personal temperature scaling uses ten labelled recordings; their combination uses the same ten labelled recordings. All conditions are evaluated on exactly the same later trials. Personalisation changes the problem from zero-calibration transfer to calibration-assisted transfer and must be reported as such.

## Fixed comparison

1. **Baseline:** newly trained unaligned CNN, global validation temperature.
2. **Personal temperature:** same unaligned checkpoint, absolute temperature fitted to the ten initial labelled trials of each target participant.
3. **Euclidean alignment:** CNN trained on consistently aligned source data, with target-person alignment fitted to ten initial unlabelled trials and one global validation temperature.
4. **Alignment + personal temperature:** same aligned checkpoint and reference matrix, with personal temperature fitted on the same ten initial labelled trials.

For alignment, compute the mean uncentred channel second-moment matrix `R = mean(X Xᵀ / samples)` over the ten preprocessed initial trials. Use fixed shrinkage `0.99 R + 0.01 trace(R)/channels I` and a relative eigenvalue floor of 1e-6, then left-multiply each epoch by the symmetric inverse square root. Do not fit to later target trials, refit per trial, or renormalise after alignment. This regularised adaptation of He and Wu’s method does not reproduce all experiments in their paper.

Both CNNs use the existing CompactEEG architecture and original training hyperparameters, on the same training rows. Fit paired unaligned/aligned models at seeds **2027, 2028 and 2029**, restarting the same seed and minibatch order for each pair. Select each checkpoint on participant-macro balanced accuracy from validation participants’ later runs only. Train at most 40 epochs with patience eight; do not select a seed, tune alignment or repeat training based on test scores.

Fit global temperatures on validation later-run logits. Fit personal temperatures using only each target participant’s first ten labelled trials; positive bounded temperature fitting minimises binary log loss in [0.5, 5.0]. If the initial block contains only one class, retain the global temperature and record that fallback. Report how often personal estimates hit the bounds; small calibration sets may overfit. Temperature scaling must not change the binary class decision or within-participant confidence ordering apart from numerical ties.

For each condition/seed, choose a global rejection threshold from 0.50, 0.55, …, 0.95 on validation later-run predictions only. Use the original requirements: retained-trial accuracy ≥75%, coverage ≥20% and at least 100 retained trials, taking the lowest qualifying threshold. If none qualifies, retain 0.75 and label the target unmet. Also report every condition at the fixed 0.75 threshold so probability rescaling is visible. Test thresholds are never optimised on test outcomes.

## Evaluation and uncertainty

Report each seed separately, and the arithmetic mean across seeds of participant-macro balanced accuracy, Brier score and ten-bin ECE. Report fixed-0.75 and validation-rule coverage/accuracy/counts per seed, with pooled accepted accuracy across repeated-seed predictions explicitly labelled as such. Repeated predictions are not additional independent trials.

Estimate paired changes against baseline by first averaging each participant’s metric over the three seeds, then bootstrapping participants 2,000 times with a fixed seed. This interval is conditional on this split and these fitted models; it does not cover population shifts or full training uncertainty. Publish all seeds, per-participant scores, probabilities and adaptation metadata, including validation threshold selection. Do not infer an improvement merely from a higher mean when uncertainty includes no change.

Retain the original C3/C4 sensitivity result as context; this follow-up does not establish a minimal montage. A smaller live headset would need preprocessing that does not rely on the original 64-channel common-average reference, causal filtering, a rest/no-intention condition and prospective evaluation.

## Reproducibility and decision

Save the protocol/config hashes, input hashes, code revision, environment, six ONNX exports, reference matrices, temperatures, validation histories and trial-level predictions. Export parity must be below 1e-4. Automated tests cover isolation of calibration/evaluation rows, stable alignment, temperature invariance, threshold selection and reconstruction of published metrics. Protect completed outputs from overwriting.

Publish a separate results page and link it from the existing research report and issue. Do not silently replace the deployed baseline, change its reliability claim, or claim a successful hardware fix. A negative or inconclusive comparison still answers the issue and should be reported without tuning further on these outcomes.

## References

- He, H. and Wu, D. (2020). Transfer Learning for Brain–Computer Interfaces: A Euclidean Space Data Alignment Approach. *IEEE Transactions on Biomedical Engineering*, 67(2), 399–410. https://doi.org/10.1109/TBME.2019.2913914 · https://arxiv.org/abs/1808.05464
- Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *PMLR*, 70, 1321–1330. https://proceedings.mlr.press/v70/guo17a.html
- Dataset attribution and processing: [NOTICE.md](../../NOTICE.md) and the [original protocol](../../docs/PROTOCOL.md).
