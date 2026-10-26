# Masked-channel pretraining feasibility protocol

Written before running this experiment on 26 October 2026. This is a small, exploratory development study on an already inspected PhysioNet validation cohort, not a new independent test, Neuralink replication or foundation-model benchmark.

## Question and boundaries

Does learning to reconstruct hidden EEG channels from source-participant recordings help a small classifier when fewer source labels are used? The motivation is the general idea of learning representations from unlabelled recordings; scalp EEG and implanted neural recordings are different measurements. No Neuralink recordings, weights or private data are used.

Use the existing checksum-verified PhysioNet EEGMMIDB derivatives, with all established preprocessing and participant splits unchanged. Pretraining sees only 2,832 three-second epochs from the 63 source training participants: 2.36 hours of retained epochs. This is not 2.36 hours of additional recordings, and the labels exist but are ignored during reconstruction. Validation uses the established 22 people and 989 trials, whose results were previously inspected. The 21-person test cohort is not scored in this experiment. Original model artifacts and published results remain unchanged.

## Fixed design

- Seeds 2027, 2028 and 2029; two arms: random initialisation and masked pretraining
- A matched encoder: Conv1d(9,16,kernel=7,padding=3), ELU, Conv1d(16,16,kernel=7,padding=3), ELU; no temporal downsampling or batch-normalisation statistics
- During pretraining only, a 1x1 convolution reconstructs nine channels. Exactly two whole channels per epoch are zeroed at random; mean-squared reconstruction error is computed on hidden channels only
- Pretrain for 20 fixed epochs on all source epochs, without labels, validation-based selection or early stopping
- Classify by averaging encoder activations over time and applying one linear binary-logit head, for 20 fixed supervised epochs
- Both arms begin with identical encoder and classifier-head weights for each seed; only the pretrained encoder receives reconstruction updates. Both arms use the same labelled subset, batch ordering, supervised optimiser and number of supervised updates
- Label budgets: 10% and 100% per source participant/class cell, with floor(n*fraction), at least one example per cell. Publish actual counts, since rounding changes the realised fraction. These are source training-label budgets, not new-person calibration budgets
- Batch size64, AdamW learning rate0.001, weight decay0.01, four CPU threads; binary cross-entropy for supervised learning
- Use final-epoch weights only. No tuning, temperature scaling, threshold fitting, selection of successful seeds or production-model promotion

## Reporting

Publish every seed and budget, training histories, exact selected trial IDs, input/code/protocol hashes, validation probabilities and participant-level balanced accuracy and Brier scores. The primary comparison is pretrained minus scratch balanced accuracy. First average seed-specific differences within each participant, then bootstrap the 22 paired participant differences 2,000 times with seed5027. This conditional interval measures participant-sampling variation; it does not treat repeated seeds as independent people or capture all optimisation/protocol uncertainty.

Pretraining adds compute, so this is not a compute-matched comparison. A null or negative result is publishable and does not justify increasing the epoch budget after seeing outcomes. Reconstruction may mostly recover correlated scalp signals and need not learn intention-specific features. Offline zero-phase preprocessing remains non-causal. Confidence values are uncalibrated, so this study makes no reliable-control or improved-abstention claim.
