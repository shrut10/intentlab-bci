# Masked-channel pretraining on public scalp EEG

This is a deliberately small test of whether learning from unlabelled EEG helps a classifier when fewer training labels are available. It uses IntentLab's existing PhysioNet data and evaluates only the already inspected validation participants; it does not use Neuralink data or assess a new independent test cohort.

The motivation is the broad pretraining idea discussed in [Neuralink's update on unlabelled brain recordings](https://neuralink.com/updates/pretraining-on-50000-hours/). The public data used here are [PhysioNet EEGMMIDB v1.0.0](https://physionet.org/content/eegmmidb/1.0.0/), distributed under ODC-BY 1.0. Masked-channel reconstruction on scalp EEG is an independent feasibility design, not a reproduction of the implanted-recording study.

The [protocol](PROTOCOL.md) and [configuration](protocol.json) were saved and fingerprinted before training. See the [archived results](../../artifacts/pretraining-v1/evaluation.json), [trial membership](../../artifacts/pretraining-v1/split_manifest.json) and [validation predictions](../../artifacts/pretraining-v1/predictions.csv).

## What was compared

A small, 2,849-parameter convolutional classifier was trained in two ways. The scratch arm learns only from labelled source trials. The pretrained arm first learns to reconstruct two hidden channels from the seven remaining channels, using all source trials without their labels, then learns the same classification task. The reconstruction head is discarded before classification. Both arms have identical initial classifier weights, labelled examples, supervised batch order and supervised training duration; pretraining adds computation and is not a compute-matched control.

All 2,832 source epochs from 63 people are available for pretraining, representing 2.36 hours of retained three-second epochs. The nominal 10% per-participant/class budget yields 252 labelled epochs, or 8.9% overall after rounding down within each cell; the full budget uses all 2,832. Selection is independently seeded for 2027, 2028 and 2029 and matched between the two arms. These are training-label budgets, not personal calibration trials from new people.

Both stages run for a fixed 20 epochs, with no validation-based checkpoint selection, early stopping, temperature fitting or confidence threshold tuning. The evaluation uses 989 trials from the established 22 validation participants. The original 21 test participants are not scored, and the deployed model is not replaced.

Supervised updates are matched between the two arms within a budget. Across budgets, 20 epochs means 80 updates for 252 labelled trials versus 900 updates for 2,832, so the difference between budgets is not a compute-controlled label-efficiency curve. Pretraining contributes another 900 reconstruction updates per seed.

## Reproduce

From the repository root, use Python 3.12 and the project's training environment:

```sh
python -m pip install -r requirements-training.txt
python scripts/prepare_data.py
python scripts/run_pretraining.py
```

The runner refuses to overwrite an existing output directory. For a clean reproduction, keep the committed outputs separately before running, or use a fresh checkout and move `artifacts/pretraining-v1` outside the checkout. The raw EDFs and full epoch archive are not committed; `prepare_data.py` obtains the published recordings and verifies checksums. On a machine where the recordings are already cached, use `python scripts/prepare_data.py --offline` instead. Do not change the fixed protocol to chase a better validation result.

Lightweight data/report tests run in the normal development environment:

```sh
python -m pytest -q tests/test_pretraining.py
```

Two additional tests exercise the real PyTorch pretraining loss and deterministic training in the training environment:

```sh
python -m pytest -q experiments/pretraining-v1/test_training.py
```

The regular tests reconstruct every metric and paired comparison from the committed CSV, check exact source/validation trial membership, verify code/data/protocol fingerprints and confirm that original serving artifacts are unchanged. The training runner writes progress to `artifacts/pretraining-v1/progress.json` during a run, replacing it with the final report on completion.

## Interpret the evidence

The completed three-seed study did not show an improvement from this pretraining configuration:

| Labelled source trials per seed | Scratch mean balanced accuracy | Pretrained mean balanced accuracy | Paired difference, percentage points (95% participant-bootstrap interval) |
| --- | ---: | ---: | ---: |
| 252 (nominal 10% per person/class; 8.9% overall) | 53.73% | 52.51% | -1.22 (-3.01 to +0.36) |
| 2,832 (all source trials) | 56.18% | 55.43% | -0.76 (-1.44 to -0.12) |

The small-budget interval includes zero; the full-budget interval lies below zero for this fixed, exploratory setup. Neither result establishes the effect of pretraining in other architectures, datasets or recording modalities. The original deployed decoder remains unchanged. All 12 classifier runs completed in approximately 20.4 CPU minutes on the development machine, including pretraining; exact environment and timings are in the archived report.

The primary outcome is mean participant balanced accuracy. Each participant receives equal weight; the two arms are paired on the same people. Seed-specific differences are averaged within a person before a participant bootstrap. The interval therefore describes variation across these participants conditional on the fixed methods and seeds, rather than treating repeated predictions as extra independent observations. All seeds and budgets remain visible regardless of the result.

Pretraining on 2.36 hours of scalp-EEG epochs cannot replicate a large implanted-recording study. These epochs were originally collected with labels, reconstruction may exploit ordinary channel correlations, the validation cohort is already familiar to the project, and the offline filter is non-causal. An improvement here would motivate a new preregistered evaluation on independent data; a null or negative result is still useful evidence about this small configuration. This study does not establish better real-time control or calibrated command confidence.
