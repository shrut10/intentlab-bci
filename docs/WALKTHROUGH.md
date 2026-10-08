# Understand and explain IntentLab

## The project in thirty seconds

IntentLab tests whether machine learning can distinguish imagined left- and right-fist movements from recorded EEG. It uses public data, keeps entire people out of training, compares a compact neural network with simpler models, and deploys actual inference in a browser experiment. Users can inspect signals, disconnect an electrode, add noise and change when the system abstains.

The main result is 61.1% balanced accuracy across unseen participants, with substantial variation between people. The predefined reliability target was not reached. The project demonstrates both a working deployment and how to measure the gap between a promising signal and a dependable interface.

## 1. What the data actually means

An EEG electrode measures voltage differences over time. It does not record a labelled stream of thoughts. In this dataset, people received visual cues and imagined moving a particular fist. The annotation says what the experiment asked them to imagine.

An original recording contains 64 channels at 160 samples per second. This project first uses all 64 to calculate a common average reference, then keeps nine prespecified sensorimotor channels. It takes the interval from one to four seconds after a cue: 3 seconds × 160 samples = **480 values per channel**. One model input therefore contains **9 × 480 = 4,320 numbers**.

Read `scripts/prepare_data.py`. Find the run numbers and T1/T2 mapping. Runs 4, 8 and 12 are the imagined left/right task. Other runs contain different actions; reusing their annotation codes would silently mislabel the data.

**Try:** download one epoch from the app. Its first column is time after the cue. Its remaining columns are the nine processed signals. These are normalised amplitudes, not raw microvolt readings.

## 2. Why preprocess?

EEG contains activity outside the frequency range of interest, reference effects, changes in amplitude and artefacts. The pipeline:

1. Checks the publisher's SHA-256 checksum so a corrupted download cannot silently become training data.
2. Uses a common average reference and selects the fixed channel list.
3. Applies eligibility checks without looking at classifier performance.
4. Extracts the completed three-second interval and removes each channel's mean.
5. Filters to 8–30 Hz, covering the mu/alpha and beta ranges used in this experiment.
6. Divides the whole epoch by one RMS value, retaining relative amplitudes between channels.

Read `src/intentlab/signal.py`. Notice that there is no learned preprocessing fitted on all participants. The filter is applied independently to each completed epoch. Zero-phase filtering sees later samples in that epoch, so this pipeline cannot be described as sample-by-sample live decoding.

**Try:** run `pytest tests/test_signal.py -q`. The tests include known 10 Hz and 25 Hz signals to verify that the bandpower features distinguish alpha and beta energy.

## 3. The most important modelling choice: split by person

Two trials from the same person share anatomy, electrode placement, recording conditions and other characteristics. Randomly scattering their trials across train and test would answer an easier question than generalising to a new person.

IntentLab assigns all source participant IDs to fixed train, validation and test groups before exclusions. All of a person's trials stay together. The remaining counts are **63 / 22 / 21 people**.

Training data fits model weights and spatial filters. Validation chooses the CNN checkpoint, the serving model, temperature and abstention threshold. Test data supplies the final result after those choices are frozen.

**Interview question:** “Why not pick CSP after seeing that it beats your neural network on the test set?”

Because that would use the test as another selection set. The CNN was chosen on validation, so the public service keeps that choice. Both models' test results stay visible. A future comparison needs another untouched evaluation or a separately designed nested experiment.

## 4. What each model learns

**Majority baseline.** Always predicts the training set's more common class. Its 50% balanced accuracy is the reference for a binary classifier that effectively ignores the signal.

**Bandpower logistic regression.** Welch's method estimates frequency power. Three frequency bands per electrode give 27 log-power features. A linear classifier combines those features. This is compact and easy to inspect, but it may miss useful relationships between channels.

**CSP + LDA.** Common spatial patterns learn weighted combinations of electrodes whose variances distinguish the classes. Four projected signals become log-power features. A shrinkage linear discriminant classifier then makes a prediction. The spatial filters are learned exclusively from training people.

**Compact EEG CNN.** Temporal filters learn time patterns. Depthwise spatial filters combine electrodes within each temporal feature. A second separable block and pooling produce a small representation for a binary output. There are only 1,289 learned parameters. Read `CompactEEG` in `scripts/train.py`; it is short enough to draw by hand. It is inspired by EEGNet, with explicit architectural differences.

**Try:** switch between all four decoders on the same recording. Explain why a more complex model need not make the best prediction on every trial.

## 5. Accuracy, uncertainty and abstention

For each participant, balanced accuracy is the average of recall for left and recall for right. Participant-macro balanced accuracy then averages those scores across people. This keeps a person with more available trials from dominating the headline measure.

The CNN scored **61.1%**, with a **55.8–66.6%** interval from resampling held-out people. The interval quantifies uncertainty across this test cohort. It is not a guarantee of performance on another dataset or headset.

A sigmoid converts the model logit to a right-class probability. Temperature scaling, fitted on validation, makes those probabilities less overconfident. Confidence is the larger of the left and right probabilities. If it is below the threshold, the cursor does not move.

The desired validation operating point was at least 75% retained accuracy while retaining at least 20% of trials and 100 examples. **No threshold met it.** The fixed fallback at 75% confidence accepts only 2.9% of test trials. Its 88.9% accuracy is 24 correct predictions out of 27 retained trials; it is not an overall score to put on a CV.

**Try:** use a 50% threshold. Every binary prediction becomes a command. Raise it towards 75%. More steps should abstain. Explain why a system that never acts could avoid mistakes yet still be useless.

## 6. Explanations and robustness

The app zeros one processed channel at a time, reruns inference, and shows how the right-class probability changes. A large bar means the model's answer depends on that channel under this artificial intervention. It does not prove that the channel contains an isolated intention or that the electrode is causally necessary.

Across the test set, zeroing C3 or C4 lowers the score to around 53%. This is an interesting failure mode to investigate. The synthetic noise experiment asks a different question: whether adding a controlled amount of random disturbance changes performance. Neither replaces testing a real headset with realistic faults.

**Try:** choose one recording, disconnect C3 and then C4, and compare the probabilities with the clean signal. Use the model-sensitivity chart to explain the changes.

## 7. What makes this a deployed ML project

Training and serving have different environments. The production app does not load PyTorch or MNE. The CNN is exported to ONNX; the linear models use plain JSON coefficients. The server checks artifact hashes before serving requests.

`POST /api/predict` receives a trial ID and bounded experiment settings. It retrieves the processed signal, applies the requested disturbance, runs inference, applies abstention and computes explanations. The recorded label is looked up only for comparison after inference. The browser draws the returned signal and moves a simulated cursor when appropriate.

The container runs as an unprivileged user. GitHub Actions checks source formatting, JavaScript syntax, data split integrity, artifact integrity and API behaviour, then builds and smoke-tests the actual Docker image. The public app exposes a health endpoint and an interactive API specification.

**Try:** run `python scripts/smoke_test.py --url https://intentlab-bci.vercel.app`. Then read `tests/test_api.py`, especially the test that changes a recorded label and checks that the prediction stays the same.

## The completed follow-up and a sensible next experiment

The [adaptation follow-up](../experiments/adaptation-v1/README.md) now compares personal temperature scaling, Euclidean alignment and their combination using ten early trials and evaluation on later runs, repeated across three training seeds. Alignment's mean balanced accuracy was higher, but the paired interval included no improvement, and none of the conditions met the validation reliability target. The deployed decoder is unchanged. Read the full comparison before using its numbers, because it has a different evaluation block from the original result above.

An informative next experiment would test the fixed method on fresh participants or an independently sourced dataset with equivalent task labels and electrode definitions. Write its protocol and stopping rule before evaluating it. The existing benchmark has now been inspected repeatedly, so further improvements on it cannot be described as a new independent confirmation.

Streaming support would need causal filtering, window timing, artefact handling and a new evaluation. Connecting a headset without redesigning those parts would not turn this offline model into a validated live BCI.

## CV wording supported by this repository

**IntentLab — motor-imagery EEG decoding | Python, PyTorch, ONNX, FastAPI, Docker, GitHub Actions**

Built and deployed an EEG decoding experiment using 4,766 public trials from 106 participants; compared signal-processing baselines with a compact CNN using participant-disjoint evaluation. Implemented live inference, confidence-based abstention, channel-occlusion explanations, reproducible data checks and automated API/container tests.

Be ready to explain the choices and reproduce an experiment. The value is the complete, inspectable process and your understanding of its limits, rather than implying that this solves thought reading.
