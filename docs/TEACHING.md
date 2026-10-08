# A 20-minute exercise in EEG decoding and uncertainty

**Audience:** students, educators and developers who know what a classifier is and want to see why accuracy alone is insufficient for a brain–computer interface.

**You need:** a browser and [the public demo](https://intentlab-bci.vercel.app). No headset, sign-up or model training is required. The optional coding exercise uses Python 3.12 and the repository's development dependencies.

By the end, you should be able to distinguish a class prediction from its confidence, explain what is lost when a decoder abstains, and identify which data a personal calibration step may legitimately use. This is an exercise using recorded public data, not a validated control system.

## 1. Inspect a prediction — about 4 minutes

Open the experiment, keep the compact CNN selected, set added noise to zero and leave the electrodes connected. Select a recording and note its trial ID so someone else can repeat your choices. Decode it and record the predicted hand, confidence, recorded cue and whether a command was accepted.

The recording is a completed three-second EEG interval. Its cue tells us which fist the participant was asked to imagine moving. The cue is used for evaluation after inference; it is not an input to the model, nor proof of a person's private thoughts.

**Discuss:** what does a confident but incorrect prediction tell you about treating a model's confidence as a guarantee?

## 2. Change the decision threshold — about 5 minutes

Keep the recording and decoder fixed. Compare a threshold of 50% with the default 75%, then replay the same participant's six displayed trials at each threshold. Count accepted commands and rejected commands separately. If every trial is rejected at 75%, record that outcome rather than searching for a more flattering sequence.

| Threshold | Accepted commands | Correct among accepted | Rejected commands |
| --- | --- | --- | --- |
| 50% | | | |
| 75% | | | |

Changing this threshold does not retrain the classifier. It changes which predictions can become commands. At 50%, every binary prediction has sufficient confidence; at higher thresholds, more can be rejected. When none are accepted, accepted accuracy is undefined, not 0% or 100%.

**Discuss:** could a system be accurate whenever it acts but still be unusable because it hardly ever acts?

The demo contains 126 examples, selected deterministically for display. Do not use your six-trial replay to replace the full original test result: the default threshold accepted 27 of 945 test trials, with 24 correct. The small accepted subset and failed validation target prevent a claim of reliable control.

## 3. Inspect a model sensitivity — about 4 minutes

Return to one unchanged recording, disconnect C3 and then C4 separately, and compare the outputs with the clean prediction. Restore each channel before changing the next. The direction and size of the effect can differ by recording, so a change is an observation rather than a required outcome.

**Discuss:** does zeroing a processed channel tell us what would happen if we bought a headset without that electrode?

It measures dependence on an artificial input intervention. The original signals were referenced using all 64 electrodes before nine were selected, and correlated channels can share information. Choosing a smaller headset would require new referencing and evaluation.

## 4. Compare personal calibration with alignment — about 5 minutes

Open [the follow-up report](https://intentlab-bci.vercel.app/research/adaptation). Compare all four conditions in Tables 2 and 4, reading accuracy and coverage together.

- The baseline uses no initial recordings from a new person.
- Personal temperature uses ten labelled initial trials to adjust confidence without changing the left/right classification.
- Euclidean alignment uses ten unlabelled initial trials to transform that person's inputs; the model is trained with the same type of alignment.
- Their combination uses those same ten trials with labels.

Later runs supply 630 evaluation trials from 21 test participants. Three model seeds give repeated predictions for those same people. They do not create 63 independent test participants or 1,890 independent trials. No condition met the validation reliability target, despite some more favourable test operating points.

**Discuss:** why would fitting an alignment matrix to all of a new person's later test recordings answer a different question from fitting it only to their first ten recordings?

## Optional coding exercise: reconstruct the coverage table

From a clone of the repository with `requirements-dev.txt` installed:

```sh
python examples/coverage_audit.py --output /tmp/intentlab-coverage.csv
```

This reads the saved trial predictions, verifies their checksum and exports one row per condition, seed, split and predeclared threshold. It reports accepted accuracy as blank when there are no accepted predictions. It does not train a model or pick a new threshold from test performance.

Choose one condition and one seed. Compare the validation and test rows at 0.75, then calculate coverage yourself as `accepted / trials` and accepted accuracy as `correct / accepted`. The accepted rows always remain part of the same evaluation cohort; a higher threshold does not produce a fresh independent test set.

The CSV includes the validation-rule threshold and whether that rule met its target. Any exploration of test thresholds is descriptive and must not be presented as validation of a newly selected operating point. Use a new, prespecified evaluation to assess a new selection rule.

## Facilitator notes and feedback

Useful answers should mention the difference between confidence and correctness, the cost of abstention, separation of calibration and evaluation data, and the limitations of artificial channel removal. The exercise is complete when a learner can explain those distinctions using a recorded observation, even if no command moved.

An educator can reuse this guide under the code repository's MIT licence, retaining the source-data attribution in [NOTICE.md](../NOTICE.md) when sharing data derivatives. For methods and academic references, use the [original report](https://intentlab-bci.vercel.app/research) and [adaptation report](https://intentlab-bci.vercel.app/research/adaptation). [CITATION.cff](../CITATION.cff) provides a software citation; include the commit you used.

If you use this exercise, an optional [teaching or research feedback issue](https://github.com/shrut10/intentlab-bci/issues/new?template=research.yml) can describe what you tried and what was confusing. There is no need to share names, participant data or personal EEG. External reproductions and improvements to the exercise are useful evidence of reuse; stars alone are not evidence of scientific or practical impact.
