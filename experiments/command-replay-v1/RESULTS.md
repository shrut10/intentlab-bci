# Findings from the recorded command study

The task detector reduced unwanted commands during instructed rest, but it also removed many useful responses. Neither condition met the prespecified validation target, so this experiment does not justify replacing IntentLab's deployed model or claiming a reliable live controller.

The experiment used the original Zhou2016 archive, with four participants and three sessions per person. Session 1 fitted each participant's models, session 2 selected the command policy and session 3 supplied the final evaluation. The [protocol](PROTOCOL.md) and the [source reader amendment](AMENDMENT-01.md) were committed before fitting. No model or policy settings were changed after the final results were inspected.

## Final session

Both conditions used the prespecified fallback entry threshold of 0.9, a one second hold, release threshold 0.4 and cooldown of 1.5 seconds. There were 400 left or right cues and 30 minutes of eligible rest in the final session recordings.

| Measure | Movement confidence alone | With task detector |
| :--- | ---: | ---: |
| Cues whose first command was correct | 189 / 400, 47.25% | 71 / 400, 17.75% |
| Cues receiving any correct command | 200 / 400 | 74 / 400 |
| Cues receiving no command | 169 / 400 | 313 / 400 |
| Commands during eligible rest | 174 | 27 |
| Commands per minute of eligible rest | 5.80 | 0.90 |
| Wrong hand commands during hand cues | 44 | 17 |
| Commands during feet imagery | 103 | 24 |
| Commands during unknown periods | 243 | 35 |
| Median time to first correct command | 2.802 seconds | 3.166 seconds |

The latency medians describe only cues that received a correct command, with 200 and 74 observations respectively. Missing responses are not assigned a zero latency, and these medians should not be compared as though they described the same set of successful cues. A command is scored at the end of its available input window, while the controller runs continuously without being told when a cue begins.

The rest denominator consists of 600 guarded three second intervals, for 1,800 seconds in total. There were no prediction gaps above the allowed 0.75 seconds and no gaps within those rest intervals. The recorded files total 7,070.8 seconds, of which 7,052 seconds lie between the first and last prediction timestamps in each run. The remaining 18.8 seconds at recording edges are reported separately.

## Differences between people

Each participant contributed 100 hand cues and 7.5 minutes of eligible rest to the final session.

| Participant | First command correct, confidence alone | First command correct, task detector | Idle commands per minute, confidence alone | Idle commands per minute, task detector |
| :--- | ---: | ---: | ---: | ---: |
| S1 | 52% | 1% | 3.73 | 0.27 |
| S2 | 35% | 9% | 5.60 | 1.07 |
| S3 | 65% | 9% | 5.07 | 0.40 |
| S4 | 37% | 52% | 8.80 | 1.87 |

The pooled idle rate of 0.9 does not describe every participant. The detector almost always withheld useful commands for S1, whereas S4 received more correct first commands but remained above the idle rate target. Four participants do not support a population estimate or a claim that this pattern will generalise to a new headset user.

## Validation and policy selection

Session 2 contained 390 hand cues and 1,758 seconds of eligible rest. No movement only policy in the 15 candidate settings reached the idle limit of one command per minute. Its fallback produced 138 correct first commands, or 35.38% of cues, at 5.29 idle commands per minute.

For the task detector, the only candidate satisfying the idle limit was entry threshold 0.9 with a one second hold. It produced 57 correct first commands, or 14.62% of cues, at 0.785 idle commands per minute. This failed the second requirement of a correct first command on at least half the cues. The fallback therefore applied to both conditions, and the final session did not influence policy selection.

## What can and cannot be concluded

This baseline demonstrates a measurable tension between suppressing unwanted commands and responding to intended movement. A low idle rate by itself would conceal substantial failure to respond. A useful controller needs both measures and individual results, which is why the public audit exposes them together.

The recordings come from a cue driven experiment. The arrow shown during imagery differs from the black screen shown during relaxation, so a task detector could use visual response differences or artefacts as well as imagery related activity. Rest labels are inferred from the published timing procedure and are not measurements proving the absence of intention. This study therefore does not establish spontaneous intention detection, independent control, or transfer to people not present in training.

The original source also contains inconsistent sample count headers and several runs with trial counts different from the paper's description. These discrepancies were recorded before fitting, and the supported reader correction is preserved in the amendment. All source annotations were retained in provenance; no trials were discarded to force the expected count.

The contribution is a reusable command audit and a reproducible counterexample to reporting idle suppression alone as progress. A later study could compare better task detectors on an independent recording session, but it would need a new protocol and fresh evaluation data. The present final session should not become a tuning set.

## Inspect the evidence

The [full evaluation](../../artifacts/command-replay-v1/evaluation.json) contains every validation candidate, participant results, event results and emitted commands. The [provenance](../../artifacts/command-replay-v1/provenance.json) records archive and recording hashes, software versions, the source reader amendment commit and executable source hashes. The [artifact directory](../../artifacts/command-replay-v1) also contains prediction and event CSV files that can be passed to the standalone command audit without loading EEG or fitting a model.
