# IntentLab research and community roadmap

Updated 14 November 2026, with completed evidence in the original, adaptation, pretraining, command and new participant reports; this roadmap sets priorities rather than preregistering future experiments, and none of the completed studies establishes reliable live control

## November research update

The [command audit](COMMANDS.md) now accepts timestamped scores and annotated intervals from other models, applies an explicit waiting and release rule, and reports unwanted commands, missed cues, delays and prediction gaps. Its local browser viewer allows inspection without uploading a report.

The separate [Zhou2016 study](../experiments/command-replay-v1/README.md) uses four people, with session 1 for fitting, session 2 for selecting a policy and session 3 for final evaluation. The protocol and a source header correction were committed before fitting. A task detector reduced final session idle commands from 5.8 to 0.9 per minute, but correct first commands fell from 189 of 400 movement cues to 71 of 400. Both conditions failed the validation target. This identifies a practical tradeoff rather than a replacement for the deployed model.

The subsequent [Dreyer2023 study](../experiments/unseen-v1/README.md) evaluates separately fitted methods on 27 people absent from fitting and selection, giving a successful but bounded discrimination result; transfer of the deployed CNN to another acquisition setting, prospective interaction and independent reproduction remain unresolved

The [research review](RESEARCH_REVIEW_2026_11.md) records the primary papers, student projects and community requests behind these priorities

## October research update

The [masked-channel pretraining study](../experiments/pretraining-v1/README.md) adds an exploratory comparison using only source-training EEG, two supervised-label budgets and three fixed seeds. Its evaluation is restricted to the previously inspected validation cohort, so independent data remain the highest scientific priority. The deployed decoder is unchanged. The site now distinguishes server processing, browser round-trip and recorded-window duration, with a reproducible request-latency benchmark.

The [NoMAD metadata audit](NOMAD_DATA.md) records a separately licensed intracortical resource and its expected checksums. The tested recording endpoints rejected access; no NoMAD recordings have been ingested. This is preparation for a distinct future study, not an additional EEG benchmark.

## 1. Make existing evidence useful to other people — implemented, external validation pending

The [reliability audit](RELIABILITY.md) accepts matched prediction CSVs, preserves participant/seed boundaries, exposes zero-command participants, and provides reproducible participant bootstrap diagnostics. The browser viewer supports local JSON without uploading it. The teaching exercise and archived predictions make a contribution possible without owning EEG equipment.

Evidence of usefulness would be an independent reproduction, a concrete methodological correction, or an educator reporting how the exercise worked. A resource-list entry records discoverability; stars, automated clones and page views do not establish scientific impact. No external audit reproduction is claimed yet.

## 2. Evaluation on unseen participants completed in a separate corpus

The frozen [Dreyer2023 protocol](../experiments/unseen-v1/PROTOCOL.md) assigns 45 people to training, 15 to validation and 27 to final testing, comparing bandpower and covariance tangent features with fixed logistic regression settings and no target adaptation

The selected bandpower method reached 66.02% mean participant balanced accuracy with a 95% interval of 61.80% to 70.23%, satisfying the declared discrimination criterion; the separate confidence rule failed its validation requirement for minimum predictions per person, while its test fallback retained only 18.94% of trials

The repository now contains the source and split manifests, committed validation selection, runnable methods, complete predictions, participant uncertainty and generated report; independent CSV checks reproduce the numerical findings, while a source review documents visual cue and sham feedback limitations

This study uses the MOABB distributed source conversion without running a MOABB benchmark, and it does not test the original deployed CNN on the new corpus; its final cohort is now inspected, so future model changes require a new declared evaluation design and cannot reuse these outcomes as fresh evidence

The next useful step is independent reproduction, followed by a new protocol that tests transfer or adaptation in a different acquisition setting while preserving separate calibration and final evaluation data

## 3. Extend recorded command evaluation to prospective interaction

The original deployed model uses zero phase filtering of complete three second epochs and assumes every trial is left or right imagery. The separate command study now tests past sample filtering, instructed rest and a tested temporal controller. Its rest labels are inferred from the source procedure, and the task detector may partly distinguish screen conditions or artefacts rather than intention. A prospective study should record explicit task and rest annotations, device timestamps, acquisition gaps and physical responses under a new frozen protocol.

Acceptance evidence: future-sample perturbations cannot affect earlier outputs; intended-command success and false activations during recorded idle periods are both measured; threshold decisions use independent validation; latency includes acquisition and processing. Existing recordings can support an offline feasibility study, but live robustness still needs prospective data and suitable hardware. An extra rest class alone does not establish reliable asynchronous control.

## 4. Build a bounded physical replay after the scientific interface is explicit

A low-cost command display or small rover can demonstrate delivery, expiry and stopping with recorded predictions. Add sequence numbers, timestamps, command expiry and acknowledgement; distinguish receiving a command from observing the physical movement. Test stale packets, disconnection and abstentions, and report device failures separately from decoder errors.

Acceptance evidence: a reproducible hardware bill, clear replay labelling, measured failure cases and a short unedited demonstration. Hardware remains optional; it would not improve EEG accuracy or convert this project into a validated live BCI. Hexbug acquisition remains on hold.

## Community work with specific asks

| Community | Useful contribution | Ask | Current boundary |
| --- | --- | --- | --- |
| NeuroTechX and educators | Existing teaching demo, now with participant audit | Try one exercise and identify one unclear concept or reproduce one operating point | Already listed in awesome-bci; do not resubmit the same listing |
| MOABB developers | Tested chronological calibration example | Review the split contract before a benchmark extension | A runnable synthetic example with 16 tests was shared on issue #1077 after a maintainer's invitation; it is not a real-data MOABB benchmark |
| neoxai curator | Local prediction-audit method with licence/provenance | Consider a methods entry and review the sparse-coverage treatment | Respect the repository's documented PR format; catalogue inclusion is not endorsement |
| Open Neuroscience | Open-source teaching/evaluation resource | Consider a resource entry under the site's submission process | Submission is distinct from acceptance |
| NeuroTechX London / Slack | Short practical demonstration and a focused question | Find a peer willing to reproduce one result or test the teaching exercise | Requires community access and checking the actual channel rules |
| OpenBCI Community | Finished tutorial explaining what must change before live use | Review acquisition/reference assumptions | Contributor registration needs approval; no claim that a supported live OpenBCI integration exists |

Useful outcomes to record: reproducible bug reports resolved, independent reruns, teaching reuse, and accepted contributions that others can inspect. Do not attribute a scientific endorsement, affiliation or collaboration to someone merely because they accepted a link or left a comment.

Public entry points: [NeuroTechX community](https://neurotechx.com/community/), [OpenBCI contribution process](https://docs.openbci.com/GettingStarted/Community/Community/), [neoxai contribution guidelines](https://github.com/neowalter/neoxai#contributing), [existing MOABB discussion](https://github.com/NeuroTechX/moabb/issues/1077#issuecomment-6058411343).
