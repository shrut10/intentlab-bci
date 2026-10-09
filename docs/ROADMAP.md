# IntentLab research and community roadmap

Updated 9 October 2026. This is a prioritised plan, not a report of completed experiments or a preregistration. Completed evidence lives in the original and adaptation reports; neither establishes reliable live control.

## 1. Make existing evidence useful to other people — implemented, external validation pending

The [reliability audit](RELIABILITY.md) accepts matched prediction CSVs, preserves participant/seed boundaries, exposes zero-command participants, and provides reproducible participant bootstrap diagnostics. The browser viewer supports local JSON without uploading it. The teaching exercise and archived predictions make a contribution possible without owning EEG equipment.

Evidence of usefulness would be an independent reproduction, a concrete methodological correction, or an educator reporting how the exercise worked. A resource-list entry records discoverability; stars, automated clones and page views do not establish scientific impact. No external audit reproduction is claimed yet.

## 2. Evaluate on independent data with a frozen protocol — highest scientific priority

PhysioNet's current test set has already informed the follow-up and must be treated as inspected. Before another performance comparison, choose an obtainable, appropriately licensed motor-imagery dataset and state its channel/reference differences, label mapping, sampling, participant split, calibration budget, primary metric and stopping rule. A common-channel pipeline must be fitted and validated for that acquisition setting; the existing nine-channel/64-channel-reference model cannot simply be presented as headset-independent.

Compare classical CSP/LDA and a small neural model under the same protocol, reporting all seeds and participant-level paired differences. Separate confidence fitting and threshold selection where sample size permits, and report the limitation if data are reused. Include coverage, wrong commands, zero-command participants and uncertainty rather than ranking methods by accuracy alone. Freeze the protocol before loading final held-out outcomes and publish unsuccessful results as well.

Acceptance evidence: a versioned protocol, auditable provenance and split manifests, runnable baseline implementation, frozen predictions, participant-level uncertainty and a clearly delimited generalisation claim. A MOABB-based implementation should name the exact version and evaluation contract and pass a synthetic split test before any compatibility claim. The project has not yet run a MOABB benchmark.

## 3. Investigate unintended commands and causal processing — prerequisite for live EEG

Current epochs use zero-phase filtering of complete three-second windows and assume every trial is left/right imagery. Neither idle periods nor real electrode loss are represented by the current model. Add a new data/evaluation design with an explicit rest/no-intention condition and preprocessing that uses only samples available at the prediction time. Define warm-up, reset, gaps, timestamps and end-to-end latency.

Acceptance evidence: future-sample perturbations cannot affect earlier outputs; intended-command success and false activations during recorded idle periods are both measured; threshold decisions use independent validation; latency includes acquisition and processing. Existing recordings can support an offline feasibility study, but live robustness still needs prospective data and suitable hardware. An extra rest class alone does not establish reliable asynchronous control.

## 4. Build a bounded physical replay after the scientific interface is explicit

A low-cost command display or small rover can demonstrate delivery, expiry and stopping with recorded predictions. Add sequence numbers, timestamps, command expiry and acknowledgement; distinguish receiving a command from observing the physical movement. Test stale packets, disconnection and abstentions, and report device failures separately from decoder errors.

Acceptance evidence: a reproducible hardware bill, clear replay labelling, measured failure cases and a short unedited demonstration. Hardware remains optional; it would not improve EEG accuracy or convert this project into a validated live BCI. Hexbug acquisition remains on hold.

## Community work with specific asks

| Community | Useful contribution | Ask | Current boundary |
| --- | --- | --- | --- |
| NeuroTechX and educators | Existing teaching demo, now with participant audit | Try one exercise and identify one unclear concept or reproduce one operating point | Already listed in awesome-bci; do not resubmit the same listing |
| MOABB developers | Chronological calibration use case and future tested integration | Review the split contract before a benchmark extension | A use-case comment was posted on issue #1077; wait for discussion rather than repeatedly bumping it |
| neoxai curator | Local prediction-audit method with licence/provenance | Consider a methods entry and review the sparse-coverage treatment | Respect the repository's documented PR format; catalogue inclusion is not endorsement |
| Open Neuroscience | Open-source teaching/evaluation resource | Consider a resource entry under the site's submission process | Submission is distinct from acceptance |
| NeuroTechX London / Slack | Short practical demonstration and a focused question | Find a peer willing to reproduce one result or test the teaching exercise | Requires community access and checking the actual channel rules |
| OpenBCI Community | Finished tutorial explaining what must change before live use | Review acquisition/reference assumptions | Contributor registration needs approval; no claim that a supported live OpenBCI integration exists |

Useful outcomes to record: reproducible bug reports resolved, independent reruns, teaching reuse, and accepted contributions that others can inspect. Do not attribute a scientific endorsement, affiliation or collaboration to someone merely because they accepted a link or left a comment.

Public entry points: [NeuroTechX community](https://neurotechx.com/community/), [OpenBCI contribution process](https://docs.openbci.com/GettingStarted/Community/Community/), [neoxai contribution guidelines](https://github.com/neowalter/neoxai#contributing), [existing MOABB discussion](https://github.com/NeuroTechX/moabb/issues/1077#issuecomment-6058411343).
