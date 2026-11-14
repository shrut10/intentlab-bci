# Research priorities for IntentLab

Reviewed 14 November 2026. This review compares public research, student projects and requests from software maintainers. It is a deliberately selected sample, not a survey of all graduates or a ranking of individual students. Project descriptions establish what their authors report, not independent verification of their performance.

## The strongest direction

IntentLab should help researchers and students measure whether EEG predictions can produce useful commands. Its value lies in making the decision process inspectable: when it acts, when it waits, how often it acts during rest, and whose intended commands it misses. This is a practical research direction rather than a claim that the project has invented asynchronous BCI evaluation.

The existing project already has deployment, compact models, participant separation, explanations and reproducible reports. Another interface around a classifier would add less value than testing the gap between prediction and action. Independent recordings are also necessary because the original PhysioNet test cohort has already informed subsequent experiments.

## What the field supports

| Evidence | Implication for this project |
| --- | --- |
| Carrara and Papadopoulo's [evaluation framework](https://arxiv.org/abs/2308.11656) uses sliding windows and an idle state to study differences from conventional offline evaluation | Include recorded time outside movement cues and measure emitted commands, rather than interpreting window accuracy as control performance |
| Liu and colleagues' [2026 preprint](https://arxiv.org/abs/2605.01014) separates task detection, movement classification and unfamiliar activity | A modest task gate is worth comparing with confidence alone; the paper does not establish that IntentLab's implementation will improve |
| The [MOABB benchmark](https://arxiv.org/abs/2404.15319) compares 30 pipelines over 36 datasets and reports strong covariance based methods | Larger neural networks are not automatically the most useful next step; future model comparisons should include serious classical baselines |
| Braindecode's [practical tutorial request](https://github.com/braindecode/braindecode/issues/610) asks for examples closer to EEG applications and VR | A reusable example with explicit timing and decision rules answers an existing need; the discussion already mentions work planned by a maintainer |
| MOABB's [preprocessing documentation request](https://github.com/NeuroTechX/moabb/issues/743) and [Cho2017 investigation](https://github.com/NeuroTechX/moabb/issues/1340) concern interpretation of preprocessing and event timing | Document what information is available when a prediction is made, and test that future signal changes cannot affect earlier outputs |
| The [accepted Cho2017 fix](https://github.com/NeuroTechX/moabb/pull/1341) already addresses that particular issue | Do not create a duplicate contribution or imply that the same defect was found in IntentLab |

The task gate study is motivated by these sources but does not reproduce their architectures, datasets or performance claims. It is a separate small experiment with a fixed protocol.

## What students and graduate researchers publish

| Public example | Verified context and transferable lesson |
| --- | --- |
| [Neurotech at USC](https://github.com/Neurotech-BCI/ErrP) | A 2026 conference project combines EEG collection, streaming and cursor tasks; its reported participant group narrowed to two, illustrating the importance of keeping scope visible |
| [EEGo Lab](https://eego-unc.github.io/Lab/) and [EEGProc](https://github.com/EEGo-UNC/EEGProc) | A student research group publishes reusable processing software and worked documentation, giving other people a reason to adopt its work |
| [Muse course project](https://github.com/kkentia/muse_cyb_sys) from the [Fribourg programme](https://digital-neuroscience.human-ist.ch/projects) | Separate simulator and headset modes make the source of each demonstration clear; state management matters beyond the classifier |
| [Hannah Portmann's thesis](https://github.com/portmannh/brain-age-from-sleep) | A narrowly defined sequence of sleep representation learning and downstream tasks makes a research contribution easier to inspect |
| [Yanzhe Zhao's OpenBMI project](https://github.com/JustinZHAO-05/time-frequency-analysis-eeg-openbmi) | Signal examples, frequency analysis and a report help explain the physiology; a convincing visualisation is not itself a model explanation |
| [Fribourg gaze and speech editor](https://github.com/DiogoRoch/SR_Gaze_TextEditor) | A concrete communication task provides a clear demonstration of utility, while its input modalities remain distinct from EEG |
| [Triton NeuroTech RC car](https://sites.google.com/ucsd.edu/triton-neurotech/projects/eeg-rc-car) | Student progress reports expose practical recording, streaming and device communication problems that a tested replay tool can address |
| [Neuromatch RECAP](https://github.com/obayalalousi/ecog-noisy-faces-recognition) | A course project presents a specific neural response question with analysis materials; stimulus recognition from ECoG is not free thought reconstruction |

These examples support a combination of a useful task, a reproducible method and a readable account of limitations. They do not justify comparing headline accuracies across different participants, recording systems or targets.

## Priority decisions

| Rank | Improvement | Value | Decision |
| --- | --- | --- | --- |
| 1 | Audit commands over time using an independent recording corpus | Adds a practical question, a reusable tool and evidence outside the original dataset | Implement a saved session audit and a separate Zhou2016 experiment |
| 2 | Make the audit usable with other researchers' scores | Allows adoption without EEG hardware or retraining | Provide a documented CSV contract, local report import and a command log |
| 3 | Measure transfer on a larger independent participant population | Would strengthen generalisation claims | Retain as the next scientific priority; four people do not settle it |
| 4 | Compare covariance methods and calibration budgets | Provides a stronger modelling comparison than adding a larger network without a reason | Plan a separate frozen experiment |
| 5 | Add a headset, rover or VR task | Could test interaction and end to end delay | Proceed after recorded command behaviour and failure handling are understood |
| 6 | Add dream reconstruction or a foundation model wrapper | Would require a different signal modality, evidence base and likely resources | Do not attach these claims to the current scalp EEG project |

These rankings are engineering and research judgements, not measured impact scores. The chosen study uses a simple model so that changes in command behaviour can be traced to a task gate and an explicit decision policy.

## Dataset choice and provenance

The original [Zhou2016 archive](https://doi.org/10.6084/m9.figshare.2061654) is an obtainable recording corpus with four participants and three sessions per person. The [source paper](https://doi.org/10.1371/journal.pone.0162657) documents movement cues and instructed relaxation. Rest intervals in this extension are inferred conservatively from that protocol and are not direct measurements of intention.

The original Figshare metadata declares CC0. The [later BIDS conversion](https://zenodo.org/records/16534752) used by current MOABB has its own CC BY 4.0 metadata. These are different distributions; the difference is not evidence of a licence bug. The experiment records its exact source, publisher checksum and local SHA256 and does not redistribute raw EEG.

[AlexMI](https://zenodo.org/records/806023) offers explicit rest trials and eight participants, but its movement classes do not match the current left and right hand focus. It remains a useful alternative for a future study. BNCI2014 datasets are established benchmarks, but their catalogue licence and different acquisition setups need to be considered before distributing transformed signals.

The [command study protocol](../experiments/command-replay-v1/PROTOCOL.md) was committed before fitting or inspecting final session outcomes. Session 1 fits participant specific models, session 2 selects policies and session 3 evaluates them. This is transfer to later sessions of the same people, not transfer to unseen people, and does not establish prospective control.

## A credible public identity

Use the same practical description across the repository and application: IntentLab explores EEG decoding and audits the commands that follow from its predictions. Keep the existing recorded signal demo as the introduction, with the reusable audits and research reports immediately accessible.

Evidence of impact would include an independent reproduction, an educator using an exercise, a developer finding a decision bug with the audit, or an accepted contribution that improves another project. Repository topics, resource listings and page views help people find the work but are not scientific endorsement.

NeuroTechX explicitly [invites projects to use the neurotechx GitHub topic](https://www.neurotechx.org/projects). This is an appropriate discovery route. IntentLab is already listed in Awesome BCI, so there is no reason to resubmit that listing. The existing neoxai contribution is open and free of merge conflicts, while the MOABB example has already been shared in response to a maintainer's invitation.

Any further community message should accompany a finished, tested artefact and ask a narrow question, such as whether the command definition or rest denominator is appropriate. Avoid repetitive promotional comments, invented endorsements or claims of adoption before somebody actually uses the work.
