# Recorded command study

This experiment asks whether a simple detector of task activity can reduce unwanted commands during instructed rest. It uses a separate public EEG collection and a small reproducible model, while leaving IntentLab's deployed decoder unchanged.

The [findings](RESULTS.md) show the full tradeoff: the task detector lowered final session idle commands from 5.8 to 0.9 per minute, while the proportion of cues whose first command was correct fell from 47.25% to 17.75%. Neither condition met the validation target.

The [protocol](PROTOCOL.md) was committed before model fitting or inspection of final session outcomes. Numerical settings are in [protocol.json](protocol.json), frozen at commit `2879a01f88a9b7d59275491f6a6d48cda2826cc5`.

## What the comparison means

The movement classifier sees two seconds of recorded EEG and produces left and right scores. With confidence alone, it must still distribute those scores between the two hands when the person is resting. The second condition adds a detector trained to distinguish left or right imagery from a guarded period of instructed rest, then multiplies the movement scores by its task score. Both conditions use the same command controller.

Four participants each contributed three sessions. The first session fits each person's models, the second chooses a command policy and the third supplies the final evaluation. This is a test on later recordings from the same people. It is neither a test on unseen people nor evidence of live control.

The evaluation counts unwanted commands during eligible rest and how often the first command in a movement cue is correct. A controller that stays silent throughout has no unwanted commands but also fails to respond to intended movement. Both outcomes are reported together, including each participant's results and the delay for cues that receive a correct command.

## Reproduce

Run from the repository root with Python 3.12. The study requirements avoid installing PyTorch because this experiment uses logistic regression.

```bash
python -m venv /tmp/intentlab-command-env
/tmp/intentlab-command-env/bin/python -m pip install -r experiments/command-replay-v1/requirements.txt
/tmp/intentlab-command-env/bin/python -m pytest tests/test_command_study.py tests/test_commands.py
/tmp/intentlab-command-env/bin/python scripts/run_command_study.py \
  --protocol-commit 2879a01f88a9b7d59275491f6a6d48cda2826cc5 \
  --source-layout-commit a1daaa71a6ff616150d88a67b91fb5204ba01a8f \
  --download \
  --output-dir /tmp/intentlab-command-reproduction
```

The explicit download flag retrieves the original 156 MB archive into ignored `data/raw/zhou2016`. The script verifies the publisher's size and MD5, computes SHA256, validates archive paths and reads only the 24 expected CNT recordings. Existing cached data is verified again. It refuses to overwrite a completed evaluation, so reproductions should use a separate output directory.

The [source reader amendment](AMENDMENT-01.md) was also committed before fitting. It documents inconsistent sample count headers and trial counts in the original archive, and specifies supported reader arguments that reconstruct valid lengths from file bytes. The original files are unchanged. Exact per file annotation counts and lengths are verified against [source_layout.json](source_layout.json).

The source record states CC0. The later BIDS conversion distributed through MOABB uses a separate CC BY 4.0 record and is not the source used here. Raw EEG is not included in the repository. The outputs contain decision scores, event intervals, command audits, candidate policy results and source provenance.

## Scientific boundaries

The continuous filter uses past samples only and resets between recording runs. Features come from trailing windows that never cross sessions. Training accepts only windows fully contained in a labelled imagery period or an eligible rest interval. Foot imagery is not used for fitting or policy selection and is reported as other activity.

The source has imagery annotations, not explicit measurements of rest or the absence of intention. Rest is inferred from the published procedure only between 5.5 and 8.5 seconds after a cue, when the whole interval fits before the next preparation period and inside the recording. Other time remains unknown. The test labels never trigger or reset the command controller.

The product of movement probability and task probability is a decision score rather than a validated joint probability. The validation target requires no more than one idle command per minute and a correct first command on at least half the movement cues. If no policy meets both requirements, the prespecified conservative fallback is used and target failure is reported openly.

## Sources

1. [Zhou et al. (2016), original research article](https://doi.org/10.1371/journal.pone.0162657), supplies the recording procedure and original dataset
2. [Original Figshare record and metadata](https://api.figshare.com/v2/articles/2061654), including archive size, publisher checksum and CC0 licence declaration
3. [Historical MOABB CNT reader](https://github.com/NeuroTechX/moabb/blob/v1.2.0/moabb/datasets/Zhou2016.py), documenting source file names and cue codes
4. Carrara and Papadopoulo (2023), [Pseudo online framework for BCI evaluation: A MOABB perspective](https://arxiv.org/abs/2308.11656), motivates evaluation on continuous recorded streams with idle periods
5. Liu et al. (2026), [Temporal Out of Distribution Detection for Asynchronous Motor Imagery Brain Computer Interfaces](https://arxiv.org/abs/2605.01014), a preprint motivating the distinction between task detection and movement classification; this experiment does not reproduce its neural network or claim its results
