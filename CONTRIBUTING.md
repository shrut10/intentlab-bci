# Contributing to IntentLab

IntentLab is a recorded-data teaching and research project. Contributions should help someone inspect the evidence, reproduce a result or understand a failure. The live app does not control a headset or decode unrestricted thoughts.

## Useful starting points

- Try the [20-minute teaching exercise](docs/TEACHING.md) and report where the explanation or interface was confusing.
- Rebuild the coverage table with [the small worked example](examples/coverage_audit.py), recording the commit and environment you used.
- Reproduce a [participant reliability audit](docs/RELIABILITY.md), or propose a synthetic counterexample to its handling of sparse accepted predictions, bootstrap uncertainty or cohort boundaries.
- Check keyboard navigation, small screens and the clarity of error messages.
- Propose an independently evaluated extension with a fixed data split, adaptation budget and stopping rule before running it.

For a bug, use the repository's bug-report form with steps to reproduce and the relevant page or command. For a reproduction or methodological observation, use the research-feedback form. Public issues should contain public or synthetic examples only; do not attach personal EEG, credentials or identifiable participant information.

## Development

Use Python 3.12 and install `requirements-dev.txt` in a virtual environment. Training dependencies are separate in `requirements-training.txt`. Before submitting a code change, run the checks relevant to it:

```sh
ruff check .
ruff format --check .
node --check web/app.js
node --check web/reliability.js
python scripts/render_adaptation_report.py --check
python examples/reliability_audit.py --check
pytest -q
```

Describe the user-visible change, the evidence supporting it and the checks run. Mention whether AI assistance was used where it helps a reviewer understand the contribution; contributors remain responsible for reviewing and understanding their changes.

## Preserve the experiment record

The original artefacts and the completed adaptation study are archived results. Their hash checks are deliberate. Do not overwrite an evaluation, revise a protocol after seeing its outcomes or change the headline by editing a web page. A new experiment needs its own versioned protocol, output directory and discussion of whether the data have already been inspected. Treat the existing test population as used.

Temperature scaling changes probabilities, not left/right decisions. Coverage, accepted accuracy and the number of retained observations must be reported together. Repeated training seeds do not create new participants, and the 126 display examples are not the full benchmark.

Original code is MIT licensed, while public-data derivatives retain the attribution in [NOTICE.md](NOTICE.md). New dependencies and datasets need clear, compatible reuse terms and primary source citations. A proposed hardware extension should state its reference, timing and evaluation assumptions before being described as live BCI support.
