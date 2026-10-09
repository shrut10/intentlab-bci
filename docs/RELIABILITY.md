# A participant-aware reliability audit

The [public audit](https://intentlab-bci.vercel.app/research/reliability) answers a practical question: when a decoder abstains, which participants receive commands, and how much evidence supports the accepted predictions? It reanalyses the archived adaptation study without retraining or changing the deployed model. It is also a reusable tool for matched binary prediction tables from other projects.

## A result that average accuracy misses

For the adaptation baseline at seed 2027 and threshold 0.75, 9 of 630 test trials were accepted and all nine were correct. Only 3 of the 21 test participants received any command; 18 received none. For alignment plus personal temperature at the same seed and threshold, 182 trials were accepted, 158 were correct and 7 participants still received no commands. This is a descriptive comparison on the previously inspected test population, not an independently confirmed benefit. No condition met the original validation reliability target.

The original deployed model's 27/945 result belongs to a different evaluation and must not be replaced by these follow-up numbers. Three seeds repeat predictions for the same people and are deliberately displayed separately.

## Use your own predictions locally

Clone the repository and use Python 3.12. For the custom CSV path, NumPy is the only third-party dependency; it does not import the EEG pipeline or require PyTorch, a GPU, raw EEG, an account or a model API.

```sh
python3 -m venv .venv-audit
source .venv-audit/bin/activate
pip install numpy
python examples/reliability_audit.py --input predictions.csv --output audit.json
```

The full development environment in `requirements-dev.txt` is the tested, pinned environment. An isolated install of a different NumPy version may produce small floating-point differences. Run the commands from the repository root; the script resolves its own imports if invoked by an absolute path elsewhere.

The CSV needs these columns; additional columns are ignored:

```csv
condition,seed,split,subject,trial_id,label,probability
model_a,1,validation,V01,V01-001,0,0.12
model_a,1,validation,V01,V01-002,1,0.82
model_a,1,test,T01,T01-001,0,0.31
model_a,1,test,T01,T01-002,1,0.54
```

This four-row example illustrates the format only and cannot support a reliability claim. `probability` must be the probability of class 1, not maximum confidence; `label` is 0 or 1. Ties at 0.5 predict class 1, and equality at the confidence threshold is accepted. Use pseudonymous participant identifiers and globally unique trial IDs, including a dataset identifier if needed.

Every condition and training seed requires both splits, with identical trial sets across compared conditions/seeds. Validation and test participants must be disjoint. Duplicate predictions within a group, changed participant/label/split metadata for the same trial, non-finite probabilities and unmatched trial sets are rejected. A study with within-person validation needs a different declared evaluation contract; do not rename people to bypass this check.

The audit cannot prove that a submitted model was trained without leakage, that its calibration data were separate from evaluation, or that the supplied labels are correct. Those are provenance obligations for the contributor.

Open the generated JSON using **Open a local audit JSON** on the public audit page, or serve the application locally. The file is read in the browser without uploading it. The viewer labels a custom report as unverified and distinct from the published study. Files must be under 5 MB. Keep a local copy; reloading the page restores the published study.

## Fix the selection rule before inspecting test results

The default rule reproduces IntentLab's original settings: choose the lowest threshold in 0.50, 0.55, …, 0.95 achieving at least 75% accepted accuracy, 20% coverage and 100 retained validation trials; otherwise fall back to 0.75 and report failure. These values are a research choice, not a universal BCI safety standard. Validation target success is based on point estimates and is not a statistical guarantee.

To use a different prespecified rule, supply a JSON file with `--rule rule.json`:

```json
{
  "thresholds": [0.5, 0.6, 0.7, 0.75, 0.8, 0.9],
  "target_accuracy": 0.8,
  "target_coverage": 0.2,
  "target_retained": 100,
  "fallback_threshold": 0.75
}
```

Only validation predictions select the threshold. Test curves are descriptive; using them to select a new operating point consumes the test set and requires a fresh evaluation. This CLI does not fit or recalibrate a model. The bundled study always uses its archived rule.

## Quantities and uncertainty

- **Coverage:** accepted trials / all trials, pooled across participants.
- **Accepted accuracy:** correct accepted predictions / accepted predictions; undefined when the denominator is zero.
- **Selective risk:** 1 − accepted accuracy, also undefined when no predictions are accepted.
- **People without commands:** participants with zero accepted trials, still counted in the cohort.
- **Median participant coverage:** median of each person's accepted fraction, including zeros.
- **Participant balanced accuracy:** mean recall over classes 0 and 1, using all trials; undefined if either class is absent. It is independent of the rejection threshold.

Percentile 95% intervals use 2,000 draws of whole participants with replacement and seed 4027, keeping each participant's trials together and the model and threshold fixed. Each draw recomputes the pooled ratio, so these are intervals for pooled coverage/accepted accuracy, not the equally weighted mean of participant accuracies. The same deterministic draw schedule is used within each cohort; no hypothesis test or paired improvement interval is inferred from overlapping marginal intervals.

Coverage intervals require at least two participants. Accepted-accuracy intervals require at least two participants with any accepted trial and condition on bootstrap draws with a nonzero accepted denominator; `accuracy_bootstrap_valid` reports how many draws are defined. Undefined draws are not assigned zero, and a high fraction of undefined draws is a warning about sparse evidence. No interval is reported if fewer than two people contribute accepted predictions.

If every accepted prediction was correct, empirical resampling can yield a 100–100% interval. This degenerate interval does not establish perfect performance: it cannot model errors absent from the sample. The page explicitly retains this limitation alongside counts. These intervals also exclude retraining, split selection, calibration-fitting uncertainty and deployment shift, and do not justify clinical, safety or live-control claims. A future study should choose its inferential method and sample size before viewing outcomes.

## Reproduce the public report

With `requirements-dev.txt` installed:

```sh
python examples/reliability_audit.py --output /tmp/intentlab-audit.json
python examples/reliability_audit.py --check
```

The first command verifies SHA-256 hashes for the archived Parquet predictions and evaluation, then creates a new JSON file without overwriting an existing one. The second reconstructs the report and compares it with `web/reliability.json`; CI runs this check. The source hashes, bootstrap settings and threshold rule are included in the output. Exact byte equality is a check for the pinned CI environment, not a promise across arbitrary NumPy versions.

Implementation: `src/intentlab/reliability.py`; CLI: `examples/reliability_audit.py`; viewer: `web/reliability.html`, `web/reliability.js`, `web/reliability.css`. Original code is MIT licensed; the bundled PhysioNet derivatives retain [ODC-BY attribution](../NOTICE.md). Cite the software version and original study/data, not this audit as a peer-reviewed publication.

## Background and feedback

The distinction between probability calibration and classification is described by Guo et al. (2017), [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html). Risk–coverage analysis is discussed by Geifman and El-Yaniv (2019), [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html); IntentLab uses a post-hoc confidence threshold, not their jointly trained rejector. Cross-dataset evaluation is motivated by Jayaram and Barachant (2018), [MOABB: Trustworthy algorithm benchmarking for BCIs](https://arxiv.org/abs/1805.06427). These papers motivate aspects of the evaluation; they do not validate this tool's bootstrap procedure or establish its adequacy for small accepted samples.

Please report a reproduction, counterexample or methodological concern through the [research feedback form](https://github.com/shrut10/intentlab-bci/issues/new?template=research.yml). Especially useful: a synthetic case the audit mishandles, an independently reproduced operating point, or a proposed treatment of sparse accepted predictions with an explicit estimand. Include the software commit and environment; avoid attaching private neural data.
