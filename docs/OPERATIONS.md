# Running and publishing IntentLab

## Environments

- Local: Python 3.12 virtual environment in `.venv`.
- Container: Python 3.12 slim, non-root UID 10001, port 8000.
- Public: Vercel FastAPI project `intentlab-bci`; `app.py` exports the ASGI application.
- Source: `https://github.com/shrut10/intentlab-bci`, production branch `main`.

The app needs no API keys and has no paid model calls. Vercel's platform limits still apply. Serverless cold starts can be slower than warm requests. The UI's measured milliseconds cover server computation, not the network round trip or EEG acquisition.

## Startup and input boundaries

The app checks the model/report hashes in `artifacts/checksums.json`, loads the ONNX model once per process and caches the small demonstration cohort. A failed integrity check prevents readiness.

Requests accept a fixed set of trial IDs, model names, channels and noise levels. The request body is limited to 4 KiB. Confidence must be finite and between 0.5 and 0.99. Unknown fields and invalid values are rejected. Validation responses do not echo arbitrary request contents. No arbitrary file path, URL fetch, model upload or user EEG upload is exposed.

Security response headers restrict resource origins, embedding and browser permissions. The interactive API docs use their framework-provided resources. There is no custom user tracking or analytics; the hosting provider may collect normal infrastructure logs. The app returns a request ID and prediction latency. Container access logging is disabled.

The demo has no persistent distributed rate limiter. Inputs and computation are bounded, but a public service can still receive unwanted traffic. Production use at a larger scale would require platform traffic controls, monitoring, a service-level objective and a reviewed privacy policy. No such operational guarantee is claimed here.

## Verification

```sh
source .venv/bin/activate
ruff check .
ruff format --check .
node --check web/app.js
pytest -q
python scripts/smoke_test.py --url https://intentlab-bci.vercel.app
```

The CI workflow also builds the Docker image and runs the smoke test against a real container. Raw EEG downloads and model training do not run in routine CI; the tests validate the checked-in data/serving contract and frozen artifacts offline. Re-training is an explicit research action.

## Publish changes

The GitHub repository is connected to the Vercel project. A push or merged pull request to `main` triggers deployment. Check both GitHub Actions and Vercel status; the hosting integration is not gated on CI success by this repository. Use a branch and preview deployment for substantial changes.

Manual deploy from the project root:

```sh
vercel deploy --prod
```

Only public code, processed demonstration data and model artifacts are uploaded. `.vercelignore` excludes raw recordings, the full training tensor, local environments and credentials. The larger Parquet feature table stays in GitHub for analysis but is excluded from the serverless bundle. Training dependencies are kept out of `requirements.txt`.

For model changes, create a new protocol and preserve the previous report in the release history. Update all artifacts together, regenerate their checksums, test inference, and update the model card. Do not manually change a probability, accuracy or threshold in an evaluation file to improve the presentation.

## Roll back

Revert the relevant Git commit and push to `main`, or redeploy a verified prior Vercel deployment from its dashboard. Check `/health` and run the smoke test afterwards. Model, evaluation and checksum files must come from the same version.

## Text changes

See [EDITING.md](EDITING.md). Most wording is in `web/index.html`; dynamic messages are in `web/app.js`. Neither requires re-training.
