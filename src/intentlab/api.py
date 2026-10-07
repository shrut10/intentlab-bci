"""Public, bounded inference API. No accounts, secrets, uploads or paid services."""

import csv
import hashlib
import io
import json
import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from scipy.signal import welch

from intentlab import __version__
from intentlab.inference import Predictor
from intentlab.signal import CHANNELS, SFREQ, perturb

ROOT = Path(__file__).resolve().parents[2]
ModelName = Literal["bandpower", "csp_lda", "compact_eeg", "majority"]
ChannelName = Literal["FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CPz", "CP4"]


@lru_cache(maxsize=1)
def assets():
    for name, digest in json.loads((ROOT / "artifacts/checksums.json").read_text()).items():
        if hashlib.sha256((ROOT / "artifacts" / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Model artifact integrity check failed: {name}")
    predictor = Predictor()
    with np.load(ROOT / "data/demo/epochs.npz", allow_pickle=False) as data:
        x = data["x"]
    trials = json.loads((ROOT / "data/demo/trials.json").read_text())
    return {
        "predictor": predictor,
        "x": x,
        "trials": trials,
        "index": {row["id"]: i for i, row in enumerate(trials)},
        "manifest": json.loads((ROOT / "data/derived/manifest.json").read_text()),
        "evaluation": json.loads((ROOT / "artifacts/evaluation.json").read_text()),
    }


@asynccontextmanager
async def lifespan(app):
    assets()
    yield


app = FastAPI(
    title="IntentLab · motor-imagery EEG",
    version=__version__,
    lifespan=lifespan,
    description="Replay public, held-out EEG trials and run actual model inference. Inputs are completed, preprocessed epochs. Research demonstration; no live headset connection or medical use.",
)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, error: RequestValidationError):
    # Invalid NaN/Infinity inputs cannot themselves be encoded in a JSON response.
    # Return field and reason without echoing arbitrary client input.
    details = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in error.errors()]
    return JSONResponse({"detail": details}, status_code=422)


@app.middleware("http")
async def request_bounds(request: Request, call_next):
    if request.method == "POST":
        body = bytearray()
        async for part in request.stream():
            body.extend(part)
            if len(body) > 4096:
                return JSONResponse({"detail": "Request body exceeds 4096 bytes"}, status_code=413)
        request._body = bytes(body)
    response = await call_next(request)
    response.headers["X-Request-ID"] = uuid.uuid4().hex
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if request.url.path not in ("/docs", "/redoc"):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; font-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        )
    response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api") else "no-cache"
    return response


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    trial_id: str = Field(pattern=r"^S\d{3}-R(?:04|08|12)-E\d{2}$", max_length=20)
    model: ModelName | None = None
    threshold: float | None = Field(default=None, ge=0.5, le=0.99)
    noise: Literal[0.0, 0.5, 1.0] = 0.0
    drop_channel: ChannelName | None = None


def trial_index(identifier):
    try:
        return assets()["index"][identifier]
    except KeyError:
        raise HTTPException(404, "This trial is not in the published demo cohort") from None


@app.get("/health")
def health():
    data = assets()
    return {
        "status": "ok",
        "version": __version__,
        "model": data["predictor"].config["selected"],
        "runtime": "onnxruntime-cpu + numpy",
        "artifacts_verified": True,
        "demo_trials": len(data["trials"]),
    }


@app.get("/api/overview")
def overview():
    data = assets()
    report, manifest, config = data["evaluation"], data["manifest"], data["predictor"].config
    return {
        "name": "IntentLab",
        "version": __version__,
        "channels": CHANNELS,
        "sampling_hz": SFREQ,
        "window_seconds": [1, 4],
        "counts": manifest["counts"],
        "quality_exclusions": manifest["exclusions"],
        "demo_trials": len(data["trials"]),
        "selected": config["selected"],
        "threshold": config["threshold"],
        "threshold_target_met": config["threshold_target_met"],
        "models": [{"id": k, "name": v["name"]} for k, v in config["models"].items()],
        "headline": report["models"][config["selected"]]["test"],
        "operating_point": report["operating_point"],
        "source": {
            "name": "PhysioNet EEGMMIDB 1.0.0",
            "url": "https://physionet.org/content/eegmmidb/1.0.0/",
            "doi": manifest["doi"],
            "license": manifest["license"],
        },
    }


@app.get("/api/evaluation")
def evaluation():
    return assets()["evaluation"]


@lru_cache(maxsize=1)
def adaptation_report():
    """Load the separate offline study; this never changes the live predictor."""
    folder = ROOT / "artifacts/adaptation-v1"
    checksums = json.loads((folder / "checksums.json").read_text())
    for name, digest in checksums.items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Adaptation study integrity check failed: {name}")
    return json.loads((folder / "evaluation.json").read_text())


@app.get("/api/adaptation")
def adaptation_evaluation():
    """Archived four-condition comparison, not online personal calibration."""
    return adaptation_report()


@app.get("/api/adaptation/predictions")
def adaptation_predictions():
    """Download the public benchmark's validation/test probabilities as Parquet."""
    adaptation_report()
    return FileResponse(
        ROOT / "artifacts/adaptation-v1/predictions.parquet",
        media_type="application/octet-stream",
        filename="intentlab-adaptation-predictions.parquet",
    )


@app.get("/api/trials")
def trials():
    return {
        "selection": "First six valid epochs per held-out participant, in recording order; no selection by correctness",
        "trials": assets()["trials"],
    }


@app.get("/api/trials/{identifier}")
def trial(identifier: str):
    i = trial_index(identifier)
    data = assets()
    return {
        **data["trials"][i],
        "channels": CHANNELS,
        "sampling_hz": SFREQ,
        "display_stride": 2,
        "units": "Normalised amplitude (one RMS per epoch)",
        "signal": data["x"][i, :, ::2].round(4).tolist(),
    }


@app.get("/api/trials/{identifier}/csv")
def trial_csv(identifier: str):
    i = trial_index(identifier)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["seconds_after_cue", *CHANNELS])
    for sample, values in enumerate(assets()["x"][i].T):
        writer.writerow([f"{1 + sample / SFREQ:.5f}", *[f"{v:.7f}" for v in values]])
    return Response(
        output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{identifier}-normalised.csv"'},
    )


@app.post("/api/predict")
def predict(payload: PredictionRequest):
    started = time.perf_counter()
    data = assets()
    i = trial_index(payload.trial_id)
    predictor = data["predictor"]
    name = payload.model or predictor.config["selected"]
    threshold = payload.threshold if payload.threshold is not None else predictor.config["threshold"]
    x = perturb(data["x"][i : i + 1], payload.noise, payload.drop_channel, seed=2027 + i)
    # Explanation variants and prediction use the same inference path, never the label.
    variants = np.concatenate([x, *[perturb(x, drop=ch) for ch in CHANNELS]], axis=0)
    probabilities = predictor.predict(variants, name)
    right = float(probabilities[0])
    confidence = max(right, 1 - right)
    frequencies, powers = welch(x[0], fs=SFREQ, nperseg=160)
    keep = (frequencies >= 4) & (frequencies <= 40)
    accepted = confidence >= threshold
    return {
        "trial_id": payload.trial_id,
        "model": name,
        "model_name": predictor.config["models"][name]["name"],
        "model_version": __version__,
        "right_probability": right,
        "left_probability": 1 - right,
        "confidence": confidence,
        "threshold": threshold,
        "accepted": accepted,
        "decision": ("right" if right >= 0.5 else "left") if accepted else "abstain",
        "predicted_label": int(right >= 0.5),
        "recorded_label": data["trials"][i]["label"],
        "noise": payload.noise,
        "drop_channel": payload.drop_channel,
        "signal": x[0, :, ::2].round(4).tolist(),
        "spectrum": {"hz": frequencies[keep].tolist(), "power": powers[:, keep].mean(0).round(6).tolist()},
        "occlusion": [
            {"channel": ch, "right_probability_change": float(probabilities[j + 1] - right)}
            for j, ch in enumerate(CHANNELS)
        ],
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "note": "Recorded-data replay. Confidence and channel sensitivity are model estimates, not evidence of thought reading or causal brain activity.",
    }


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "web/index.html")


@app.get("/research", include_in_schema=False)
def research():
    return FileResponse(ROOT / "web/research.html")


@app.get("/research/adaptation", include_in_schema=False)
def adaptation_research():
    return FileResponse(ROOT / "web/adaptation.html")


@app.get("/favicon.svg", include_in_schema=False)
def favicon():
    return FileResponse(ROOT / "web/favicon.svg", media_type="image/svg+xml")


app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
