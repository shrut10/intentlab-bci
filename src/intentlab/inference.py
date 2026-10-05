"""Small, pickle-free runtime shared by evaluation and the public API."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.special import expit

from intentlab.signal import bandpower, csp_features

ROOT = Path(__file__).resolve().parents[2]


class Predictor:
    def __init__(self, directory: Path = ROOT / "artifacts"):
        ort.disable_telemetry_events()
        self.config = json.loads((directory / "models.json").read_text())
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(directory / "compact_eeg.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def logits(self, x: np.ndarray, model: str) -> np.ndarray:
        spec = self.config["models"][model]
        if model == "compact_eeg":
            return self.session.run(None, {"eeg": np.asarray(x, dtype=np.float32)})[0].reshape(-1)
        if model == "majority":
            return np.full(len(x), spec["logit"])
        if model == "bandpower":
            features = bandpower(x)
        elif model == "csp_lda":
            features = csp_features(x, np.asarray(spec["filters"]))
        else:
            raise ValueError("Unknown model")
        features = (features - np.asarray(spec["mean"])) / np.asarray(spec["scale"])
        return features @ np.asarray(spec["coef"]) + spec["intercept"]

    def predict(self, x: np.ndarray, model: str | None = None) -> np.ndarray:
        name = model or self.config["selected"]
        return expit(self.logits(x, name) / self.config["models"][name]["temperature"])
