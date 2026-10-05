"""Fit on training people, select/calibrate on validation, evaluate frozen models once.

Training takes CPU minutes, not a paid GPU. See docs/PROTOCOL.md before changing it.
"""

import copy
import hashlib
import json
import platform
import random
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from intentlab.inference import Predictor  # noqa: E402
from intentlab.signal import CHANNELS, bandpower, csp_features, perturb  # noqa: E402

NAMES = {
    "majority": "Constant majority",
    "bandpower": "Bandpower + logistic regression",
    "csp_lda": "CSP + shrinkage LDA",
    "compact_eeg": "Compact EEG CNN",
}


class CompactEEG(nn.Module):
    """EEGNet-inspired temporal → depthwise spatial → separable temporal CNN.

    Odd temporal kernels, adaptive pooling and one binary logit are adaptations.
    Input: batch × 9 electrodes × 480 samples. No participant ID or cue is input.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, (1, 65), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (9, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.35),
            nn.Conv2d(16, 16, (1, 17), padding=(0, 8), groups=16, bias=False),
            nn.Conv2d(16, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.35),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        return self.head(self.features(x.unsqueeze(1))).flatten()


def macro_score(y, p, subjects):
    return float(
        np.mean(
            [balanced_accuracy_score(y[subjects == s], p[subjects == s] >= 0.5) for s in np.unique(subjects)]
        )
    )


def calibrate(logits, y):
    result = minimize_scalar(lambda t: log_loss(y, expit(logits / t)), bounds=(0.5, 5.0), method="bounded")
    return float(result.x)


def selective(y, p, threshold):
    retained = np.maximum(p, 1 - p) >= threshold
    return {
        "threshold": float(threshold),
        "coverage": float(retained.mean()),
        "retained": int(retained.sum()),
        "accuracy": float(accuracy_score(y[retained], p[retained] >= 0.5)) if retained.any() else None,
    }


def metrics(y, p, subjects):
    people = [
        {
            "subject": int(s),
            "trials": int((subjects == s).sum()),
            "balanced_accuracy": float(balanced_accuracy_score(y[subjects == s], p[subjects == s] >= 0.5)),
        }
        for s in np.unique(subjects)
    ]
    scores = np.array([r["balanced_accuracy"] for r in people])
    rng = np.random.default_rng(2027)
    ci = np.quantile(rng.choice(scores, (2000, len(scores)), replace=True).mean(1), [0.025, 0.975]).tolist()
    reliability = []
    confidence = np.maximum(p, 1 - p)
    correct = (p >= 0.5) == y
    for lo in np.linspace(0, 0.9, 10):
        hi = lo + 0.1
        mask = (confidence >= lo) & (confidence < hi if hi < 0.999 else confidence <= 1)
        if mask.any():
            reliability.append(
                {
                    "confidence": float(confidence[mask].mean()),
                    "accuracy": float(correct[mask].mean()),
                    "count": int(mask.sum()),
                }
            )
    ece = sum(abs(r["confidence"] - r["accuracy"]) * r["count"] / len(y) for r in reliability)
    return {
        "participant_balanced_accuracy": float(scores.mean()),
        "bootstrap_95_ci": ci,
        "balanced_accuracy": float(balanced_accuracy_score(y, p >= 0.5)),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(ece),
        "n": len(y),
        "confusion_matrix": confusion_matrix(y, p >= 0.5, labels=[0, 1]).tolist(),
        "participants": people,
        "reliability": reliability,
        "selective_curve": [selective(y, p, t) for t in np.arange(0.5, 1, 0.025)],
    }


def main():
    output = ROOT / "artifacts"
    if (output / "evaluation.json").exists() and "--overwrite" not in sys.argv:
        raise SystemExit(
            "An evaluation already exists. Use a new experiment directory or explicitly pass --overwrite for a reproducibility run; do not tune against the test set."
        )
    torch.set_num_threads(4)
    torch.manual_seed(2027)
    torch.use_deterministic_algorithms(True)
    np.random.seed(2027)
    random.seed(2027)
    archive = np.load(ROOT / "data/derived/epochs.npz", allow_pickle=False)
    x, y, subjects, split = [archive[key] for key in ["x", "y", "subjects", "split"]]
    tr, va, te = [split == s for s in ["train", "validation", "test"]]
    for a, b in [(tr, va), (tr, te), (va, te)]:
        assert not set(subjects[a]) & set(subjects[b])
    spec = {"version": "1.0.0", "seed": 2027, "models": {}}
    candidates = spec["models"]
    # The baseline always predicts the training majority; its probability is prevalence.
    prevalence = float(y[tr].mean())
    candidates["majority"] = {
        "name": NAMES["majority"],
        "logit": float(np.log(prevalence / (1 - prevalence))),
        "temperature": 1.0,
    }
    valid_logits = {"majority": np.full(va.sum(), candidates["majority"]["logit"])}
    feature_arrays = {"bandpower": bandpower(x)}
    covs = np.einsum("nct,ndt->ncd", x[tr], x[tr]) / x.shape[-1]
    covs /= np.trace(covs, axis1=1, axis2=2)[:, None, None]
    class_cov = [covs[y[tr] == c].mean(0) for c in [0, 1]]
    class_cov = [0.9 * c + 0.1 * np.trace(c) / len(CHANNELS) * np.eye(len(CHANNELS)) for c in class_cov]
    _, vectors = eigh(class_cov[1], class_cov[0] + class_cov[1])
    filters = vectors[:, [0, 1, -2, -1]].T
    feature_arrays["csp_lda"] = csp_features(x, filters)
    for key, features in feature_arrays.items():
        scaler = StandardScaler().fit(features[tr])
        z = scaler.transform(features)
        model = (
            LogisticRegression(C=1.0, max_iter=1000, random_state=2027)
            if key == "bandpower"
            else LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        )
        model.fit(z[tr], y[tr])
        candidates[key] = {
            "name": NAMES[key],
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coef": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
            "temperature": 1.0,
        }
        if key == "csp_lda":
            candidates[key]["filters"] = filters.tolist()
        valid_logits[key] = model.decision_function(z[va])
        print(
            f"{key}: validation participant balanced accuracy {macro_score(y[va], expit(valid_logits[key]), subjects[va]):.4f}",
            flush=True,
        )

    network = CompactEEG()
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=0.01)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x[tr]), torch.tensor(y[tr], dtype=torch.float32)),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(2027),
    )
    validation_tensor = torch.from_numpy(x[va])
    best_score, best_epoch, stale, history, best_state = -1, 0, 0, [], None
    for epoch in range(1, 41):
        network.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(network(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        network.eval()
        with torch.no_grad():
            logits = torch.cat([network(batch) for batch in validation_tensor.split(128)]).numpy()
        score = macro_score(y[va], expit(logits), subjects[va])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_participant_balanced_accuracy": score,
            }
        )
        print(f"CNN epoch {epoch:02d}: loss {np.mean(losses):.4f}; validation {score:.4f}", flush=True)
        if score > best_score:
            best_score, best_epoch, stale, best_state = score, epoch, 0, copy.deepcopy(network.state_dict())
        else:
            stale += 1
        if stale >= 8:
            break
    network.load_state_dict(best_state)
    network.eval()
    with torch.no_grad():
        valid_logits["compact_eeg"] = torch.cat(
            [network(batch) for batch in validation_tensor.split(128)]
        ).numpy()
    candidates["compact_eeg"] = {
        "name": NAMES["compact_eeg"],
        "temperature": 1.0,
        "parameters": sum(p.numel() for p in network.parameters()),
        "best_epoch": best_epoch,
    }
    torch.onnx.export(
        network,
        torch.from_numpy(x[:1]),
        str(output / "compact_eeg.onnx"),
        input_names=["eeg"],
        output_names=["logit"],
        dynamic_axes={"eeg": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    session = ort.InferenceSession(str(output / "compact_eeg.onnx"), providers=["CPUExecutionProvider"])
    exported_logits = session.run(None, {"eeg": x[va][:16]})[0]
    parity_error = float(np.max(np.abs(exported_logits - valid_logits["compact_eeg"][:16])))
    assert parity_error < 1e-4, parity_error
    validation_scores = {}
    for name, logits in valid_logits.items():
        if name != "majority":
            candidates[name]["temperature"] = calibrate(logits, y[va])
        validation_scores[name] = macro_score(y[va], expit(logits), subjects[va])
    spec["selected"] = max(["bandpower", "csp_lda", "compact_eeg"], key=lambda k: validation_scores[k])
    chosen = spec["selected"]
    vp = expit(valid_logits[chosen] / candidates[chosen]["temperature"])
    eligible = [selective(y[va], vp, t) for t in np.arange(0.5, 0.951, 0.05)]
    qualifying = [
        r for r in eligible if r["retained"] >= 100 and r["coverage"] >= 0.2 and r["accuracy"] >= 0.75
    ]
    threshold = qualifying[0]["threshold"] if qualifying else 0.75
    spec["threshold"] = round(threshold, 2)
    spec["threshold_target_met"] = bool(qualifying)
    (output / "models.json").write_text(json.dumps(spec, indent=2) + "\n")
    predictor = Predictor(output)
    # Only now evaluate the fixed models on held-out people.
    report = {
        "protocol": "docs/PROTOCOL.md",
        "seed": 2027,
        "selected": chosen,
        "selection_metric": "validation participant-macro balanced accuracy",
        "threshold": spec["threshold"],
        "threshold_target_met": bool(qualifying),
        "validation_threshold": selective(y[va], vp, threshold),
        "onnx_max_absolute_logit_error": parity_error,
        "training_history": history,
        "models": {},
    }
    for name in candidates:
        p = predictor.predict(x[te], name)
        report["models"][name] = {
            "name": NAMES[name],
            "validation_participant_balanced_accuracy": validation_scores[name],
            "test": metrics(y[te], p, subjects[te]),
        }
        print(
            f"{name}: TEST participant balanced accuracy {report['models'][name]['test']['participant_balanced_accuracy']:.4f}",
            flush=True,
        )
    clean = predictor.predict(x[te])
    report["operating_point"] = selective(y[te], clean, threshold)
    report["robustness"] = []
    for label, noise, drop in [
        ("Clean", 0, None),
        ("Noise × 0.5 RMS", 0.5, None),
        ("Noise × 1.0 RMS", 1.0, None),
        ("C3 disconnected", 0, "C3"),
        ("C4 disconnected", 0, "C4"),
    ]:
        p = predictor.predict(perturb(x[te], noise, drop))
        report["robustness"].append(
            {
                "condition": label,
                "participant_balanced_accuracy": macro_score(y[te], p, subjects[te]),
                **selective(y[te], p, threshold),
            }
        )
    report["channel_occlusion"] = []
    base_score = macro_score(y[te], clean, subjects[te])
    for channel in CHANNELS:
        p = predictor.predict(perturb(x[te], drop=channel))
        report["channel_occlusion"].append(
            {
                "channel": channel,
                "balanced_accuracy_drop": base_score - macro_score(y[te], p, subjects[te]),
                "mean_absolute_probability_change": float(np.abs(p - clean).mean()),
            }
        )
    report["environment"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "device": "cpu",
        "threads": 4,
    }
    report["data_manifest_sha256"] = hashlib.sha256(
        (ROOT / "data/derived/manifest.json").read_bytes()
    ).hexdigest()
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    checksums = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [output / "models.json", output / "compact_eeg.onnx", output / "evaluation.json"]
    }
    (output / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
    print(
        f"Selected {chosen}; abstention threshold {threshold:.2f}; validation target met={bool(qualifying)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
