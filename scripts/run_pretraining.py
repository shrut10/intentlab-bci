"""Run the fixed, validation-only masked-channel pretraining feasibility study.

Requires requirements-training.txt. No downloads and no test-cohort scoring.
python scripts/run_pretraining.py
"""

import copy
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from intentlab.pretraining import (
    development_rows,
    labelled_subset,
    mask_channels,
    participant_metrics,
    summarise_pairs,
)

CONFIG = ROOT / "experiments/pretraining-v1/protocol.json"
OUTPUT = ROOT / "artifacts/pretraining-v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Encoder(nn.Sequential):
    def __init__(self, config):
        width, kernel = config["encoder_channels"], config["encoder_kernel"]
        super().__init__(
            nn.Conv1d(9, width, kernel, padding=kernel // 2),
            nn.ELU(),
            nn.Conv1d(width, width, kernel, padding=kernel // 2),
            nn.ELU(),
        )


class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.head = nn.Linear(config["encoder_channels"], 1)

    def forward(self, x):
        return self.head(self.encoder(x).mean(dim=-1)).flatten()


def masked_loss(predicted, target, mask):
    if predicted.shape != target.shape or mask.shape != target.shape[:2]:
        raise ValueError("Reconstruction and mask dimensions must match the target")
    if not mask.any():
        raise ValueError("At least one masked channel is required")
    return ((predicted - target).square() * mask[:, :, None]).sum() / (mask.sum() * target.shape[-1])


def optimiser(model, config):
    return torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )


def pretrain(initial_encoder, source_x, config, seed):
    """There is intentionally no labels, validation or test argument."""
    torch.manual_seed(seed + 10)
    encoder = copy.deepcopy(initial_encoder)
    model = nn.Sequential(encoder, nn.Conv1d(config["encoder_channels"], 9, 1))
    opt = optimiser(model, config)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(source_x)),
        batch_size=config["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 11),
    )
    mask_rng = np.random.default_rng(seed + 12)
    history = []
    for epoch in range(1, config["pretraining_epochs"] + 1):
        model.train()
        total_loss, count = 0.0, 0
        for (target,) in loader:
            masked, mask = mask_channels(target.numpy(), mask_rng, config["masked_channels"])
            opt.zero_grad()
            loss = masked_loss(model(torch.from_numpy(masked)), target, torch.from_numpy(mask))
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(target)
            count += len(target)
        history.append({"epoch": epoch, "masked_channel_mse": total_loss / count})
        if epoch % 5 == 0:
            print(f"seed {seed}: pretraining epoch {epoch}, masked MSE {total_loss / count:.5f}", flush=True)
    return encoder.state_dict(), history


def fit_classifier(initial, encoder_state, source_x, source_y, positions, config, seed):
    model = copy.deepcopy(initial)
    model.encoder.load_state_dict(encoder_state)
    torch.manual_seed(seed + 100)
    opt = optimiser(model, config)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(source_x[positions]), torch.tensor(source_y[positions], dtype=torch.float32)
        ),
        batch_size=config["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 101),
    )
    history = []
    for epoch in range(1, config["supervised_epochs"] + 1):
        model.train()
        total_loss, count = 0.0, 0
        for x, y in loader:
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(x)
            count += len(x)
        history.append({"epoch": epoch, "binary_cross_entropy": total_loss / count})
    model.eval()
    return model, history


def main():
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit("Study output exists; use a new versioned experiment for changed code or a new run")
    config = json.loads(CONFIG.read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = [
        "experiments/pretraining-v1/protocol.json",
        "experiments/pretraining-v1/PROTOCOL.md",
        "scripts/run_pretraining.py",
        "src/intentlab/pretraining.py",
        "data/derived/epochs.npz",
        "data/derived/trials.parquet",
        "data/derived/manifest.json",
    ]
    started = time.perf_counter()
    run = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision_at_start": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "input_and_code_sha256": {p: digest(ROOT / p) for p in paths},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": "cpu",
        },
        "production_artifacts_sha256": {
            p: digest(ROOT / p)
            for p in ("artifacts/models.json", "artifacts/evaluation.json", "artifacts/compact_eeg.onnx")
        },
    }
    # Persist protocol and code fingerprints before the first model is initialised.
    save(OUTPUT / "run.json", run)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    archive = np.load(ROOT / "data/derived/epochs.npz", allow_pickle=False)
    train, valid = development_rows(archive["subjects"], archive["split"])
    source_x, source_y, source_subjects = (archive[key][train] for key in ("x", "y", "subjects"))
    valid_x, valid_y, valid_subjects = (archive[key][valid] for key in ("x", "y", "subjects"))
    rows = pq.read_table(ROOT / "data/derived/trials.parquet").to_pylist()
    if len(rows) != len(archive["split"]) or any(
        (r["subject"], r["label"], r["split"]) != (int(s), int(y), split)
        for r, s, y, split in zip(rows, archive["subjects"], archive["y"], archive["split"], strict=True)
    ):
        raise ValueError("Trial metadata and epoch archive do not match")
    manifest = {
        "pretraining_trial_ids": [rows[i]["id"] for i in train],
        "validation_trial_ids": [rows[i]["id"] for i in valid],
        "pretraining_participants": sorted(np.unique(source_subjects).tolist()),
        "validation_participants": sorted(np.unique(valid_subjects).tolist()),
        "labelled_subsets": [],
    }
    for seed in config["seeds"]:
        for fraction in config["label_fractions"]:
            chosen = labelled_subset(source_y, source_subjects, fraction, seed)
            manifest["labelled_subsets"].append(
                {
                    "seed": seed,
                    "label_fraction": fraction,
                    "trial_ids": [rows[train[i]]["id"] for i in chosen],
                }
            )
    save(OUTPUT / "split_manifest.json", manifest)
    results, predictions, pretraining_histories = [], [], []
    for seed in config["seeds"]:
        torch.manual_seed(seed)
        initial = Classifier(config)
        initial_state = copy.deepcopy(initial.encoder.state_dict())
        pretrained_state, history = pretrain(initial.encoder, source_x, config, seed)
        pretraining_histories.append({"seed": seed, "history": history})
        for fraction in config["label_fractions"]:
            chosen = labelled_subset(source_y, source_subjects, fraction, seed)
            for arm, state in (("scratch", initial_state), ("masked_pretraining", pretrained_state)):
                model, history = fit_classifier(initial, state, source_x, source_y, chosen, config, seed)
                with torch.no_grad():
                    probabilities = torch.cat(
                        [model(torch.from_numpy(x)).sigmoid() for x in np.array_split(valid_x, 8)]
                    ).numpy()
                result = {
                    "seed": seed,
                    "arm": arm,
                    "label_fraction": fraction,
                    "labelled_trials": len(chosen),
                    "realised_label_fraction": len(chosen) / len(source_y),
                    "classifier_parameters": sum(p.numel() for p in model.parameters()),
                    "validation": participant_metrics(valid_y, probabilities, valid_subjects),
                    "supervised_history": history,
                }
                results.append(result)
                for position, label, probability, subject in zip(
                    valid, valid_y, probabilities, valid_subjects, strict=True
                ):
                    predictions.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "label_fraction": fraction,
                            "trial_id": rows[position]["id"],
                            "subject": int(subject),
                            "label": int(label),
                            "probability": float(probability),
                            "split": "validation",
                        }
                    )
                save(
                    OUTPUT / "progress.json",
                    {"results": results, "pretraining_histories": pretraining_histories},
                )
                print(
                    f"seed {seed} / labels {fraction:.0%} / {arm}: validation BA {result['validation']['participant_balanced_accuracy']:.5f}",
                    flush=True,
                )
    for path, expected in run["production_artifacts_sha256"].items():
        if digest(ROOT / path) != expected:
            raise RuntimeError("Production artifact changed during this isolated experiment")
    report = {
        "experiment": config["experiment"],
        "status": config["status"],
        "production_model_changed": False,
        "test_evaluated": False,
        "counts": {
            "training_participants": len(np.unique(source_subjects)),
            "training_trials": len(source_y),
            "training_epoch_hours": len(source_y) * source_x.shape[-1] / 160 / 3600,
            "validation_participants": len(np.unique(valid_subjects)),
            "validation_trials": len(valid_y),
        },
        "protocol": config,
        "run": run,
        "results": results,
        "pretraining_histories": pretraining_histories,
        "summary": summarise_pairs(results, config["bootstrap_samples"], config["bootstrap_seed"]),
        "elapsed_seconds": time.perf_counter() - started,
    }
    save(OUTPUT / "evaluation.json", report)
    with (OUTPUT / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    save(
        OUTPUT / "checksums.json",
        {
            name: digest(OUTPUT / name)
            for name in ("run.json", "split_manifest.json", "evaluation.json", "predictions.csv")
        },
    )
    (OUTPUT / "progress.json").unlink()
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(
        f"Completed in {report['elapsed_seconds']:.1f}s; original test cohort was not evaluated", flush=True
    )


if __name__ == "__main__":
    main()
