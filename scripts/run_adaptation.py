"""Run the frozen issue-1 comparison without changing production artefacts.

python scripts/run_adaptation.py
python scripts/run_adaptation.py --resume  # Interrupted run with identical inputs/code only.
"""

import argparse
import copy
import hashlib
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from intentlab.adaptation import (
    CONDITIONS,
    apply_alignment,
    binary_metrics,
    fit_alignment,
    fit_temperature,
    paired_interval,
    partition_rows,
    probabilities,
    select_threshold,
    selective_metrics,
)
from scripts.train import CompactEEG

CONFIG_PATH = ROOT / "experiments/adaptation-v1/protocol.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def onnx_logits(path, x):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    return np.concatenate(
        [session.run(None, {"eeg": batch})[0].ravel() for batch in np.array_split(x, max(1, len(x) // 128))]
    )


def fit_network(x, y, subjects, train, valid, seed, kind, output, config, resume):
    name = f"{kind}-{seed}"
    model_file = output / f"{name}.onnx"
    metadata_file = output / f"{name}.json"
    if resume and metadata_file.exists():
        metadata = json.loads(metadata_file.read_text())
        if not model_file.exists() or digest(model_file) != metadata["onnx_sha256"]:
            raise RuntimeError(f"Corrupt resumable checkpoint: {name}")
        print(f"Reusing completed checkpoint {name}", flush=True)
        return metadata
    training = config["training"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    network = CompactEEG()
    optimiser = torch.optim.AdamW(
        network.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x[train]), torch.tensor(y[train], dtype=torch.float32)),
        batch_size=training["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation = torch.from_numpy(x[valid])
    best, best_epoch, stale, best_state, history = -1, 0, 0, None, []
    for epoch in range(1, training["epochs"] + 1):
        network.train()
        losses = []
        for xb, yb in loader:
            optimiser.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(network(xb), yb)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.item()))
        network.eval()
        with torch.no_grad():
            logits = torch.cat([network(batch) for batch in validation.split(128)]).numpy()
        score = binary_metrics(y[valid], probabilities(logits, 1), subjects[valid])[
            "participant_balanced_accuracy"
        ]
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_participant_balanced_accuracy": score,
            }
        )
        print(f"{name}: epoch {epoch:02d}, validation {score:.4f}", flush=True)
        if score > best:
            best, best_epoch, stale, best_state = score, epoch, 0, copy.deepcopy(network.state_dict())
        else:
            stale += 1
        if stale >= training["patience"]:
            break
    network.load_state_dict(best_state)
    network.eval()
    with torch.no_grad():
        logits = torch.cat([network(batch) for batch in validation.split(128)]).numpy()
    global_temperature = fit_temperature(logits, y[valid], bounds=config["temperature_bounds"])
    torch.onnx.export(
        network,
        torch.from_numpy(x[train][:1]),
        model_file,
        input_names=["eeg"],
        output_names=["logit"],
        dynamic_axes={"eeg": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    parity = float(np.max(np.abs(onnx_logits(model_file, x[valid][:16]) - logits[:16])))
    if parity >= 1e-4:
        raise RuntimeError(f"ONNX parity failed for {name}: {parity}")
    metadata = {
        "name": name,
        "kind": kind,
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_participant_balanced_accuracy": best,
        "global_temperature": global_temperature,
        "history": history,
        "onnx_max_absolute_logit_error": parity,
        "onnx_sha256": digest(model_file),
    }
    save(metadata_file, metadata)
    return metadata


def summarise(results, config):
    summary = {}
    baseline = [r for r in results if r["condition"] == "baseline"]
    for condition, name in CONDITIONS.items():
        records = [r for r in results if r["condition"] == condition]
        assert [r["seed"] for r in records] == config["seeds"]
        comparison = {}
        for metric in ("balanced_accuracy", "brier"):
            base = [[p[metric] for p in r["test"]["metrics"]["participants"]] for r in baseline]
            candidate = [[p[metric] for p in r["test"]["metrics"]["participants"]] for r in records]
            comparison[metric] = paired_interval(
                base, candidate, config["bootstrap_samples"], config["bootstrap_seed"]
            )
        operating = {}
        for rule in ("fixed_threshold", "validation_threshold"):
            points = [r["test"][rule] for r in records]
            retained, correct = sum(p["retained"] for p in points), sum(p["correct"] for p in points)
            operating[rule] = {
                "mean_coverage": float(np.mean([p["coverage"] for p in points])),
                "mean_retained_per_seed": float(np.mean([p["retained"] for p in points])),
                "pooled_repeated_prediction_accuracy": correct / retained if retained else None,
                "total_repeated_predictions": sum(p["n"] for p in points),
                "retained_repeated_predictions": retained,
                "correct_repeated_predictions": correct,
            }
        summary[condition] = {
            "name": name,
            **{
                f"mean_{key}": float(np.mean([r["test"]["metrics"][key] for r in records]))
                for key in ("participant_balanced_accuracy", "brier", "ece")
            },
            "seed_balanced_accuracies": [
                r["test"]["metrics"]["participant_balanced_accuracy"] for r in records
            ],
            "paired_change_vs_baseline": comparison,
            "operating_points": operating,
            "validation_target_met_seeds": sum(
                r["validation"]["threshold_rule"]["target_met"] for r in records
            ),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text())
    output = ROOT / "artifacts/adaptation-v1"
    protocol = ROOT / "experiments/adaptation-v1/PROTOCOL.md"
    sources = [
        CONFIG_PATH,
        protocol,
        Path(__file__).resolve(),
        ROOT / "src/intentlab/adaptation.py",
        ROOT / "scripts/train.py",
        ROOT / "data/derived/epochs.npz",
        ROOT / "data/derived/trials.parquet",
        ROOT / "data/derived/manifest.json",
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    if (output / "evaluation.json").exists():
        raise SystemExit("Completed evaluation exists; do not overwrite or tune against this result")
    if output.exists():
        if not args.resume or not (output / "run.json").exists():
            raise SystemExit("Output exists; --resume requires an identical recorded run")
        run = json.loads((output / "run.json").read_text())
        if hashes != run["input_and_code_sha256"]:
            raise SystemExit("Resume refused: inputs, implementation or protocol changed")
    else:
        output.mkdir(parents=True)
        run = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "code_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "input_and_code_sha256": hashes,
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "onnxruntime": ort.__version__,
                "device": "cpu",
            },
        }
        save(output / "run.json", run)
    torch.set_num_threads(config["training"]["threads"])
    torch.use_deterministic_algorithms(True)
    archive = np.load(ROOT / "data/derived/epochs.npz", allow_pickle=False)
    x, y, subjects, splits = [archive[k] for k in ("x", "y", "subjects", "split")]
    rows = pq.read_table(ROOT / "data/derived/trials.parquet").to_pylist()
    for key, values in (("subject", subjects), ("label", y), ("split", splits)):
        if [r[key] for r in rows] != values.tolist():
            raise RuntimeError("Epoch/metadata row order differs")
    early, later = partition_rows(
        rows, config["calibration_trials"], config["calibration_run"], config["evaluation_runs"]
    )
    matrices = {
        person: fit_alignment(x[indices], config["alignment_shrinkage"], config["eigenvalue_floor"])
        for person, indices in early.items()
    }
    aligned = np.empty_like(x)
    for person, matrix in matrices.items():
        aligned[subjects == person] = apply_alignment(x[subjects == person], matrix)
    audit = []
    for person in sorted(early):
        if set(y[later[person]]) != {0, 1}:
            raise RuntimeError(f"Both classes required for participant {person}")
        audit.append(
            {
                "subject": person,
                "split": rows[int(early[person][0])]["split"],
                "calibration_ids": [rows[i]["id"] for i in early[person]],
                "evaluation_ids": [rows[i]["id"] for i in later[person]],
                "alignment_matrix": matrices[person].tolist(),
            }
        )
    save(output / "calibration_manifest.json", audit)
    train = np.flatnonzero(splits == "train")
    evaluation = {
        split: np.concatenate([later[p] for p in sorted(later) if rows[int(later[p][0])]["split"] == split])
        for split in ("validation", "test")
    }
    counts = {
        split: {
            "participants": len(np.unique(subjects[splits == split])),
            "all_trials": int((splits == split).sum()),
            "calibration_trials": sum(
                len(indices) for p, indices in early.items() if rows[int(indices[0])]["split"] == split
            ),
            "evaluation_trials": sum(
                len(indices) for p, indices in later.items() if rows[int(indices[0])]["split"] == split
            ),
        }
        for split in ("train", "validation", "test")
    }
    print(json.dumps({"design_counts": counts}), flush=True)
    models = []
    # Finish every training/selection decision before computing any test score.
    for seed in config["seeds"]:
        for kind, values in (("unaligned", x), ("aligned", aligned)):
            models.append(
                fit_network(
                    values,
                    y,
                    subjects,
                    train,
                    evaluation["validation"],
                    seed,
                    kind,
                    output,
                    config,
                    args.resume,
                )
            )
    print("All six models frozen; evaluating the predeclared four conditions", flush=True)
    results, predictions, calibration_fits = [], [], []
    for seed in config["seeds"]:
        for kind, values in (("unaligned", x), ("aligned", aligned)):
            metadata = next(m for m in models if m["seed"] == seed and m["kind"] == kind)
            global_t = metadata["global_temperature"]["temperature"]
            logits = {}
            temperatures = {}
            for split in ("validation", "test"):
                indices = evaluation[split]
                logits[split] = onnx_logits(output / f"{kind}-{seed}.onnx", values[indices])
                temperatures[split] = np.full(len(indices), global_t)
                for person in np.unique(subjects[indices]):
                    initial_logits = onnx_logits(output / f"{kind}-{seed}.onnx", values[early[person]])
                    fit = fit_temperature(
                        initial_logits, y[early[person]], global_t, config["temperature_bounds"]
                    )
                    temperatures[split][subjects[indices] == person] = fit["temperature"]
                    calibration_fits.append(
                        {
                            "seed": seed,
                            "kind": kind,
                            "split": split,
                            "subject": int(person),
                            "labelled_trials": len(early[person]),
                            **fit,
                        }
                    )
            names = (
                ("baseline", "personal_temperature")
                if kind == "unaligned"
                else ("alignment", "alignment_temperature")
            )
            for personal, condition in enumerate(names):
                ps = {
                    split: probabilities(logits[split], temperatures[split] if personal else global_t)
                    for split in ("validation", "test")
                }
                selection = select_threshold(y[evaluation["validation"]], ps["validation"], config)
                result = {
                    "seed": seed,
                    "condition": condition,
                    "name": CONDITIONS[condition],
                    "model": metadata["name"],
                }
                for split in ("validation", "test"):
                    indices, p = evaluation[split], ps[split]
                    result[split] = {
                        "metrics": binary_metrics(y[indices], p, subjects[indices]),
                        "fixed_threshold": selective_metrics(y[indices], p, config["fallback_threshold"]),
                        "validation_threshold": selective_metrics(y[indices], p, selection["threshold"]),
                        "selective_curve": [
                            selective_metrics(y[indices], p, t) for t in config["thresholds"]
                        ],
                    }
                    if split == "validation":
                        result[split]["threshold_rule"] = selection
                    for i, row in enumerate(indices):
                        predictions.append(
                            {
                                "seed": seed,
                                "condition": condition,
                                "split": split,
                                "subject": int(subjects[row]),
                                "trial_id": rows[row]["id"],
                                "run": rows[row]["run"],
                                "label": int(y[row]),
                                "logit": float(logits[split][i]),
                                "temperature": float(temperatures[split][i]) if personal else global_t,
                                "probability": float(p[i]),
                            }
                        )
                results.append(result)
                print(
                    f"{condition} {seed}: test BA {result['test']['metrics']['participant_balanced_accuracy']:.4f}, coverage {result['test']['fixed_threshold']['coverage']:.4f}",
                    flush=True,
                )
    # Match the declared seed order for the paired summary.
    results.sort(key=lambda r: (config["seeds"].index(r["seed"]), list(CONDITIONS).index(r["condition"])))
    pq.write_table(pa.Table.from_pylist(predictions), output / "predictions.parquet", compression="zstd")
    save(output / "personal_temperatures.json", calibration_fits)
    report = {
        "experiment": config["experiment"],
        "version": config["version"],
        "issue": config["issue"],
        "status": "exploratory_existing_benchmark",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "production_model_changed": False,
        "protocol": "experiments/adaptation-v1/PROTOCOL.md",
        "config": config,
        "run": run,
        "counts": counts,
        "models": models,
        "results": results,
        "summary": summarise(results, config),
        "personal_temperature_audit": {
            split: {
                kind: {
                    "fits": sum(r["split"] == split and r["kind"] == kind for r in calibration_fits),
                    "at_bound": sum(
                        r["at_bound"] for r in calibration_fits if r["split"] == split and r["kind"] == kind
                    ),
                    "fallbacks": sum(
                        r["fallback"] is not None
                        for r in calibration_fits
                        if r["split"] == split and r["kind"] == kind
                    ),
                }
                for kind in ("unaligned", "aligned")
            }
            for split in ("validation", "test")
        },
    }
    save(output / "evaluation.json", report)
    save(
        output / "checksums.json",
        {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "checksums.json"},
    )
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
