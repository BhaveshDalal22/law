"""Write a reviewable report from existing measurements; never load or score a model."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def report(directory, output):
    directory, output = directory.resolve(), output.resolve()
    artifact_dir = directory.parents[1]
    metadata = read(directory / "metadata.json")
    validation = read(directory / "validation.json")
    test = read(directory / "test.json")
    history = read(directory / "epoch_history.json")["epochs"]
    training = read(artifact_dir / "reports" / "training.json")
    baseline = read(artifact_dir / "reports" / f"test-{training['metadata']['model_version']}.json")
    cpu_pointer = read(artifact_dir / "experiments" / "latest.json")
    cpu = read(Path(cpu_pointer["experiment_dir"]) / "test.json")
    if test["evaluation_count"] != 1 or len(history) != metadata["epochs_completed"]:
        raise ValueError("Evaluation count or completed epoch history is inconsistent.")
    if not test["baseline_unchanged"]:
        raise ValueError("Original baseline integrity has not been verified.")
    if len({item["metrics"]["documents"] for item in (test, baseline, cpu)}) != 1:
        raise ValueError("Comparison reports have different held-out document counts.")
    import os

    def link(path):
        return Path(os.path.relpath(path, output.parent)).as_posix()

    lines = [
        "# Recorded GPU experiment results",
        "",
        (
            "Actual measurements from fine-tuning a pretrained DistilBERT encoder on the prepared LeSICiN IPC task. "
            "The original Logistic Regression and CPU SGD epoch models are retained."
        ),
        "",
        f"- Model version: `{metadata['model_version']}`.",
        f"- Device: {metadata['device']}; CUDA {metadata['cuda']}; precision {metadata['precision']}.",
        (
            f"- Completed **{metadata['epochs_completed']} epochs** of the configured {metadata['config']['epochs']}; "
            f"selected **epoch {metadata['best_epoch']}** using minimum validation loss "
            f"({metadata['best_validation_loss']:.6f})."
        ),
        (
            f"- Early-stopping patience: {metadata['config']['patience']}; "
            f"stopped early: {metadata['stopped_early']}."
        ),
        (
            f"- Pipeline wall time: {metadata['seconds']:.2f} seconds "
            "(includes preparation, epoch scoring and checkpoint restoration; excludes final test evaluation)."
        ),
        (
            f"- Peak PyTorch allocated GPU memory: {metadata['gpu_peak_allocated_mib']:.1f} MiB "
            "(does not include all driver/reserved memory)."
        ),
        (
            f"- Training: {metadata['training_documents']:,}; validation: {metadata['validation_documents']:,}; "
            f"held-out test: {test['metrics']['documents']:,}; supported IPC labels: {len(metadata['supported_labels'])}."
        ),
        (
            f"- Final global prediction threshold: **{test['threshold']}**, selected using validation data only "
            "after restoring the best checkpoint."
        ),
        "",
        "## Held-out measurements",
        "",
        (
            "The selected GPU model was evaluated once. The prior CPU results below are read from their saved reports; "
            "their held-out evaluations were not repeated."
        ),
        "",
        "| Metric | GPU transformer validation | GPU transformer test | CPU Logistic Regression test | CPU SGD test |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, label in [
        ("micro_precision", "Micro precision"),
        ("micro_recall", "Micro recall"),
        ("micro_f1", "Micro F1"),
        ("macro_f1", "Macro F1"),
    ]:
        values = [
            validation["metrics"][key],
            test["metrics"][key],
            baseline["metrics"][key],
            cpu["metrics"][key],
        ]
        lines.append(f"| {label} | " + " | ".join(f"{value:.6f}" for value in values) + " |")
    lines += [
        (
            f"| Mean binary log loss | {validation['binary_log_loss']:.6f} | {test['binary_log_loss']:.6f} | "
            f"Not recorded | {cpu['binary_log_loss']:.6f} |"
        ),
        (
            f"| Exact-match subset accuracy | {validation['subset_accuracy']:.6f} | {test['subset_accuracy']:.6f} | "
            f"Not recorded | {cpu['subset_accuracy']:.6f} |"
        ),
        "",
        (
            "Subset accuracy requires the entire predicted IPC label set to match the gold set. "
            "It is distinct from per-label accuracy. Loss is the mean unweighted binary cross-entropy over document-label pairs."
        ),
        "",
    ]
    for name, old in [("Logistic Regression", baseline), ("CPU SGD", cpu)]:
        delta = 100 * (test["metrics"]["micro_f1"] - old["metrics"]["micro_f1"])
        direction = "higher" if delta > 0 else "lower" if delta < 0 else "unchanged"
        lines.append(
            f"GPU test micro F1 is {abs(delta):.2f} percentage points {direction} "
            f"than {name}. This is a measured comparison, not a significance test."
        )
        lines.append("")
    lines += [
        "## Measured epoch history",
        "",
        (
            "Every epoch is one complete seeded shuffled pass over all eligible training records. "
            "The following F1 and subset accuracy values use threshold 0.5. Epoch graphs contain no test measurements; "
            "final threshold-selected test scores must not be confused with these comparison curves."
        ),
        "",
        "| Epoch | Training loss | Validation loss | Training micro F1 | Validation micro F1 | Validation subset accuracy |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in history:
        lines.append(
            f"| {row['epoch']} | {row['train_binary_log_loss']:.6f} | "
            f"{row['validation_binary_log_loss']:.6f} | {row['train_micro_f1']:.6f} | "
            f"{row['validation_micro_f1']:.6f} | {row['validation_subset_accuracy']:.6f} |"
        )
    lines += [
        "",
        f"![Training and validation curves]({link(directory / 'figures' / 'epoch_overview.png')})",
        "",
    ]
    for name, filename in [
        ("Loss graph", "figures/loss_vs_epoch.png"),
        ("Micro and macro F1 graph", "figures/f1_vs_epoch.png"),
        ("Exact-match accuracy graph", "figures/subset_accuracy_vs_epoch.png"),
        ("Epoch CSV", "epoch_metrics.csv"),
        ("Epoch JSON", "epoch_history.json"),
        ("Full test metrics and per-label results", "test.json"),
        ("Configuration and provenance", "metadata.json"),
    ]:
        lines.append(f"- [{name}]({link(directory / filename)})")
    lines += [
        "",
        "## Configuration, provenance and limits",
        "",
        (
            f"Base model: [{metadata['config']['model_id']}]({metadata['base_model_provenance']['url']}), "
            f"commit `{metadata['base_model_provenance']['revision']}`. The publisher declares Apache 2.0. "
            "The encoder was publicly pretrained; this project fine-tunes it and a newly initialized IPC classification head. "
            "LeSICiN data attribution and CC BY 4.0 provenance are recorded separately with original file hashes."
        ),
        "",
        (
            f"AdamW learning rate {metadata['config']['learning_rate']}, weight decay {metadata['config']['weight_decay']}, "
            f"batch size {metadata['config']['batch_size']}, accumulation {metadata['config']['accumulation_steps']}, "
            f"maximum sequence length {metadata['config']['max_length']}, seed {metadata['config']['seed']}. "
            "The base tokenizer vocabulary is frozen, and the task label encoder uses only training labels. "
            "The CPU TF-IDF models see longer narratives, so this is not a hardware-only comparison."
        ),
        "",
    ]
    for split, stats in {**metadata["tokenization"], "test": test["tokenization"]}.items():
        lines.append(
            f"- {split}: {stats['truncated_documents']:,}/{stats['documents']:,} documents "
            f"({100 * stats['truncated_documents'] / stats['documents']:.2f}%) "
            "were truncated to the sequence limit."
        )
    lines += [
        "",
        (
            "Label suggestions remain historical IPC labels. This experiment does not establish BNS support, "
            "current statutory validity or conversational question-answering capability. "
            "Rule-selected narratives and dataset labels may be noisy. The pretrained corpus was not independently audited "
            "for overlap with these cases. Validation is reused for checkpoint and threshold selection. "
            "No test result was used to choose epochs, thresholds or hyperparameters."
        ),
        "",
        (
            "Model and tokenizer files are stored with SHA-256 integrity checks. "
            "The original CPU artifacts were verified unchanged. The running application still serves its original CPU model."
        ),
        "",
        "## Commands",
        "",
        "```powershell",
        (
            ".\\.venv\\Scripts\\python.exe -m legal_intelligence.cli train-gpu --epochs 3 --patience 2 --seed 42 "
            f"--revision {metadata['base_model_provenance']['revision']}"
        ),
        ".\\.venv\\Scripts\\python.exe -m legal_intelligence.cli evaluate-gpu",
        ".\\.venv\\Scripts\\python.exe -m legal_intelligence.cli plot-gpu",
        "```",
        "",
        (
            "Training creates a new experiment version. Evaluation refuses a repeat for an already evaluated version. "
            "Plotting and this report script only read saved measurements."
        ),
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path, nargs="?")
    parser.add_argument("--output", type=Path, default=Path("GPU_RESULTS.md"))
    args = parser.parse_args()
    directory = args.experiment_dir or Path(
        read(Path("artifacts/gpu_experiments/latest.json"))["experiment_dir"]
    )
    report(directory, args.output)
