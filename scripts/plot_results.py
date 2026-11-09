"""Plot saved evaluation reports without retraining or re-evaluating the held-out test set."""

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from legal_intelligence.config import Settings

BLUE = "#2463A0"
TEAL = "#168174"
ORANGE = "#D46B2B"
INK = "#172B43"


def compact_label(label):
    return label.replace("Section ", "IPC ").replace(" in The Indian Penal Code", "")


def section_order(item):
    match = re.search(r"Section (\d+)(.*?) in", item[0])
    return (int(match[1]), match[2]) if match else (999999, item[0])


def style_axis(axis, horizontal=False):
    axis.set_axisbelow(True)
    axis.grid(axis="x" if horizontal else "y", color="#DFE5EB", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#CCD5DF")
    axis.tick_params(colors=INK, labelsize=10)
    (axis.set_xlim if horizontal else axis.set_ylim)(0, 1)


def save_figure(figure, output_dir, name):
    paths = []
    for extension in ("png", "svg"):
        path = output_dir / f"{name}.{extension}"
        figure.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
        paths.append(str(path))
    plt.close(figure)
    return paths


def plot_reports(reports_dir, output_dir):
    training_path = reports_dir / "training.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    metadata = training["metadata"]
    version = metadata["model_version"]
    test_path = reports_dir / f"test-{version}.json"
    evaluation = json.loads(test_path.read_text(encoding="utf-8"))
    if evaluation["model_version"] != version or evaluation["split"] != "test":
        raise ValueError("Training and test report versions/splits do not match.")
    validation, test = training["validation"], evaluation["metrics"]
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))
    figure.subplots_adjust(left=0.08, right=0.97, bottom=0.09, top=0.82, hspace=0.42, wspace=0.39)
    figure.suptitle(
        "AI-Based Legal Case Intelligence\nRecorded model evaluation",
        x=0.08,
        y=0.98,
        ha="left",
        fontsize=22,
        fontweight="bold",
        color=INK,
    )
    selected = metadata["selected_configuration"]
    figure.text(
        0.08,
        0.89,
        f"{version}  |  {metadata['training_documents']:,} training cases  |  "
        f"{len(metadata['supported_labels'])} IPC labels  |  selected C={selected['C']:g}, "
        f"threshold={selected['threshold']:g}",
        fontsize=11,
    )

    axis = axes[0, 0]
    names = ["Micro\nprecision", "Micro\nrecall", "Micro F1", "Macro F1"]
    keys = ["micro_precision", "micro_recall", "micro_f1", "macro_f1"]
    positions = np.arange(len(keys))
    for shift, metrics, name, color in [
        (-0.18, validation, "Validation", BLUE),
        (0.18, test, "Held-out test", TEAL),
    ]:
        bars = axis.bar(
            positions + shift,
            [metrics[key] for key in keys],
            width=0.34,
            label=f"{name} (n={metrics['documents']:,})",
            color=color,
        )
        axis.bar_label(bars, fmt="%.3f", padding=4, fontsize=10)
    axis.set_xticks(positions, names)
    axis.set_ylabel("Score (0 to 1)")
    axis.set_title("Validation and held-out performance", loc="left", pad=14)
    axis.legend(loc="upper right", frameon=False, fontsize=9)
    style_axis(axis)

    axis = axes[0, 1]
    groups = defaultdict(list)
    for run in training["validation_search"]:
        groups[(run["analyzer"], run["C"])].append(run)
    colors = ["#8798AB", BLUE, TEAL, ORANGE]
    for i, ((analyzer, c), runs) in enumerate(sorted(groups.items())):
        runs.sort(key=lambda run: run["threshold"])
        axis.plot(
            [run["threshold"] for run in runs],
            [run["validation_supported_micro_f1"] for run in runs],
            marker="o",
            linewidth=2,
            color=colors[i % len(colors)],
            label=f"{analyzer}, C={c:g}",
        )
    axis.scatter(
        [selected["threshold"]],
        [selected["validation_supported_micro_f1"]],
        s=100,
        color=ORANGE,
        edgecolor="white",
        zorder=4,
        label="Selected on validation",
    )
    axis.set_xticks(sorted({run["threshold"] for run in training["validation_search"]}))
    axis.set_xlabel("Prediction threshold")
    axis.set_ylabel("Validation micro F1")
    axis.set_title("Regularization and threshold comparison", loc="left", pad=14)
    axis.legend(loc="upper right", frameon=False, fontsize=9)
    style_axis(axis)

    axis = axes[1, 0]
    ks = sorted(test["top_k"], key=int)
    for key, name, color in [
        ("precision_at_k", "Precision@k", BLUE),
        ("recall_at_k", "Recall@k", ORANGE),
        ("hit_rate_at_k", "Hit rate@k", TEAL),
    ]:
        values = [test["top_k"][k][key] for k in ks]
        axis.plot([int(k) for k in ks], values, marker="o", linewidth=2, color=color, label=name)
        for k, value in zip(ks, values):
            axis.annotate(
                f"{value:.3f}",
                (int(k), value),
                xytext=(0, -16 if key == "recall_at_k" or (key == "precision_at_k" and k == "1") else 9),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=color,
            )
    axis.set_xticks([int(k) for k in ks])
    axis.set_xlabel("Top k ranked labels (independent of thresholds)")
    axis.set_ylabel("Document-mean score")
    axis.set_title("Held-out ranking metrics", loc="left", pad=14)
    axis.legend(loc="lower right", frameon=False, fontsize=9)
    style_axis(axis)

    axis = axes[1, 1]
    frequent = sorted(test["per_label"].items(), key=lambda item: (-item[1]["support"], item[0]))[:12]
    axis.barh(np.arange(len(frequent)), [metrics["f1"] for _, metrics in frequent], color=TEAL, height=0.65)
    axis.set_yticks(
        np.arange(len(frequent)),
        [f"{compact_label(label)} (n={metrics['support']:,})" for label, metrics in frequent],
    )
    axis.invert_yaxis()
    axis.set_xlabel("Per-label F1; n = gold test support")
    axis.set_title("Test F1: 12 most frequent IPC labels", loc="left", pad=14)
    for i, (_, metrics) in enumerate(frequent):
        axis.text(min(metrics["f1"] + 0.015, 0.94), i, f"{metrics['f1']:.3f}", va="center", fontsize=9)
    style_axis(axis, horizontal=True)
    figure.text(
        0.08,
        0.025,
        "Source: saved training.json and matching test report. No new fitting or test evaluation.\n"
        "No epoch-loss history was recorded. Scores describe retained dataset excerpts, not legal certainty.",
        fontsize=10,
        color="#53677B",
    )
    files = save_figure(figure, output_dir, "training_results")

    ordered = sorted(test["per_label"].items(), key=section_order)
    figure, axis = plt.subplots(figsize=(11, max(6, len(ordered) * 0.29 + 2.2)))
    figure.subplots_adjust(left=0.29, right=0.95, top=0.965, bottom=0.035)
    bars = axis.barh(
        np.arange(len(ordered)), [metrics["f1"] for _, metrics in ordered], color=TEAL, height=0.68
    )
    axis.set_yticks(
        np.arange(len(ordered)),
        [f"{compact_label(label)}  |  n={metrics['support']:,}" for label, metrics in ordered],
    )
    axis.invert_yaxis()
    axis.bar_label(bars, fmt="%.3f", padding=4, fontsize=8)
    axis.set_title(f"Held-out per-label F1: all {len(ordered)} IPC labels\n{version}", loc="left", pad=22)
    axis.set_xlabel("F1 score (0 to 1); n = gold test support")
    style_axis(axis, horizontal=True)
    axis.tick_params(axis="y", labelsize=9)
    files.extend(save_figure(figure, output_dir, "per_label_f1"))

    csv_path = output_dir / "per_label_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["label", "precision", "recall", "f1", "support", "supported"]
        )
        writer.writeheader()
        writer.writerows({"label": label, **metrics} for label, metrics in ordered)
    files.append(str(csv_path))
    provenance = {
        "model_version": version,
        "files": files,
        "source_reports": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in (training_path, test_path)
        ],
        "new_training_performed": False,
        "new_test_evaluation_performed": False,
        "epoch_loss_history_available": False,
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=Settings().artifact_dir / "reports")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        result = plot_reports(args.reports_dir, args.output_dir or args.reports_dir / "figures")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Cannot plot results: {exc}\nTrain and run final evaluation before plotting.\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
