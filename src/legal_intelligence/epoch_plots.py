"""Export epoch curves from measured JSON history; never fit or score a model."""

import json
from pathlib import Path


def plot_epochs(run_dir):
    # Optional plotting dependency is not needed by the API or ordinary baseline training.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    run_dir = Path(run_dir)
    history = json.loads((run_dir / "epoch_history.json").read_text(encoding="utf-8"))
    rows = history["epochs"]
    if not rows:
        raise ValueError("No completed epochs have been recorded.")
    epochs = [row["epoch"] for row in rows]
    best = min(rows, key=lambda row: row["validation_binary_log_loss"])["epoch"]
    version = history["model_version"]
    directory = run_dir / "figures"
    directory.mkdir(exist_ok=True)
    blue, teal, ink = "#2463A0", "#168174", "#172B43"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": ink,
            "axes.labelcolor": ink,
            "axes.titleweight": "bold",
        }
    )

    def draw(axis, key, title, ylabel, extra_macro=False):
        for split, label, color in [("train", "Training", blue), ("validation", "Validation", teal)]:
            axis.plot(
                epochs,
                [row[f"{split}_{key}"] for row in rows],
                "o-",
                color=color,
                linewidth=2,
                markersize=4,
                label=label + (" micro" if extra_macro else ""),
            )
            if extra_macro:
                axis.plot(
                    epochs,
                    [row[f"{split}_macro_f1"] for row in rows],
                    "s--",
                    color=color,
                    linewidth=1.7,
                    markersize=3,
                    label=label + " macro",
                )
        axis.axvline(best, color="#D46B2B", linestyle=":", linewidth=1.5, label=f"Selected epoch {best}")
        axis.set_title(title, loc="left", pad=12)
        axis.set_xlabel("Completed training epoch")
        axis.set_ylabel(ylabel)
        axis.set_xlim(0.7, max(epochs) + 0.3)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
        if key != "binary_log_loss":
            axis.set_ylim(0, 1)
        else:
            maximum = max(row[f"{split}_{key}"] for row in rows for split in ("train", "validation"))
            axis.set_ylim(0, max(0.001, maximum * 1.15))
        axis.grid(color="#DFE5EB", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#CCD5DF")
        axis.legend(frameon=False, fontsize=9, loc="best")

    files = []

    def save(figure, name):
        for extension in ("png", "svg"):
            path = directory / f"{name}.{extension}"
            figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
            files.append(str(path))
        plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    figure.subplots_adjust(left=0.08, right=0.97, top=0.81, bottom=0.11, hspace=0.50, wspace=0.27)
    figure.suptitle(
        (
            "GPU transformer experiment"
            if version.startswith("ipc-distilbert-gpu-")
            else "CPU epoch experiment"
        )
        + "\nMeasured training and validation history",
        x=0.08,
        y=0.985,
        ha="left",
        fontsize=21,
        fontweight="bold",
    )
    figure.text(
        0.08, 0.875, f"{version}  |  {len(rows)} completed epochs  |  selected epoch {best}", fontsize=11
    )
    draw(axes[0, 0], "binary_log_loss", "Mean binary log loss", "Cross-entropy per document-label pair")
    draw(axes[0, 1], "micro_f1", "Micro F1 at threshold 0.5", "Micro F1")
    draw(axes[1, 0], "macro_f1", "Macro F1 at threshold 0.5", "Macro F1")
    draw(
        axes[1, 1], "subset_accuracy", "Exact-match subset accuracy", "Fraction of entire label sets correct"
    )
    figure.text(
        0.08,
        0.025,
        "Checkpoint: lowest validation loss. All epoch classification scores use threshold 0.5.\n"
        "Final threshold tuning happens after checkpoint restoration. No held-out test data appears in these curves.",
        fontsize=10,
        color="#53677B",
    )
    save(figure, "epoch_overview")
    specifications = [
        ("binary_log_loss", "Loss versus epoch", "Mean binary log loss", "loss_vs_epoch", False),
        ("micro_f1", "F1 versus epoch (fixed threshold 0.5)", "F1 score", "f1_vs_epoch", True),
        (
            "subset_accuracy",
            "Exact-match accuracy versus epoch (threshold 0.5)",
            "Fraction of entire label sets correct",
            "subset_accuracy_vs_epoch",
            False,
        ),
    ]
    for key, title, ylabel, filename, macro in specifications:
        figure, axis = plt.subplots(figsize=(11, 6))
        draw(axis, key, title, ylabel, macro)
        figure.suptitle(version, fontsize=10, y=0.995, color="#53677B")
        figure.tight_layout(pad=2)
        save(figure, filename)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "model_version": version,
                "epochs": len(rows),
                "best_epoch": best,
                "source": str(run_dir / "epoch_history.json"),
                "files": files,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return files
