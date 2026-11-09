import argparse
import json
import sqlite3
import sys
from pathlib import Path

from legal_intelligence.config import Settings


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="AI-Based Legal Case Intelligence: reproducible offline ML pipeline"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "download", help="Download pinned LeSICiN data (~511 MB); verify official MD5 and record SHA-256"
    )
    local = commands.add_parser(
        "import-lesicin", help="Verify and copy the five official files from a local folder"
    )
    local.add_argument("source", type=Path)
    commands.add_parser("prepare", help="Validate, inspect, normalize and audit; preserve original splits")
    train = commands.add_parser("train", help="Train IPC classifier and tune on validation only")
    train.add_argument("--min-label-support", type=int, default=5)
    train.add_argument("--max-features", type=int, default=30000)
    train.add_argument("--max-iter", type=int, default=1000)
    train.add_argument("--c-values", type=float, nargs="+", default=[1.0, 4.0])
    train.add_argument(
        "--compare-char",
        action="store_true",
        help="Also compare char ngrams on validation (higher resource cost)",
    )
    commands.add_parser("evaluate", help="One final held-out test evaluation per model version")
    epoch = commands.add_parser("train-epochs", help="Separate CPU SGD experiment with real per-epoch logs")
    epoch.add_argument("--epochs", type=int, default=20)
    epoch.add_argument("--patience", type=int, default=5)
    epoch.add_argument("--seed", type=int, default=42)
    epoch.add_argument("--batch-size", type=int, default=1024)
    epoch.add_argument("--min-label-support", type=int, default=5)
    epoch.add_argument("--max-features", type=int, default=30000)
    epoch.add_argument("--alpha", type=float, default=0.00001)
    for command, help_text in [
        ("evaluate-epochs", "One final test evaluation of a selected epoch experiment"),
        ("plot-epochs", "Regenerate graphs from measured epoch history only"),
    ]:
        subparser = commands.add_parser(command, help=help_text)
        subparser.add_argument(
            "experiment_dir", type=Path, nargs="?", help="Defaults to latest epoch experiment"
        )
    gpu = commands.add_parser("train-gpu", help="Separate CUDA DistilBERT IPC classification experiment")
    gpu.add_argument("--epochs", type=int, default=3)
    gpu.add_argument("--patience", type=int, default=2)
    gpu.add_argument("--seed", type=int, default=42)
    gpu.add_argument("--batch-size", type=int, default=16)
    gpu.add_argument("--eval-batch-size", type=int, default=64)
    gpu.add_argument("--accumulation-steps", type=int, default=2)
    gpu.add_argument("--max-length", type=int, default=256)
    gpu.add_argument("--learning-rate", type=float, default=0.00003)
    gpu.add_argument(
        "--revision", default="main", help="Base model revision, resolved and pinned before download"
    )
    for command in ("evaluate-gpu", "plot-gpu"):
        subparser = commands.add_parser(command, help="Evaluate/plot the separate GPU experiment")
        subparser.add_argument(
            "experiment_dir", type=Path, nargs="?", help="Defaults to latest GPU experiment"
        )
    index = commands.add_parser("index", help="Build a separate sparse judgment-passage search index")
    index.add_argument("--splits", choices=["train", "validation", "test"], nargs="+", default=["train"])
    index.add_argument(
        "--max-passages", type=int, default=100000, help="0 for all passages; default bounds CPU/RAM"
    )
    args = parser.parse_args(argv)
    settings = Settings()
    reports = settings.artifact_dir / "reports"
    raw_dir = settings.data_dir / "raw" / "lesicin"
    try:
        if (
            args.command
            in ("train", "evaluate", "index", "train-epochs", "evaluate-epochs", "train-gpu", "evaluate-gpu")
            and not settings.db_path.exists()
        ):
            raise ValueError("Prepared dataset missing. Run download (or import-lesicin), then prepare.")
        if args.command in ("download", "import-lesicin"):
            from legal_intelligence.data.download import download, import_local

            result = download(raw_dir) if args.command == "download" else import_local(args.source, raw_dir)
        elif args.command == "prepare":
            from legal_intelligence.data.lesicin import prepare

            result = prepare(raw_dir, settings.db_path, reports / "dataset.json")
            result = {
                "counts": result["counts"],
                "eligibility": result["eligibility"],
                "exclusions": result["exclusion_reasons"],
                "cross_split_pairs": result["leakage"]["cross_split_pairs"],
                "report": str(reports / "dataset.json"),
            }
        elif args.command == "train":
            from legal_intelligence.training import TrainConfig, train

            if (
                args.min_label_support < 1
                or args.max_features < 2
                or args.max_iter < 1
                or any(c <= 0 for c in args.c_values)
            ):
                raise ValueError("Training parameters must be positive; max-features must be at least 2.")
            config = TrainConfig(
                min_label_support=args.min_label_support,
                max_features=args.max_features,
                c_values=tuple(args.c_values),
                max_iter=args.max_iter,
                analyzers=("word", "char_wb") if args.compare_char else ("word",),
            )
            result = train(settings.db_path, settings.model_path, reports, config)
            result = {
                "model_version": result["metadata"]["model_version"],
                "validation_micro_f1": result["validation"]["micro_f1"],
                "supported_labels": len(result["metadata"]["supported_labels"]),
                "report": str(reports / "training.json"),
            }
        elif args.command == "evaluate":
            from legal_intelligence.training import final_evaluate

            result = final_evaluate(settings.db_path, settings.model_path, reports)
            result = {
                "model_version": result["model_version"],
                "metrics": {k: v for k, v in result["metrics"].items() if k != "per_label"},
            }
        elif args.command == "train-epochs":
            from legal_intelligence.artifacts import write_json
            from legal_intelligence.epoch_plots import plot_epochs
            from legal_intelligence.epoch_training import EpochConfig, train_epochs

            config = EpochConfig(
                epochs=args.epochs,
                patience=args.patience,
                seed=args.seed,
                batch_size=args.batch_size,
                min_label_support=args.min_label_support,
                max_features=args.max_features,
                alpha=args.alpha,
            )
            result = train_epochs(settings.db_path, settings.artifact_dir, config)
            write_json(settings.artifact_dir / "experiments" / "latest.json", result)
            result["figures"] = plot_epochs(Path(result["experiment_dir"]))
        elif args.command in ("evaluate-epochs", "plot-epochs"):
            from legal_intelligence.epoch_plots import plot_epochs
            from legal_intelligence.epoch_training import evaluate_epochs

            directory = args.experiment_dir
            if directory is None:
                pointer = json.loads(
                    (settings.artifact_dir / "experiments" / "latest.json").read_text(encoding="utf-8")
                )
                directory = Path(pointer["experiment_dir"])
            if args.command == "evaluate-epochs":
                evaluation = evaluate_epochs(settings.db_path, directory)
                result = {
                    "experiment_dir": str(directory),
                    "model_version": evaluation["model_version"],
                    "best_epoch": evaluation["best_epoch"],
                    "threshold": evaluation["threshold"],
                    "metrics": {
                        key: value for key, value in evaluation["metrics"].items() if key != "per_label"
                    },
                    "binary_log_loss": evaluation["binary_log_loss"],
                    "subset_accuracy": evaluation["subset_accuracy"],
                    "baseline_unchanged": evaluation["baseline_unchanged"],
                    "test_report": str(directory / "test.json"),
                }
            else:
                result = {"experiment_dir": str(directory), "figures": plot_epochs(directory)}
        elif args.command == "train-gpu":
            from legal_intelligence.epoch_plots import plot_epochs
            from legal_intelligence.gpu_training import GPUConfig, train_gpu

            config = GPUConfig(
                epochs=args.epochs,
                patience=args.patience,
                seed=args.seed,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                accumulation_steps=args.accumulation_steps,
                max_length=args.max_length,
                learning_rate=args.learning_rate,
                revision=args.revision,
            )
            result = train_gpu(settings.db_path, settings.artifact_dir, config)
            result["figures"] = plot_epochs(Path(result["experiment_dir"]))
        elif args.command in ("evaluate-gpu", "plot-gpu"):
            from legal_intelligence.epoch_plots import plot_epochs

            directory = args.experiment_dir
            if directory is None:
                pointer = json.loads(
                    (settings.artifact_dir / "gpu_experiments" / "latest.json").read_text(encoding="utf-8")
                )
                directory = Path(pointer["experiment_dir"])
            if args.command == "evaluate-gpu":
                from legal_intelligence.gpu_training import evaluate_gpu

                evaluation = evaluate_gpu(settings.db_path, directory)
                result = {key: value for key, value in evaluation.items() if key != "metrics"}
                result["metrics"] = {
                    key: value for key, value in evaluation["metrics"].items() if key != "per_label"
                }
                result["test_report"] = str(directory / "test.json")
            else:
                result = {"experiment_dir": str(directory), "figures": plot_epochs(directory)}
        elif args.command == "index":
            from legal_intelligence.retrieval import build_index

            if args.max_passages < 0:
                raise ValueError("max-passages must be non-negative")
            result = build_index(
                settings.db_path,
                settings.index_path,
                reports,
                tuple(dict.fromkeys(args.splits)),
                args.max_passages,
            )
            result = {k: result[k] for k in ("passages", "cases", "splits", "truncated")}
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, sqlite3.Error, ImportError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
