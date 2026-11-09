"""CPU-only SGD experiment. Baseline artifacts and held-out test rows are never used for tuning."""

import copy
import csv
import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from legal_intelligence.artifacts import load_bundle, save_bundle, write_json
from legal_intelligence.data.store import connect, load_cases, load_metadata
from legal_intelligence.evaluation import evaluate_labels
from legal_intelligence.preprocessing import DECISIVE_RE, OUTCOME_RE, PROVISION_RE, preprocess_texts
from legal_intelligence.training import package_versions


@dataclass(frozen=True)
class EpochConfig:
    epochs: int = 20
    patience: int = 5
    seed: int = 42
    batch_size: int = 1024
    min_label_support: int = 5
    max_features: int = 30000
    min_df: int = 2
    alpha: float = 0.00001
    thresholds: tuple = (0.15, 0.25, 0.35, 0.5)

    def validate(self):
        if any(
            value < 1
            for value in (
                self.epochs,
                self.patience,
                self.batch_size,
                self.min_label_support,
                self.max_features,
                self.min_df,
            )
        ):
            raise ValueError(
                "Epoch counts, patience, batch size and support/feature limits must be positive."
            )
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive.")
        if not self.thresholds or any(not 0 < value < 1 for value in self.thresholds):
            raise ValueError("Threshold candidates must be between zero and one.")


class IncrementalOVRClassifier(ClassifierMixin, BaseEstimator):
    """One binary SGD model per label, using only public sklearn incremental APIs.

    Each column has fixed binary classes [0, 1], even when a minibatch has no positives.
    sklearn 1.9.1 OneVsRestClassifier.partial_fit cannot initialize a multilabel indicator
    matrix directly; this small adapter avoids changing its private fitted attributes.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        """Reset and perform one pass; epoch training calls partial_fit explicitly."""
        for attribute in ("estimators_", "classes_", "n_features_in_"):
            if hasattr(self, attribute):
                delattr(self, attribute)
        return self.partial_fit(X, y)

    def partial_fit(self, X, y):
        y = np.asarray(y)
        if y.ndim != 2 or len(y) != X.shape[0] or y.shape[1] < 1:
            raise ValueError("Expected an aligned two-dimensional multilabel indicator matrix.")
        if not np.isin(y, (0, 1)).all():
            raise ValueError("Targets must contain only zero or one.")
        if not hasattr(self, "estimators_"):
            base = self.estimator if self.estimator is not None else SGDClassifier(loss="log_loss")
            self.estimators_ = [clone(base) for _ in range(y.shape[1])]
            self.classes_ = np.arange(y.shape[1])
            self.n_features_in_ = X.shape[1]
        if len(self.estimators_) != y.shape[1] or self.n_features_in_ != X.shape[1]:
            raise ValueError("Feature count and label order must stay fixed across batches.")
        for index, estimator in enumerate(self.estimators_):
            estimator.partial_fit(X, y[:, index], classes=np.array([0, 1]))
        return self

    def decision_function(self, X):
        check_is_fitted(self, "estimators_")
        return np.column_stack([estimator.decision_function(X) for estimator in self.estimators_])

    def predict_proba(self, X):
        return expit(self.decision_function(X))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(np.uint8)


@dataclass
class MetricTarget:
    truth: np.ndarray
    supported_count: int
    universe: list


def metric_target(gold, labels):
    universe = list(labels) + sorted({label for row in gold for label in row} - set(labels))
    positions = {label: index for index, label in enumerate(universe)}
    truth = np.zeros((len(gold), len(universe)), dtype=np.uint8)
    for row_index, row in enumerate(gold):
        for label in row:
            truth[row_index, positions[label]] = 1
    return MetricTarget(truth, len(labels), universe)


def mean_binary_log_loss(y, logits):
    """Unregularized mean BCE over all supported document-label pairs, evaluated from logits.

    logaddexp avoids log(0) from saturated probabilities. Unsupported labels have no
    learned score, so they enter classification metrics as FN but not this loss.
    """
    if y.shape != logits.shape or not y.size or not np.isfinite(logits).all():
        raise ValueError("Loss requires finite, non-empty, aligned logits and labels.")
    values = np.asarray(logits, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, values) - np.asarray(y, dtype=np.float64) * values))


def classification_metrics(target, scores, threshold=0.5):
    if scores.shape != (target.truth.shape[0], target.supported_count):
        raise ValueError("Metric scores and label order do not align.")
    predicted = np.zeros_like(target.truth)
    predicted[:, : target.supported_count] = scores >= threshold
    truth = target.truth
    tp = np.count_nonzero(predicted & truth, axis=0).astype(np.float64)
    predicted_count = predicted.sum(axis=0)
    support = truth.sum(axis=0)
    denominator = predicted_count + support
    per_f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator != 0)
    total_tp = tp.sum()
    return {
        "micro_precision": float(total_tp / max(int(predicted_count.sum()), 1)),
        "micro_recall": float(total_tp / max(int(support.sum()), 1)),
        "micro_f1": float(2 * total_tp / max(int(denominator.sum()), 1)),
        "macro_f1": float(per_f1.mean()),
        "subset_accuracy": float(np.all(predicted == truth, axis=1).mean()),
    }


def _epoch_metrics(classifier, matrix, target):
    logits = classifier.decision_function(matrix)
    result = classification_metrics(target, expit(logits), 0.5)
    result["binary_log_loss"] = mean_binary_log_loss(target.truth[:, : target.supported_count], logits)
    return result


@dataclass
class EarlyStopping:
    patience: int
    best_loss: float = float("inf")
    best_epoch: int = 0
    stale_epochs: int = 0

    def update(self, epoch, validation_loss):
        if not np.isfinite(validation_loss):
            raise ValueError("Validation loss is non-finite; experiment stopped without test evaluation.")
        improved = validation_loss < self.best_loss
        if improved:
            self.best_loss, self.best_epoch, self.stale_epochs = validation_loss, epoch, 0
        else:
            self.stale_epochs += 1
        return improved, self.stale_epochs >= self.patience


def feature_signature(features):
    vectorizer = features.named_steps["tfidf"]
    vocabulary = {word: int(index) for word, index in vectorizer.vocabulary_.items()}
    digest = hashlib.sha256(json.dumps(vocabulary, sort_keys=True).encode())
    digest.update(vectorizer.idf_.tobytes())
    return digest.hexdigest()


def _baseline_snapshot(artifact_dir):
    paths = [artifact_dir / name for name in ("ipc_model.joblib", "ipc_model.sha256", "ipc_model.json")]
    reports = artifact_dir / "reports"
    paths += [reports / "training.json"] + list(reports.glob("test-*.json"))
    paths += list((reports / "figures").glob("*"))
    return {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.is_file()
    }


def verify_baseline(snapshot):
    changed = [
        path
        for path, digest in snapshot.items()
        if not Path(path).exists() or hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest
    ]
    if changed:
        raise ValueError(f"Baseline artifacts changed during the epoch experiment: {changed}")
    return True


DEFINITIONS = {
    "epoch": "Exactly one complete pass over all eligible training records, in a seeded shuffled order.",
    "binary_log_loss": "Unregularized mean binary cross-entropy over all supported document-label pairs, from logits; lower is better.",
    "classification": "At fixed score threshold 0.5; unsupported gold labels count as false negatives.",
    "macro_f1": "Unweighted F1 mean over supported labels plus any unsupported gold labels for this split; zero denominators contribute zero.",
    "subset_accuracy": "Fraction of documents whose entire predicted label set exactly matches the entire gold set; not per-label accuracy.",
    "checkpoint": "Strictly lowest validation binary log loss; ties retain the earlier checkpoint. Stop after the configured number of consecutive non-improving epochs.",
}


def _persist_history(run_dir, metadata, history, status):
    write_json(
        run_dir / "epoch_history.json",
        {
            "model_version": metadata["model_version"],
            "status": status,
            "config": metadata["config"],
            "definitions": DEFINITIONS,
            "epochs": history,
        },
    )
    if history:
        with (run_dir / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)


def train_epochs(db_path, artifact_dir, config=None):
    config = config or EpochConfig()
    config.validate()
    start_time = time.perf_counter()
    with connect(db_path) as db:
        fields = ("case_id", "facts_text", "act_section_labels")
        rows = load_cases(db, "train", fields=fields)
        validation = load_cases(db, "validation", fields=fields)
        provenance = load_metadata(db, "provenance")
        preparation = load_metadata(db, "preparation_report")
    if not rows or not validation or not provenance or not preparation:
        raise ValueError("Prepared training/validation data and provenance are required.")
    encoder = MultiLabelBinarizer()
    all_y = encoder.fit_transform([row["act_section_labels"] for row in rows])
    support = all_y.sum(axis=0)
    keep = np.flatnonzero((support >= config.min_label_support) & (support < len(rows)))
    labels = encoder.classes_[keep].tolist()
    if not labels:
        raise ValueError("No non-constant labels meet minimum training support.")
    y = all_y[:, keep].astype(np.uint8)
    train_gold = [row["act_section_labels"] for row in rows]
    validation_gold = [row["act_section_labels"] for row in validation]
    train_target, validation_target = (
        metric_target(train_gold, labels),
        metric_target(validation_gold, labels),
    )
    timestamp = datetime.now(UTC).isoformat()
    identity = timestamp + json.dumps(asdict(config), sort_keys=True) + json.dumps(provenance, sort_keys=True)
    version = "ipc-sgd-epochs-" + hashlib.sha256(identity.encode()).hexdigest()[:12]
    run_dir = artifact_dir / "experiments" / version
    run_dir.mkdir(parents=True, exist_ok=False)
    baseline = _baseline_snapshot(artifact_dir)
    write_json(run_dir / "baseline_snapshot.json", baseline)
    features = Pipeline(
        [
            (
                "preprocess",
                FunctionTransformer(
                    preprocess_texts,
                    validate=False,
                    kw_args={
                        "reference_pattern": PROVISION_RE.pattern,
                        "reference_flags": PROVISION_RE.flags,
                    },
                ),
            ),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=config.min_df,
                    max_features=config.max_features,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    print(
        f"Fitting frozen training-only TF-IDF: {len(rows)} train, {len(validation)} validation, {len(labels)} labels.",
        flush=True,
    )
    with threadpool_limits(limits=1):
        matrix = features.fit_transform([row["facts_text"] for row in rows]).tocsr()
        validation_matrix = features.transform([row["facts_text"] for row in validation]).tocsr()
    signature = feature_signature(features)
    metadata = {
        "model_version": version,
        "experiment": "one_vs_rest_sgd_logistic_epochs",
        "started_at": timestamp,
        "config": asdict(config),
        "packages": package_versions(),
        "python_version": platform.python_version(),
        "device": "cpu",
        "workers": 1,
        "blas_threads": 1,
        "training_documents": len(rows),
        "validation_documents": len(validation),
        "supported_labels": labels,
        "training_label_support": {str(label): int(support[i]) for i, label in enumerate(encoder.classes_)},
        "excluded_labels": {
            str(encoder.classes_[i]): {
                "support": int(support[i]),
                "reason": "rare" if support[i] < config.min_label_support else "constant",
            }
            for i in range(len(support))
            if i not in set(keep)
        },
        "validation_only_labels": sorted(set(validation_target.universe) - set(encoder.classes_)),
        "dataset_provenance": provenance,
        "preparation_report_sha256": hashlib.sha256(
            json.dumps(preparation, sort_keys=True).encode()
        ).hexdigest(),
        "training_case_ids_sha256": hashlib.sha256(
            "\n".join(row["case_id"] for row in rows).encode()
        ).hexdigest(),
        "validation_case_ids_sha256": hashlib.sha256(
            "\n".join(row["case_id"] for row in validation).encode()
        ).hexdigest(),
        "feature_signature": signature,
        "feature_count": matrix.shape[1],
        "label_order_sha256": hashlib.sha256(json.dumps(labels).encode()).hexdigest(),
        "preprocessing": {
            "version": 1,
            "facts_selection": preparation["facts_policy"],
            "reference_regex": PROVISION_RE.pattern,
            "outcome_regex": OUTCOME_RE.pattern,
            "decisive_regex": DECISIVE_RE.pattern,
            "unicode_normalization": "NFKC",
        },
        "optimizer": {
            "loss": "log_loss",
            "penalty": "l2",
            "alpha": config.alpha,
            "learning_rate": "optimal",
            "shuffle_in_estimator": False,
            "binary_classes": [0, 1],
            "average": False,
        },
        "epoch_metric_threshold": 0.5,
        "definitions": DEFINITIONS,
        "calibrated": False,
        "statute_version": None,
        "test_evaluated_during_training": False,
        "baseline_snapshot": baseline,
        "limitations": [
            "Historical IPC dataset labels only; no BNS translation.",
            "Rule-selected narratives and original dataset labels may be noisy.",
            "Loss averages supported label pairs; rare labels contribute little to this aggregate.",
            "Validation is reused for checkpoint selection and final threshold selection.",
        ],
    }
    write_json(run_dir / "metadata.json", metadata)
    classifier = IncrementalOVRClassifier(
        SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=config.alpha,
            learning_rate="optimal",
            shuffle=False,
            random_state=config.seed,
            tol=None,
        )
    )
    random = np.random.default_rng(config.seed)
    stopping = EarlyStopping(config.patience)
    history = []
    checkpoint_path = run_dir / "best_checkpoint.joblib"
    print(f"Experiment directory: {run_dir}", flush=True)
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        order = random.permutation(len(rows))
        examples_seen = 0
        with threadpool_limits(limits=1):
            for start in range(0, len(order), config.batch_size):
                batch = order[start : start + config.batch_size]
                classifier.partial_fit(matrix[batch], y[batch])
                examples_seen += len(batch)
            train_metrics = _epoch_metrics(classifier, matrix, train_target)
            validation_metrics = _epoch_metrics(classifier, validation_matrix, validation_target)
        if examples_seen != len(rows) or feature_signature(features) != signature:
            raise ValueError("Epoch pass count or frozen feature invariant failed.")
        improved, should_stop = stopping.update(epoch, validation_metrics["binary_log_loss"])
        record = {
            "epoch": epoch,
            "examples_seen": examples_seen,
            "order_sha256": hashlib.sha256(order.astype("<i8").tobytes()).hexdigest(),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "best_checkpoint": improved,
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(record)
        if improved:
            pipeline = Pipeline(features.steps + [("classifier", copy.deepcopy(classifier))])
            checkpoint = {
                "kind": "ipc_classifier",
                "format_version": 1,
                "pipeline": pipeline,
                "label_encoder": encoder,
                "supported_indices": keep,
                "labels": labels,
                "thresholds": [0.5] * len(labels),
                "metadata": {**metadata, "best_epoch": epoch, "best_validation_loss": stopping.best_loss},
            }
            save_bundle(checkpoint, checkpoint_path)
        _persist_history(run_dir, metadata, history, "running")
        print(
            f"Epoch {epoch:02d}/{config.epochs}: train loss={train_metrics['binary_log_loss']:.6f}, "
            f"val loss={validation_metrics['binary_log_loss']:.6f}, "
            f"train F1={train_metrics['micro_f1']:.4f}, val F1={validation_metrics['micro_f1']:.4f}"
            + (
                " [best checkpoint]" if improved else f" [patience {stopping.stale_epochs}/{config.patience}]"
            ),
            flush=True,
        )
        if should_stop:
            break
    # Restore actual saved best weights, rather than using the last epoch's classifier.
    restored = load_bundle(checkpoint_path, "ipc_classifier")
    with threadpool_limits(limits=1):
        scores = restored["pipeline"].named_steps["classifier"].predict_proba(validation_matrix)
    threshold_search = []
    for threshold in config.thresholds:
        metrics = classification_metrics(validation_target, scores, threshold)
        threshold_search.append({"threshold": threshold, "validation_micro_f1": metrics["micro_f1"]})
    selected = max(threshold_search, key=lambda item: item["validation_micro_f1"])
    thresholds = [selected["threshold"]] * len(labels)
    metadata.update(
        {
            "completed_at": datetime.now(UTC).isoformat(),
            "epochs_completed": len(history),
            "best_epoch": stopping.best_epoch,
            "best_validation_loss": stopping.best_loss,
            "stopped_early": stopping.stale_epochs >= config.patience,
            "stop_reason": "validation_loss_patience"
            if stopping.stale_epochs >= config.patience
            else "epoch_limit",
            "seconds": time.perf_counter() - start_time,
            "threshold_policy": "Global threshold maximizing validation full-universe micro F1 after restoring the lowest-loss checkpoint; stored per label.",
            "selected_threshold": selected["threshold"],
            "baseline_unchanged": verify_baseline(baseline),
        }
    )
    restored.update({"thresholds": thresholds, "metadata": metadata})
    save_bundle(restored, run_dir / "model.joblib")
    write_json(run_dir / "metadata.json", metadata)
    write_json(
        run_dir / "validation.json",
        {
            "model_version": version,
            "threshold_search": threshold_search,
            "selected_threshold": selected["threshold"],
            "fixed_threshold_best_epoch": history[stopping.best_epoch - 1],
            "metrics": evaluate_labels(validation_gold, scores, labels, thresholds),
            "subset_accuracy": classification_metrics(validation_target, scores, selected["threshold"])[
                "subset_accuracy"
            ],
        },
    )
    _persist_history(run_dir, metadata, history, "completed")
    return {
        "experiment_dir": str(run_dir),
        "model_version": version,
        "epochs_completed": len(history),
        "best_epoch": stopping.best_epoch,
        "best_validation_loss": stopping.best_loss,
        "selected_threshold": selected["threshold"],
        "baseline_unchanged": True,
        "validation_micro_f1": selected["validation_micro_f1"],
        "test_evaluated": False,
    }


def evaluate_epochs(db_path, run_dir):
    run_dir = Path(run_dir)
    output = run_dir / "test.json"
    if output.exists() or (run_dir / "test_evaluation.started.json").exists():
        raise ValueError(
            "Held-out evaluation was already started/recorded for this experiment; no repeated test evaluation."
        )
    bundle = load_bundle(run_dir / "model.joblib", "ipc_classifier")
    metadata = bundle["metadata"]
    if metadata.get("experiment") != "one_vs_rest_sgd_logistic_epochs":
        raise ValueError("Not an epoch experiment bundle.")
    if not (run_dir / "validation.json").exists():
        raise ValueError("Validation checkpoint/threshold selection must finish before test evaluation.")
    with connect(db_path) as db:
        provenance = load_metadata(db, "provenance")
    if provenance != metadata["dataset_provenance"]:
        raise ValueError("Evaluation dataset provenance differs from the trained experiment.")
    # Exclusive marker claims this model's single evaluation before any held-out rows are read.
    with (run_dir / "test_evaluation.started.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {"model_version": metadata["model_version"], "started_at": datetime.now(UTC).isoformat()}, stream
        )
    with connect(db_path) as db:
        rows = load_cases(db, "test", fields=("case_id", "facts_text", "act_section_labels"))
    if not rows:
        raise ValueError("No eligible held-out examples.")
    gold = [row["act_section_labels"] for row in rows]
    target = metric_target(gold, bundle["labels"])
    with threadpool_limits(limits=1):
        matrix = bundle["pipeline"][:-1].transform([row["facts_text"] for row in rows])
        logits = bundle["pipeline"].named_steps["classifier"].decision_function(matrix)
        scores = expit(logits)
    result = {
        "model_version": metadata["model_version"],
        "split": "test",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "evaluation_count": 1,
        "best_epoch": metadata["best_epoch"],
        "threshold": metadata["selected_threshold"],
        "metrics": evaluate_labels(gold, scores, bundle["labels"], bundle["thresholds"]),
        "binary_log_loss": mean_binary_log_loss(target.truth[:, : target.supported_count], logits),
        "subset_accuracy": classification_metrics(target, scores, metadata["selected_threshold"])[
            "subset_accuracy"
        ],
        "test_only_labels": sorted(set(target.universe) - set(bundle["label_encoder"].classes_)),
        "dataset_checksums": provenance["files"],
        "baseline_unchanged": verify_baseline(metadata["baseline_snapshot"]),
    }
    write_json(output, result)
    pointer_path = run_dir.parent / "latest.json"
    if pointer_path.exists():
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        if pointer.get("model_version") == metadata["model_version"]:
            pointer.update({"test_evaluated": True, "test_report": str(output)})
            write_json(pointer_path, pointer)
    return result
