import hashlib
import importlib.metadata
import json
import warnings
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer
from threadpoolctl import threadpool_limits

from legal_intelligence.artifacts import load_bundle, save_bundle, write_json
from legal_intelligence.data.store import connect, load_cases, load_metadata
from legal_intelligence.evaluation import evaluate_labels
from legal_intelligence.preprocessing import DECISIVE_RE, OUTCOME_RE, PROVISION_RE, preprocess_texts


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    min_label_support: int = 5
    max_features: int = 30000
    min_df: int = 2
    c_values: tuple = (1.0, 4.0)
    thresholds: tuple = (0.15, 0.25, 0.35, 0.5)
    analyzers: tuple = ("word",)
    max_iter: int = 1000


def package_versions():
    return {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scipy", "scikit-learn", "spacy", "joblib", "PyMuPDF", "fastapi")
    }


def predict_scores(pipeline, texts, labels):
    values = np.asarray(pipeline.predict_proba(texts))
    if len(labels) == 1 and values.shape[1] == 2:
        values = values[:, 1:2]
    if values.shape != (len(texts), len(labels)):
        raise ValueError("Classifier output and saved label order do not align")
    return values


def train(db_path, model_path, report_dir, config=None):
    config = config or TrainConfig()
    with connect(db_path) as db:
        fields = ("case_id", "facts_text", "act_section_labels")
        train_rows = load_cases(db, "train", fields=fields)
        validation = load_cases(db, "validation", fields=fields)
        provenance = load_metadata(db, "provenance")
        preparation = load_metadata(db, "preparation_report")
    if not train_rows or not validation:
        raise ValueError("Training and validation data are required. Run prepare first.")
    if not preparation:
        raise ValueError("Missing dataset audit; run prepare.")
    # Test examples and labels are deliberately not loaded by this function.
    encoder = MultiLabelBinarizer()
    all_y = encoder.fit_transform([r["act_section_labels"] for r in train_rows])
    supports = all_y.sum(axis=0)
    keep = np.flatnonzero((supports >= config.min_label_support) & (supports < len(train_rows)))
    labels = encoder.classes_[keep].tolist()
    if not labels:
        raise ValueError("No non-constant labels meet minimum training support.")
    y = all_y[:, keep]
    validation_gold = [r["act_section_labels"] for r in validation]
    label_positions = {label: i for i, label in enumerate(labels)}
    validation_y = np.zeros((len(validation), len(labels)), dtype=np.uint8)
    for i, gold in enumerate(validation_gold):
        for label in gold:
            if label in label_positions:
                validation_y[i, label_positions[label]] = 1
    texts, dev_texts = [r["facts_text"] for r in train_rows], [r["facts_text"] for r in validation]
    runs, best = [], None
    for analyzer in config.analyzers:
        for c in config.c_values:
            print(
                f"Training analyzer={analyzer}, C={c}, documents={len(texts)}, labels={len(labels)}",
                flush=True,
            )
            pipeline = Pipeline(
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
                            analyzer=analyzer,
                            ngram_range=(1, 2) if analyzer == "word" else (3, 5),
                            sublinear_tf=True,
                            min_df=config.min_df,
                            max_features=config.max_features,
                            dtype=np.float32,
                        ),
                    ),
                    (
                        "classifier",
                        OneVsRestClassifier(
                            LogisticRegression(
                                C=c, solver="liblinear", max_iter=config.max_iter, random_state=config.seed
                            ),
                            n_jobs=1,
                        ),
                    ),
                ]
            )
            with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                pipeline.fit(texts, y)
                scores = predict_scores(pipeline, dev_texts, labels)
            convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
            if convergence:
                raise ValueError(
                    "Logistic regression failed to converge; increase --max-iter before reporting metrics."
                )
            for threshold in config.thresholds:
                f1 = float(
                    f1_score(
                        validation_y,
                        scores >= threshold,
                        average="binary" if len(labels) == 1 else "micro",
                        zero_division=0,
                    )
                )
                run = {
                    "analyzer": analyzer,
                    "C": c,
                    "threshold": threshold,
                    "validation_supported_micro_f1": f1,
                }
                runs.append(run)
                if best is None or f1 > best["selection_score"]:
                    best = {
                        "pipeline": pipeline,
                        "scores": scores,
                        "selection_score": f1,
                        "configuration": run,
                    }
    thresholds = [best["configuration"]["threshold"]] * len(labels)
    timestamp = datetime.now(UTC).isoformat()
    provenance_hash = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    version = (
        "ipc-tfidf-"
        + hashlib.sha256((timestamp + provenance_hash + str(asdict(config))).encode()).hexdigest()[:12]
    )
    excluded_labels = {
        str(label): {
            "support": int(supports[i]),
            "reason": "rare" if supports[i] < config.min_label_support else "constant",
        }
        for i, label in enumerate(encoder.classes_)
        if i not in set(keep)
    }
    metadata = {
        "model_version": version,
        "trained_at": timestamp,
        "config": asdict(config),
        "selected_configuration": best["configuration"],
        "packages": package_versions(),
        "dataset_provenance": provenance,
        "dataset_provenance_sha256": provenance_hash,
        "training_documents": len(train_rows),
        "validation_documents": len(validation),
        "training_case_ids_sha256": hashlib.sha256(
            "\n".join(r["case_id"] for r in train_rows).encode()
        ).hexdigest(),
        "training_label_support": {str(label): int(supports[i]) for i, label in enumerate(encoder.classes_)},
        "excluded_labels": excluded_labels,
        "supported_labels": labels,
        "validation_only_labels": sorted(
            {s for r in validation for s in r["act_section_labels"]} - set(encoder.classes_)
        ),
        "threshold_policy": "One global threshold chosen by validation supported-label micro F1; stored per label.",
        "preprocessing": {
            "version": 1,
            "mask_explicit_references": True,
            "facts_selection": preparation["facts_policy"],
            "reference_regex": PROVISION_RE.pattern,
            "outcome_regex": OUTCOME_RE.pattern,
            "decisive_regex": DECISIVE_RE.pattern,
            "unicode_normalization": "NFKC",
        },
        "statute_version": None,
        "calibrated": False,
        "test_evaluated_during_training": False,
        "limitations": [
            "IPC dataset labels only; no BNS translation.",
            "Rule-selected narratives may contain unrecognized judicial language; not a verified facts-only corpus.",
            "Dataset excerpt boundaries and supplied labels may be noisy.",
            "No calibrated confidence, category classifier, trained NER, or retrieval accuracy claim.",
        ],
    }
    bundle = {
        "kind": "ipc_classifier",
        "format_version": 1,
        "pipeline": best["pipeline"],
        "label_encoder": encoder,
        "supported_indices": keep,
        "labels": labels,
        "thresholds": thresholds,
        "metadata": metadata,
    }
    save_bundle(bundle, model_path)
    report = {
        "metadata": metadata,
        "validation": evaluate_labels(validation_gold, best["scores"], labels, thresholds),
        "validation_search": runs,
    }
    write_json(report_dir / "training.json", report)
    write_json(model_path.with_suffix(".json"), metadata)
    return report


def final_evaluate(db_path, model_path, report_dir):
    bundle = load_bundle(model_path, "ipc_classifier")
    version = bundle["metadata"]["model_version"]
    output = report_dir / f"test-{version}.json"
    if output.exists():
        raise ValueError(
            f"Test evaluation already recorded for this model: {output}; keep test separate from tuning."
        )
    with connect(db_path) as db:
        provenance = load_metadata(db, "provenance")
        rows = load_cases(db, "test", fields=("case_id", "facts_text", "act_section_labels"))
    if provenance != bundle["metadata"]["dataset_provenance"]:
        raise ValueError("Evaluation dataset provenance differs from the model's prepared release.")
    if not rows:
        raise ValueError("No eligible test examples")
    with threadpool_limits(limits=1):
        scores = predict_scores(bundle["pipeline"], [r["facts_text"] for r in rows], bundle["labels"])
    gold = [r["act_section_labels"] for r in rows]
    training_labels = set(bundle["label_encoder"].classes_)
    result = {
        "model_version": version,
        "split": "test",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "test_only_labels": sorted({s for labels in gold for s in labels} - training_labels),
        "metrics": evaluate_labels(gold, scores, bundle["labels"], bundle["thresholds"]),
        "dataset_checksums": provenance["files"],
    }
    write_json(output, result)
    return result


def train_category_classifier(dataset_path):
    """Reserved interface. A future adapter must establish real category labels and splits first."""
    raise NotImplementedError(
        "Category training disabled: no verified category-labelled dataset adapter is installed."
    )
