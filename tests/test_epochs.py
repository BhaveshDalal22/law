"""Artificial fixtures test the epoch experiment in temp directories, never in production training."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from legal_intelligence.artifacts import load_bundle
from legal_intelligence.epoch_plots import plot_epochs
from legal_intelligence.epoch_training import (
    EarlyStopping,
    EpochConfig,
    IncrementalOVRClassifier,
    classification_metrics,
    evaluate_epochs,
    feature_signature,
    mean_binary_log_loss,
    metric_target,
    train_epochs,
    verify_baseline,
)


@pytest.fixture(scope="module")
def epoch_run(prepared, tmp_path_factory):
    directory = tmp_path_factory.mktemp("epoch-test-only")
    (directory / "ipc_model.joblib").write_bytes(b"baseline sentinel: must remain unchanged")
    result = train_epochs(
        prepared[0].db_path,
        directory,
        EpochConfig(
            epochs=3,
            patience=5,
            batch_size=4,
            min_label_support=2,
            min_df=1,
            max_features=500,
            thresholds=(0.3, 0.5),
        ),
    )
    return result, directory


def test_incremental_ovr_handles_first_batch_without_rare_positives():
    features = csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    model = IncrementalOVRClassifier(SGDClassifier(loss="log_loss", shuffle=False, random_state=42))
    model.partial_fit(features, np.array([[1, 0, 0], [0, 1, 0]]))
    model.partial_fit(features, np.array([[1, 0, 1], [0, 1, 0]]))
    assert len(model.estimators_) == 3 and model.predict_proba(features).shape == (2, 3)
    assert all(estimator.classes_.tolist() == [0, 1] for estimator in model.estimators_)
    assert all(estimator.t_ == 5 for estimator in model.estimators_)
    with pytest.raises(ValueError, match="stay fixed"):
        model.partial_fit(features, np.array([[1, 0], [0, 1]]))


def test_binary_loss_is_mean_over_pairs_and_stable_at_extreme_logits():
    truth = np.array([[1, 0], [0, 1]])
    logits = np.array([[np.log(3), -np.log(3)], [0.0, 0.0]])
    assert mean_binary_log_loss(truth, logits) == pytest.approx((-np.log(0.75) - np.log(0.5)) / 2)
    assert mean_binary_log_loss(truth, np.array([[1000.0, -1000.0], [-1000.0, 1000.0]])) == 0
    assert mean_binary_log_loss(truth, np.array([[-1000.0, 1000.0], [1000.0, -1000.0]])) == 1000


def test_epoch_classification_metrics_match_sklearn_and_count_unseen_labels():
    target = metric_target([["a", "unseen"], ["b"], ["a"]], ["a", "b"])
    scores = np.array([[0.9, 0.1], [0.1, 0.8], [0.2, 0.3]])
    predicted = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    result = classification_metrics(target, scores)
    micro = precision_recall_fscore_support(target.truth, predicted, average="micro", zero_division=0)
    macro = precision_recall_fscore_support(target.truth, predicted, average="macro", zero_division=0)
    assert result["micro_precision"] == micro[0]
    assert result["micro_recall"] == micro[1]
    assert result["micro_f1"] == micro[2]
    assert result["macro_f1"] == macro[2]
    assert result["subset_accuracy"] == accuracy_score(target.truth, predicted)
    assert result["subset_accuracy"] == pytest.approx(1 / 3)


def test_early_stopping_uses_loss_and_patience_including_ties():
    stopping = EarlyStopping(5)
    outcomes = [stopping.update(i, loss) for i, loss in enumerate([0.7, 0.4, 0.4, 0.5, 0.6, 0.7, 0.8], 1)]
    assert stopping.best_epoch == 2 and stopping.best_loss == 0.4
    assert not any(stop for _, stop in outcomes[:-1]) and outcomes[-1] == (False, True)


def test_history_complete_passes_frozen_features_and_serialization(epoch_run):
    result, directory = epoch_run
    run_dir = directory / "experiments" / result["model_version"]
    history = json.loads((run_dir / "epoch_history.json").read_text())["epochs"]
    with (run_dir / "epoch_metrics.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    bundle = load_bundle(run_dir / "model.joblib", "ipc_classifier")
    checkpoint = load_bundle(run_dir / "best_checkpoint.joblib", "ipc_classifier")
    assert [row["epoch"] for row in history] == [1, 2, 3]
    assert all(row["examples_seen"] == 10 for row in history)
    assert len({row["order_sha256"] for row in history}) == 3
    assert all(
        float(row["validation_binary_log_loss"]) == measured["validation_binary_log_loss"]
        for row, measured in zip(csv_rows, history)
    )
    assert bundle["labels"] == bundle["label_encoder"].classes_[bundle["supported_indices"]].tolist()
    assert feature_signature(bundle["pipeline"][:-1]) == bundle["metadata"]["feature_signature"]
    assert "khaki" not in bundle["pipeline"].named_steps["tfidf"].vocabulary_
    for left, right in zip(
        bundle["pipeline"].named_steps["classifier"].estimators_,
        checkpoint["pipeline"].named_steps["classifier"].estimators_,
    ):
        np.testing.assert_array_equal(left.coef_, right.coef_)
    assert bundle["pipeline"].predict_proba(["A person attacked the victim with a sharp weapon."]).shape == (
        1,
        2,
    )
    assert result["test_evaluated"] is False and not (run_dir / "test.json").exists()
    assert verify_baseline(bundle["metadata"]["baseline_snapshot"])
    assert (directory / "ipc_model.joblib").read_bytes() == b"baseline sentinel: must remain unchanged"


def test_training_cannot_access_test_and_restores_earlier_best(prepared, tmp_path, monkeypatch):
    import legal_intelligence.epoch_training as module

    real_load, real_metrics = module.load_cases, module._epoch_metrics
    accessed, calls, live_models = [], 0, []
    real_partial = module.IncrementalOVRClassifier.partial_fit

    def guarded(db, split, **kwargs):
        accessed.append(split)
        assert split != "test", "Training/tuning must not access held-out records"
        return real_load(db, split, **kwargs)

    def controlled_metrics(*args):
        nonlocal calls
        metrics = real_metrics(*args)
        if calls % 2 == 1:
            metrics["binary_log_loss"] = [0.7, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9][calls // 2]
        calls += 1
        return metrics

    def observe_partial(self, *args):
        live_models[:] = [self]
        return real_partial(self, *args)

    monkeypatch.setattr(module, "load_cases", guarded)
    monkeypatch.setattr(module, "_epoch_metrics", controlled_metrics)
    monkeypatch.setattr(module.IncrementalOVRClassifier, "partial_fit", observe_partial)
    result = train_epochs(
        prepared[0].db_path,
        tmp_path,
        EpochConfig(
            epochs=12,
            patience=5,
            batch_size=4,
            min_label_support=2,
            min_df=1,
            max_features=500,
            thresholds=(0.3, 0.5),
        ),
    )
    run_dir = tmp_path / "experiments" / result["model_version"]
    bundle = load_bundle(run_dir / "model.joblib", "ipc_classifier")
    assert result["epochs_completed"] == 7 and result["best_epoch"] == 2
    assert bundle["metadata"]["stopped_early"] is True
    assert accessed == ["train", "validation"]
    best = load_bundle(run_dir / "best_checkpoint.joblib", "ipc_classifier")
    for final, saved in zip(
        bundle["pipeline"].named_steps["classifier"].estimators_,
        best["pipeline"].named_steps["classifier"].estimators_,
    ):
        np.testing.assert_array_equal(final.coef_, saved.coef_)
    assert any(
        not np.array_equal(final.coef_, last.coef_)
        for final, last in zip(
            bundle["pipeline"].named_steps["classifier"].estimators_, live_models[0].estimators_
        )
    )


def test_final_test_evaluation_is_single_and_reports_unseen_labels(epoch_run, prepared):
    result, directory = epoch_run
    run_dir = directory / "experiments" / result["model_version"]
    pointer = run_dir.parent / "latest.json"
    pointer.write_text(json.dumps(result), encoding="utf-8")
    report = evaluate_epochs(prepared[0].db_path, run_dir)
    assert report["evaluation_count"] == 1 and report["metrics"]["documents"] == 4
    assert "Section 395 in The Indian Penal Code" in report["test_only_labels"]
    assert report["baseline_unchanged"] is True
    assert json.loads(pointer.read_text())["test_evaluated"] is True
    with pytest.raises(ValueError, match="already started/recorded"):
        evaluate_epochs(prepared[0].db_path, run_dir)


def test_epoch_graphs_use_existing_completed_history(epoch_run):
    pytest.importorskip("matplotlib")
    result, directory = epoch_run
    paths = plot_epochs(directory / "experiments" / result["model_version"])
    assert len(paths) == 8
    assert all(Path(path).stat().st_size > 1000 for path in paths)


def test_seed_reproduces_epoch_orders_and_model_weights(prepared, tmp_path):
    config = EpochConfig(
        epochs=2,
        patience=5,
        batch_size=4,
        min_label_support=2,
        min_df=1,
        max_features=500,
        thresholds=(0.3, 0.5),
    )
    bundles, histories = [], []
    for name in ("first", "second"):
        result = train_epochs(prepared[0].db_path, tmp_path / name, config)
        directory = Path(result["experiment_dir"])
        bundles.append(load_bundle(directory / "model.joblib", "ipc_classifier"))
        histories.append(json.loads((directory / "epoch_history.json").read_text())["epochs"])
    assert [row["order_sha256"] for row in histories[0]] == [row["order_sha256"] for row in histories[1]]
    assert bundles[0]["metadata"]["feature_signature"] == bundles[1]["metadata"]["feature_signature"]
    for left, right in zip(
        bundles[0]["pipeline"].named_steps["classifier"].estimators_,
        bundles[1]["pipeline"].named_steps["classifier"].estimators_,
    ):
        np.testing.assert_array_equal(left.coef_, right.coef_)
