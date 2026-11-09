import numpy as np
import pytest
from conftest import RARE, UNSEEN

from legal_intelligence.artifacts import load_bundle, save_bundle
from legal_intelligence.evaluation import evaluate_labels
from legal_intelligence.training import TrainConfig, final_evaluate, predict_scores


def test_label_alignment_and_save_load(trained, tmp_path):
    settings, report = trained
    bundle = load_bundle(settings.model_path, "ipc_classifier")
    assert bundle["labels"] == bundle["label_encoder"].classes_[bundle["supported_indices"]].tolist()
    assert RARE in report["metadata"]["excluded_labels"] and UNSEEN not in bundle["label_encoder"].classes_
    text = ["The witness described an attack with a sharp weapon."]
    before = predict_scores(bundle["pipeline"], text, bundle["labels"])
    path = tmp_path / "copy.joblib"
    save_bundle(bundle, path)
    after = predict_scores(load_bundle(path, "ipc_classifier")["pipeline"], text, bundle["labels"])
    np.testing.assert_array_equal(before, after)
    vocabulary = bundle["pipeline"].named_steps["tfidf"].vocabulary_
    assert "khaki" not in vocabulary  # validation-only token cannot enter training vocabulary
    assert "section 34" not in vocabulary


def test_training_never_reads_test_split(prepared, monkeypatch, tmp_path):
    import legal_intelligence.training as module

    real = module.load_cases

    def guarded(db, split, **kwargs):
        assert split != "test", "Training must never inspect test examples"
        return real(db, split, **kwargs)

    monkeypatch.setattr(module, "load_cases", guarded)
    module.train(
        prepared[0].db_path,
        tmp_path / "model.joblib",
        tmp_path / "reports",
        TrainConfig(min_label_support=2, min_df=1, max_features=500, c_values=(1.0,), thresholds=(0.5,)),
    )


def test_test_evaluation_reports_unseen_labels_once(trained):
    settings, _ = trained
    result = final_evaluate(settings.db_path, settings.model_path, settings.artifact_dir / "reports")
    assert UNSEEN in result["test_only_labels"]
    assert result["metrics"]["per_label"][UNSEEN]["recall"] == 0
    with pytest.raises(ValueError, match="already recorded"):
        final_evaluate(settings.db_path, settings.model_path, settings.artifact_dir / "reports")


def test_metric_universe_counts_unsupported_gold_as_false_negative():
    result = evaluate_labels([["a", "unseen"]], np.array([[0.9]]), ["a"], [0.5])
    assert result["micro_recall"] == 0.5
    assert result["macro_f1"] == 0.5
    assert result["top_k"]["1"]["recall_at_k"] == 0.5


def test_single_label_metrics_do_not_count_true_negatives_as_success():
    result = evaluate_labels([["a"], []], np.array([[0.9], [0.7]]), ["a"], [0.5])
    assert result["micro_precision"] == 0.5 and result["micro_recall"] == 1
    assert result["macro_f1"] == pytest.approx(2 / 3)
