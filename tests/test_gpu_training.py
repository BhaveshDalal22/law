"""Tiny artificial fixtures are isolated in temporary test folders, never production training."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer

from legal_intelligence.gpu_training import (
    GPUConfig,
    _restore_checkpoint,
    _save_checkpoint,
    encode_records,
    evaluate_gpu,
    score_model,
    train_gpu,
)


@pytest.fixture
def tiny_model(tmp_path):
    words = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "person",
        "attacked",
        "victim",
        "legal",
        "_",
        "provision",
        "section",
        "302",
        "ipc",
        "a",
        "with",
        "weapon",
    ]
    tokenizer = DistilBertTokenizer(vocab={word: i for i, word in enumerate(words)})
    model = DistilBertForSequenceClassification(
        DistilBertConfig(
            vocab_size=len(words),
            dim=24,
            hidden_dim=48,
            n_layers=1,
            n_heads=2,
            max_position_embeddings=64,
            num_labels=2,
            dropout=0.0,
            attention_dropout=0.0,
            id2label={0: "a", 1: "b"},
            label2id={"a": 0, "b": 1},
            problem_type="multi_label_classification",
            attn_implementation="eager",
        )
    )
    base = tmp_path / "tiny-artificial-model-only"
    model.save_pretrained(base, safe_serialization=True)
    tokenizer.save_pretrained(base)
    return model, tokenizer, base


def test_encoding_masks_targets_keeps_special_tokens_and_counts_truncation(tiny_model):
    _, tokenizer, _ = tiny_model
    before = tokenizer.get_vocab().copy()
    rows = [{"facts_text": "A person attacked victim. Section 302 IPC."}, {"facts_text": "person " * 50}]
    ids, lengths, report = encode_records(rows, tokenizer, 16)
    assert ids.shape == (2, 16)
    assert report["truncated_documents"] == 1
    assert tokenizer.get_vocab() == before
    assert tokenizer.convert_tokens_to_ids("302") not in ids[0].tolist()
    assert tokenizer.convert_tokens_to_ids("section") not in ids[0].tolist()
    for row, length in zip(ids, lengths):
        assert row[0] == tokenizer.cls_token_id and row[length - 1] == tokenizer.sep_token_id


def test_checkpoint_restoration_scoring_and_corruption_detection(tiny_model, tmp_path):
    model, tokenizer, _ = tiny_model
    ids, lengths, _ = encode_records([{"facts_text": "person attacked victim"}], tokenizer, 16)
    scores = score_model(model, ids, lengths, tokenizer.pad_token_id, 2, torch.float32, device="cpu")
    path = tmp_path / "checkpoint"
    _save_checkpoint(model, path, 2, 0.2)
    with torch.no_grad():
        model.classifier.weight.add_(10)
    restored, metadata = _restore_checkpoint(path, ["a", "b"], "cpu")
    actual = score_model(restored, ids, lengths, tokenizer.pad_token_id, 1, torch.float32, device="cpu")
    np.testing.assert_array_equal(scores, actual)
    assert metadata["epoch"] == 2
    with pytest.raises(ValueError, match="label order"):
        _restore_checkpoint(path, ["b", "a"], "cpu")
    with (path / "model.safetensors").open("ab") as stream:
        stream.write(b"corrupted test fixture")
    with pytest.raises(ValueError, match="checksum"):
        _restore_checkpoint(path, ["a", "b"], "cpu")


def test_gpu_command_rejects_unavailable_cuda_without_reading_data(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        train_gpu(tmp_path / "absent.sqlite3", tmp_path)
    assert not (tmp_path / "absent.sqlite3").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA integration check needs a real GPU")
def test_cuda_training_never_reads_test_restores_best_and_evaluates_once(
    prepared, tiny_model, tmp_path, monkeypatch
):
    import legal_intelligence.gpu_training as module

    _, _, base = tiny_model
    monkeypatch.setattr(module, "_base_snapshot", lambda config: (base, {"test_fixture_only": True}))
    original_load, original_metrics = module.load_cases, module._metrics
    accessed, calls = [], 0

    def guarded(db, split, **kwargs):
        accessed.append(split)
        assert split != "test"
        return original_load(db, split, **kwargs)

    def controlled(target, logits, threshold=0.5):
        nonlocal calls
        result = original_metrics(target, logits, threshold)
        if calls < 6 and calls % 2 == 1:
            result["binary_log_loss"] = [0.4, 0.2, 0.3][calls // 2]
        calls += 1
        return result

    monkeypatch.setattr(module, "load_cases", guarded)
    monkeypatch.setattr(module, "_metrics", controlled)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "ipc_model.joblib").write_bytes(b"CPU baseline sentinel")
    result = train_gpu(
        prepared[0].db_path,
        artifacts,
        GPUConfig(
            epochs=5,
            patience=1,
            batch_size=4,
            accumulation_steps=2,
            eval_batch_size=4,
            max_length=32,
            min_label_support=2,
        ),
    )
    assert accessed == ["train", "validation"]
    assert result["epochs_completed"] == 3 and result["best_epoch"] == 2
    directory = Path(result["experiment_dir"])
    history = json.loads((directory / "epoch_history.json").read_text())["epochs"]
    assert all(row["examples_seen"] == 10 and row["optimizer_updates"] == 2 for row in history)
    assert not (directory / "test.json").exists()
    monkeypatch.setattr(module, "load_cases", original_load)
    evaluation = evaluate_gpu(prepared[0].db_path, directory)
    assert evaluation["best_epoch"] == 2 and evaluation["evaluation_count"] == 1
    assert evaluation["metrics"]["documents"] == 4 and evaluation["baseline_unchanged"]
    with pytest.raises(ValueError, match="will not be repeated"):
        evaluate_gpu(prepared[0].db_path, directory)
    assert (artifacts / "ipc_model.joblib").read_bytes() == b"CPU baseline sentinel"
