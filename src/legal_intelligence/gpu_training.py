"""Separate CUDA DistilBERT experiment; training never reads the held-out split."""

import hashlib
import importlib.metadata
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi, snapshot_download
from scipy.special import expit
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from legal_intelligence.artifacts import write_json
from legal_intelligence.data.store import connect, load_cases, load_metadata
from legal_intelligence.epoch_training import (
    DEFINITIONS,
    EarlyStopping,
    _baseline_snapshot,
    _persist_history,
    classification_metrics,
    mean_binary_log_loss,
    metric_target,
    verify_baseline,
)
from legal_intelligence.evaluation import evaluate_labels
from legal_intelligence.preprocessing import DECISIVE_RE, OUTCOME_RE, PROVISION_RE, preprocess_texts
from legal_intelligence.training import package_versions

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@dataclass(frozen=True)
class GPUConfig:
    epochs: int = 3
    patience: int = 2
    seed: int = 42
    batch_size: int = 16
    eval_batch_size: int = 64
    accumulation_steps: int = 2
    max_length: int = 256
    learning_rate: float = 0.00003
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    min_label_support: int = 5
    model_id: str = "distilbert/distilbert-base-uncased"
    revision: str = "main"
    thresholds: tuple = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)

    def validate(self):
        for name in (
            "epochs",
            "patience",
            "batch_size",
            "eval_batch_size",
            "accumulation_steps",
            "min_label_support",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if not 8 <= self.max_length <= 512:
            raise ValueError("max-length must be between 8 and 512.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning-rate must be finite and positive.")
        if not 0 <= self.warmup_ratio < 1 or not 0 <= self.weight_decay < 1:
            raise ValueError("Invalid warmup ratio or weight decay.")
        if not self.thresholds or any(not 0 < value < 1 for value in self.thresholds):
            raise ValueError("Threshold candidates must be between zero and one.")
        if self.model_id != "distilbert/distilbert-base-uncased":
            raise ValueError("This audited experiment supports distilbert/distilbert-base-uncased only.")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _existing_experiments_snapshot(artifact_dir):
    snapshot = _baseline_snapshot(artifact_dir)
    # Also preserve both existing CPU epoch versions, their reports and latest pointer.
    root = artifact_dir / "experiments"
    for path in root.rglob("*"):
        if path.is_file() and (path.name == "latest.json" or "ipc-sgd-epochs-" in str(path)):
            snapshot[str(path.resolve())] = _sha(path)
    return snapshot


def _cuda_setup(seed):
    if not torch.cuda.is_available():
        raise ValueError(
            "CUDA is unavailable. Install the CUDA PyTorch wheel from the documented index; "
            "this command will not silently substitute CPU training."
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return dtype


def _base_snapshot(config):
    info = HfApi(token=False).model_info(config.model_id, revision=config.revision)
    card = info.card_data.to_dict() if info.card_data else {}
    if card.get("license") != "apache-2.0":
        raise ValueError("Base-model licence changed or is missing; inspect before proceeding.")
    directory = Path(
        snapshot_download(
            config.model_id,
            revision=info.sha,
            token=False,
            allow_patterns=[
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.txt",
                "special_tokens_map.json",
                "model.safetensors",
                "README.md",
                "LICENSE",
            ],
            max_workers=2,
        )
    )
    if not (directory / "model.safetensors").is_file():
        raise ValueError("Pinned safetensors base weights are missing.")
    provenance = {
        "repository": config.model_id,
        "revision": info.sha,
        "url": f"https://huggingface.co/{config.model_id}/tree/{info.sha}",
        "declared_license": card["license"],
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "files_sha256": {path.name: _sha(path) for path in directory.iterdir() if path.is_file()},
        "initialization": "Public pretrained DistilBERT encoder; randomly initialized IPC classification head.",
        "pretraining_data": "Publisher reports English Wikipedia and BookCorpus; not pretrained by this project.",
    }
    return directory, provenance


def encode_records(rows, tokenizer, max_length):
    """Bounded tokenization. Freeze public base vocabulary; never adapt it to validation/test."""
    encoded = np.full((len(rows), max_length), tokenizer.pad_token_id, dtype=np.int32)
    lengths = np.zeros(len(rows), dtype=np.int32)
    original_lengths = []
    special = tokenizer.num_special_tokens_to_add(pair=False)
    if special != 2 or tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("Expected the audited DistilBERT CLS/text/SEP tokenizer format.")
    for start in range(0, len(rows), 128):
        texts = preprocess_texts([row["facts_text"] for row in rows[start : start + 128]])
        tokenized = tokenizer(
            texts, add_special_tokens=False, truncation=False, return_attention_mask=False, verbose=False
        )
        for offset, ids in enumerate(tokenized["input_ids"]):
            original_lengths.append(len(ids) + special)
            ids = [tokenizer.cls_token_id] + ids[: max_length - special] + [tokenizer.sep_token_id]
            encoded[start + offset, : len(ids)] = ids
            lengths[start + offset] = len(ids)
    return (
        torch.from_numpy(encoded),
        torch.from_numpy(lengths),
        {
            "documents": len(rows),
            "truncated_documents": int(np.count_nonzero(np.asarray(original_lengths) > max_length)),
            "mean_original_tokens": float(np.mean(original_lengths)),
            "max_original_tokens": int(max(original_lengths)),
            "policy": "First max_length tokens including special tokens; later narrative tokens are omitted.",
        },
    )


def _inputs(ids, lengths, pad_id, device):
    # Trim padding per batch and align to eight tokens for tensor cores.
    width = min(ids.shape[1], math.ceil(int(lengths.max()) / 8) * 8)
    ids = ids[:, :width].to(device=device, dtype=torch.long)
    return {"input_ids": ids, "attention_mask": (ids != pad_id).long()}


def score_model(model, ids, lengths, pad_id, batch_size, dtype, device="cuda"):
    model.eval()
    parts = []
    loader = DataLoader(TensorDataset(ids, lengths), batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.inference_mode():
        for batch_ids, batch_lengths in loader:
            with torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
                logits = model(**_inputs(batch_ids, batch_lengths, pad_id, device)).logits
            parts.append(logits.float().cpu().numpy())
    logits = np.concatenate(parts)
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite model scores; no test tuning will be performed.")
    return logits


def _metrics(target, logits, threshold=0.5):
    result = classification_metrics(target, expit(logits), threshold)
    result["binary_log_loss"] = mean_binary_log_loss(target.truth[:, : target.supported_count], logits)
    return result


def _save_checkpoint(model, path, epoch, loss):
    model.save_pretrained(path, safe_serialization=True)
    write_json(
        path / "checkpoint.json",
        {
            "epoch": epoch,
            "validation_binary_log_loss": loss,
            "model_sha256": _sha(path / "model.safetensors"),
        },
    )


def _restore_checkpoint(path, labels, device):
    checkpoint = _read_json(path / "checkpoint.json")
    if _sha(path / "model.safetensors") != checkpoint["model_sha256"]:
        raise ValueError("GPU checkpoint checksum mismatch.")
    model = AutoModelForSequenceClassification.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        attn_implementation="eager",
    )
    if [model.config.id2label[i] for i in range(len(labels))] != list(labels):
        raise ValueError("Saved transformer label order does not match its metadata.")
    return model.to(device), checkpoint


def train_gpu(db_path, artifact_dir, config=None):
    config = config or GPUConfig()
    config.validate()
    dtype = _cuda_setup(config.seed)
    started = time.perf_counter()
    artifact_dir = Path(artifact_dir)
    baseline = _existing_experiments_snapshot(artifact_dir)
    with connect(db_path) as db:
        fields = ("case_id", "facts_text", "act_section_labels")
        rows = load_cases(db, "train", fields=fields)
        validation = load_cases(db, "validation", fields=fields)
        provenance = load_metadata(db, "provenance")
        preparation = load_metadata(db, "preparation_report")
    if not rows or not validation or not provenance or not preparation:
        raise ValueError("Prepared LeSICiN training/validation data and audit are required.")
    encoder = MultiLabelBinarizer()
    all_y = encoder.fit_transform([row["act_section_labels"] for row in rows])
    support = all_y.sum(axis=0)
    keep = np.flatnonzero((support >= config.min_label_support) & (support < len(rows)))
    labels = encoder.classes_[keep].tolist()
    if not labels:
        raise ValueError("No non-constant training labels meet minimum support.")
    train_target = metric_target([row["act_section_labels"] for row in rows], labels)
    validation_gold = [row["act_section_labels"] for row in validation]
    validation_target = metric_target(validation_gold, labels)
    y = torch.from_numpy(all_y[:, keep].astype(np.float32))
    del all_y
    stamp = datetime.now(UTC).isoformat()
    identity = stamp + json.dumps(asdict(config), sort_keys=True) + json.dumps(provenance, sort_keys=True)
    version = "ipc-distilbert-gpu-" + hashlib.sha256(identity.encode()).hexdigest()[:12]
    run_dir = artifact_dir / "gpu_experiments" / version
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "status.json", {"status": "preparing", "model_version": version})
    write_json(run_dir / "config.json", asdict(config))
    write_json(run_dir / "baseline_snapshot.json", baseline)
    print(f"GPU experiment: {run_dir}", flush=True)
    base_dir, base_provenance = _base_snapshot(config)
    tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True, trust_remote_code=False)
    tokenizer.save_pretrained(run_dir / "tokenizer")
    train_ids, train_lengths, train_truncation = encode_records(rows, tokenizer, config.max_length)
    val_ids, val_lengths, val_truncation = encode_records(validation, tokenizer, config.max_length)
    metadata = {
        "model_version": version,
        "experiment": "distilbert_multilabel_cuda",
        "started_at": stamp,
        "config": asdict(config),
        "device": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "precision": str(dtype),
        "deterministic_algorithms": True,
        "reproducibility_limit": "Seeded repeatability on this software/hardware; cross-platform bitwise equality is not guaranteed.",
        "packages": {
            **package_versions(),
            **{
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "tokenizers", "safetensors", "huggingface_hub")
            },
        },
        "training_documents": len(rows),
        "validation_documents": len(validation),
        "supported_labels": labels,
        "training_encoder_classes": encoder.classes_.tolist(),
        "training_label_support": {str(label): int(support[i]) for i, label in enumerate(encoder.classes_)},
        "label_order_sha256": hashlib.sha256(json.dumps(labels).encode()).hexdigest(),
        "dataset_provenance": provenance,
        "base_model_provenance": base_provenance,
        "preparation_report_sha256": hashlib.sha256(
            json.dumps(preparation, sort_keys=True).encode()
        ).hexdigest(),
        "training_case_ids_sha256": hashlib.sha256(
            "\n".join(r["case_id"] for r in rows).encode()
        ).hexdigest(),
        "validation_case_ids_sha256": hashlib.sha256(
            "\n".join(r["case_id"] for r in validation).encode()
        ).hexdigest(),
        "preprocessing": {
            "version": 1,
            "facts_selection": preparation["facts_policy"],
            "reference_regex": PROVISION_RE.pattern,
            "reference_flags": PROVISION_RE.flags,
            "outcome_regex": OUTCOME_RE.pattern,
            "decisive_regex": DECISIVE_RE.pattern,
            "unicode_normalization": "NFKC",
            "tokenizer_policy": "Frozen public pretrained vocabulary; no vocabulary fitting on any task split.",
        },
        "tokenization": {"train": train_truncation, "validation": val_truncation},
        "epoch_metric_threshold": 0.5,
        "definitions": DEFINITIONS,
        "optimizer": "AdamW; unweighted mean BCEWithLogitsLoss; linear decay with warmup; gradient norm clipped to 1.0.",
        "baseline_snapshot": baseline,
        "test_evaluated_during_training": False,
        "calibrated": False,
        "statute_version": None,
        "limitations": [
            "Historical IPC labels only; no BNS mapping or current-law verification.",
            "Rule-selected narratives and labels may be noisy.",
            "Head truncation can omit relevant facts; CPU TF-IDF sees longer narratives.",
            "Validation is reused for checkpoint and threshold selection.",
            "Fine-tuned classifier; not a conversational or generative question-answering model.",
            "Pretrained encoder corpus overlap with legal cases was not independently audited.",
        ],
    }
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "labels.json", labels)
    del rows, validation
    model = AutoModelForSequenceClassification.from_pretrained(
        base_dir,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        num_labels=len(labels),
        problem_type="multi_label_classification",
        id2label=dict(enumerate(labels)),
        label2id={label: i for i, label in enumerate(labels)},
        attn_implementation="eager",
    ).to("cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, fused=True
    )
    batches_per_epoch = math.ceil(len(train_ids) / config.batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / config.accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(updates_per_epoch * config.epochs * config.warmup_ratio),
        num_training_steps=updates_per_epoch * config.epochs,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    rng = np.random.default_rng(config.seed)
    stopping, history = EarlyStopping(config.patience), []
    train_set = TensorDataset(train_ids, train_lengths, y)
    pad_id = tokenizer.pad_token_id
    print(
        f"Training on CUDA: {metadata['device']}; {len(train_ids)} train, {len(val_ids)} validation, "
        f"{len(labels)} IPC labels; max length {config.max_length}; {dtype}.",
        flush=True,
    )
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        order = rng.permutation(len(train_ids))
        loader = DataLoader(train_set, batch_size=config.batch_size, sampler=order.tolist(), num_workers=0)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        examples_seen, updates = 0, 0
        for step, (batch_ids, batch_lengths, batch_y) in enumerate(loader):
            group_start = (step // config.accumulation_steps) * config.accumulation_steps
            group_end = min(group_start + config.accumulation_steps, batches_per_epoch)
            group_examples = (
                min(group_end * config.batch_size, len(train_ids)) - group_start * config.batch_size
            )
            with torch.autocast("cuda", dtype=dtype):
                logits = model(**_inputs(batch_ids, batch_lengths, pad_id, "cuda")).logits
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.float(), batch_y.to("cuda")
                )
                loss = loss * len(batch_ids) / group_examples
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; experiment stopped before test access.")
            scaler.scale(loss).backward()
            examples_seen += len(batch_ids)
            if (step + 1) % config.accumulation_steps == 0 or step + 1 == batches_per_epoch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
            if step == 0 or (step + 1) % 200 == 0:
                print(
                    f"Epoch {epoch}/{config.epochs}: {examples_seen}/{len(train_ids)} records "
                    f"({time.perf_counter() - epoch_start:.0f}s)",
                    flush=True,
                )
                write_json(
                    run_dir / "status.json",
                    {
                        "status": "training",
                        "epoch": epoch,
                        "examples_seen": examples_seen,
                        "training_documents": len(train_ids),
                        "epochs_completed": len(history),
                        "model_version": version,
                    },
                )
        if examples_seen != len(train_ids) or updates != updates_per_epoch:
            raise ValueError("Incomplete epoch or optimizer update count mismatch.")
        train_logits = score_model(model, train_ids, train_lengths, pad_id, config.eval_batch_size, dtype)
        val_logits = score_model(model, val_ids, val_lengths, pad_id, config.eval_batch_size, dtype)
        train_metrics, val_metrics = (
            _metrics(train_target, train_logits),
            _metrics(validation_target, val_logits),
        )
        improved, should_stop = stopping.update(epoch, val_metrics["binary_log_loss"])
        record = {
            "epoch": epoch,
            "examples_seen": examples_seen,
            "optimizer_updates": updates,
            "order_sha256": hashlib.sha256(order.astype("<i8").tobytes()).hexdigest(),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in val_metrics.items()},
            "best_checkpoint": improved,
            "seconds": time.perf_counter() - epoch_start,
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1048576,
        }
        if improved:
            _save_checkpoint(model, run_dir / "best_checkpoint", epoch, stopping.best_loss)
        history.append(record)
        _persist_history(run_dir, metadata, history, "running")
        print(
            f"Completed epoch {epoch}: train loss={train_metrics['binary_log_loss']:.6f}, "
            f"validation loss={val_metrics['binary_log_loss']:.6f}, "
            f"train F1={train_metrics['micro_f1']:.4f}, validation F1={val_metrics['micro_f1']:.4f}"
            + (
                " [best checkpoint]" if improved else f" [patience {stopping.stale_epochs}/{config.patience}]"
            ),
            flush=True,
        )
        if should_stop:
            break
    del model, optimizer, scheduler, train_logits
    torch.cuda.empty_cache()
    model, checkpoint = _restore_checkpoint(run_dir / "best_checkpoint", labels, "cuda")
    val_logits = score_model(model, val_ids, val_lengths, pad_id, config.eval_batch_size, dtype)
    if checkpoint["epoch"] != stopping.best_epoch:
        raise ValueError("Restored checkpoint is not the selected epoch.")
    scores = expit(val_logits)
    search = [
        {
            "threshold": threshold,
            "validation_micro_f1": classification_metrics(validation_target, scores, threshold)["micro_f1"],
        }
        for threshold in config.thresholds
    ]
    selected = max(search, key=lambda item: item["validation_micro_f1"])
    thresholds = [selected["threshold"]] * len(labels)
    metadata.update(
        {
            "epochs_completed": len(history),
            "best_epoch": stopping.best_epoch,
            "best_validation_loss": stopping.best_loss,
            "selected_threshold": selected["threshold"],
            "threshold_policy": "Global threshold maximizing validation micro F1 after restoring the lowest-loss checkpoint.",
            "stopped_early": stopping.stale_epochs >= config.patience,
            "completed_at": datetime.now(UTC).isoformat(),
            "seconds": time.perf_counter() - started,
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1048576,
            "baseline_unchanged": verify_baseline(baseline),
        }
    )
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "thresholds.json", thresholds)
    write_json(
        run_dir / "validation.json",
        {
            "model_version": version,
            "threshold_search": search,
            "selected_threshold": selected["threshold"],
            "metrics": evaluate_labels(validation_gold, scores, labels, thresholds),
            "subset_accuracy": classification_metrics(validation_target, scores, selected["threshold"])[
                "subset_accuracy"
            ],
            "binary_log_loss": _metrics(validation_target, val_logits)["binary_log_loss"],
        },
    )
    _persist_history(run_dir, metadata, history, "completed")
    write_json(
        run_dir / "integrity.json",
        {
            str(p.relative_to(run_dir)): _sha(p)
            for name in ("best_checkpoint", "tokenizer")
            for p in (run_dir / name).iterdir()
            if p.is_file()
        },
    )
    result = {
        "experiment_dir": str(run_dir),
        "model_version": version,
        "epochs_completed": len(history),
        "best_epoch": stopping.best_epoch,
        "best_validation_loss": stopping.best_loss,
        "selected_threshold": selected["threshold"],
        "validation_micro_f1": selected["validation_micro_f1"],
        "test_evaluated": False,
        "baseline_unchanged": True,
    }
    write_json(run_dir / "status.json", {**result, "status": "trained"})
    write_json(artifact_dir / "gpu_experiments" / "latest.json", result)
    return result


def evaluate_gpu(db_path, run_dir):
    run_dir = Path(run_dir)
    marker, output = run_dir / "test_evaluation.started.json", run_dir / "test.json"
    if marker.exists() or output.exists():
        raise ValueError(
            "Held-out evaluation already started/recorded; test evaluation will not be repeated."
        )
    metadata = _read_json(run_dir / "metadata.json")
    if (
        metadata.get("experiment") != "distilbert_multilabel_cuda"
        or not (run_dir / "validation.json").exists()
    ):
        raise ValueError("GPU validation checkpoint/threshold selection must finish first.")
    dtype = _cuda_setup(metadata["config"]["seed"])
    with connect(db_path) as db:
        provenance = load_metadata(db, "provenance")
        preparation = load_metadata(db, "preparation_report")
    if (
        provenance != metadata["dataset_provenance"]
        or hashlib.sha256(json.dumps(preparation, sort_keys=True).encode()).hexdigest()
        != metadata["preparation_report_sha256"]
    ):
        raise ValueError("Prepared dataset provenance/audit differs from the trained experiment.")
    for relative, digest in _read_json(run_dir / "integrity.json").items():
        if _sha(run_dir / relative) != digest:
            raise ValueError(f"Saved model/tokenizer checksum mismatch: {relative}")
    labels, thresholds = _read_json(run_dir / "labels.json"), _read_json(run_dir / "thresholds.json")
    if labels != metadata["supported_labels"] or thresholds != [metadata["selected_threshold"]] * len(labels):
        raise ValueError("Saved label order/thresholds changed after validation selection.")
    model, checkpoint = _restore_checkpoint(run_dir / "best_checkpoint", labels, "cuda")
    if checkpoint["epoch"] != metadata["best_epoch"]:
        raise ValueError("Restored checkpoint differs from validation selection.")
    tokenizer = AutoTokenizer.from_pretrained(
        run_dir / "tokenizer", local_files_only=True, trust_remote_code=False
    )
    with marker.open("x", encoding="utf-8") as stream:
        json.dump(
            {"model_version": metadata["model_version"], "started_at": datetime.now(UTC).isoformat()}, stream
        )
    with connect(db_path) as db:
        rows = load_cases(db, "test", fields=("case_id", "facts_text", "act_section_labels"))
    if not rows:
        raise ValueError("No eligible held-out examples.")
    ids, lengths, tokenization = encode_records(rows, tokenizer, metadata["config"]["max_length"])
    gold = [row["act_section_labels"] for row in rows]
    target = metric_target(gold, labels)
    logits = score_model(
        model, ids, lengths, tokenizer.pad_token_id, metadata["config"]["eval_batch_size"], dtype
    )
    scores = expit(logits)
    result = {
        "model_version": metadata["model_version"],
        "split": "test",
        "evaluation_count": 1,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "best_epoch": metadata["best_epoch"],
        "threshold": metadata["selected_threshold"],
        "metrics": evaluate_labels(gold, scores, labels, thresholds),
        "binary_log_loss": mean_binary_log_loss(target.truth[:, : target.supported_count], logits),
        "subset_accuracy": classification_metrics(target, scores, metadata["selected_threshold"])[
            "subset_accuracy"
        ],
        "tokenization": tokenization,
        "dataset_checksums": provenance["files"],
        "test_only_labels": sorted(set(target.universe) - set(metadata["training_encoder_classes"])),
        "baseline_unchanged": verify_baseline(metadata["baseline_snapshot"]),
    }
    write_json(output, result)
    pointer = run_dir.parent / "latest.json"
    if pointer.exists():
        latest = _read_json(pointer)
        if latest["model_version"] == metadata["model_version"]:
            latest.update({"test_evaluated": True, "test_report": str(output)})
            write_json(pointer, latest)
    write_json(
        run_dir / "status.json",
        {
            "status": "evaluated",
            "model_version": metadata["model_version"],
            "epochs_completed": metadata["epochs_completed"],
            "test_report": str(output),
        },
    )
    return result
