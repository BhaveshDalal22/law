# Recorded GPU experiment results

Actual measurements from fine-tuning a pretrained DistilBERT encoder on the prepared LeSICiN IPC task. The original Logistic Regression and CPU SGD epoch models are retained.

- Model version: `ipc-distilbert-gpu-8501c58af60f`.
- Device: NVIDIA GeForce RTX 3060; CUDA 12.8; precision torch.bfloat16.
- Completed **3 epochs** of the configured 3; selected **epoch 3** using minimum validation loss (0.116628).
- Early-stopping patience: 2; stopped early: False.
- Pipeline wall time: 1042.66 seconds (includes preparation, epoch scoring and checkpoint restoration; excludes final test evaluation).
- Peak PyTorch allocated GPU memory: 2572.7 MiB (does not include all driver/reserved memory).
- Training: 35,192; validation: 8,250; held-out test: 10,400; supported IPC labels: 100.
- Final global prediction threshold: **0.2**, selected using validation data only after restoring the best checkpoint.

## Held-out measurements

The selected GPU model was evaluated once. The prior CPU results below are read from their saved reports; their held-out evaluations were not repeated.

| Metric | GPU transformer validation | GPU transformer test | CPU Logistic Regression test | CPU SGD test |
| --- | ---: | ---: | ---: | ---: |
| Micro precision | 0.399474 | 0.399490 | 0.488597 | 0.491766 |
| Micro recall | 0.412468 | 0.406640 | 0.399172 | 0.398462 |
| Micro F1 | 0.405867 | 0.403033 | 0.439381 | 0.440225 |
| Macro F1 | 0.161013 | 0.158229 | 0.264110 | 0.257961 |
| Mean binary log loss | 0.116628 | 0.117033 | Not recorded | 0.108380 |
| Exact-match subset accuracy | 0.012606 | 0.013654 | Not recorded | 0.039712 |

Subset accuracy requires the entire predicted IPC label set to match the gold set. It is distinct from per-label accuracy. Loss is the mean unweighted binary cross-entropy over document-label pairs.

GPU test micro F1 is 3.63 percentage points lower than Logistic Regression. This is a measured comparison, not a significance test.

GPU test micro F1 is 3.72 percentage points lower than CPU SGD. This is a measured comparison, not a significance test.

## Measured epoch history

Every epoch is one complete seeded shuffled pass over all eligible training records. The following F1 and subset accuracy values use threshold 0.5. Epoch graphs contain no test measurements; final threshold-selected test scores must not be confused with these comparison curves.

| Epoch | Training loss | Validation loss | Training micro F1 | Validation micro F1 | Validation subset accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.121560 | 0.127936 | 0.100706 | 0.104348 | 0.017455 |
| 2 | 0.108227 | 0.118316 | 0.215975 | 0.196097 | 0.028242 |
| 3 | 0.104268 | 0.116628 | 0.264396 | 0.228645 | 0.034303 |

![Training and validation curves](artifacts/gpu_experiments/ipc-distilbert-gpu-8501c58af60f/figures/epoch_overview.png)

- [Loss graph](artifacts/gpu_experiments/ipc-distilbert-gpu-8501c58af60f/figures/loss_vs_epoch.png)
- [Micro and macro F1 graph](artifacts/gpu_experiments/ipc-distilbert-gpu-8501c58af60f/figures/f1_vs_epoch.png)
- [Exact-match accuracy graph](artifacts/gpu_experiments/ipc-distilbert-gpu-8501c58af60f/figures/subset_accuracy_vs_epoch.png)
- [Epoch CSV](artifacts/gpu_experiments/ipc-distilbert-gpu-8501c58af60f/epoch_metrics.csv)
- [Epoch JSON](artifacts/gpu_experiments/ipc-distilbert-gpu-8501c58af60f/epoch_history.json)
- [Full test metrics and per-label results](artifacts/gpu_experiments/ipc-distilbert-gpu-8501c58af60f/test.json)
- [Configuration and provenance](artifacts/gpu_experiments/ipc-distilbert-gpu-8501c58af60f/metadata.json)

## Configuration, provenance and limits

Base model: [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased/tree/12040accade4e8a0f71eabdb258fecc2e7e948be), commit `12040accade4e8a0f71eabdb258fecc2e7e948be`. The publisher declares Apache 2.0. The encoder was publicly pretrained; this project fine-tunes it and a newly initialized IPC classification head. LeSICiN data attribution and CC BY 4.0 provenance are recorded separately with original file hashes.

AdamW learning rate 3e-05, weight decay 0.01, batch size 16, accumulation 2, maximum sequence length 256, seed 42. The base tokenizer vocabulary is frozen, and the task label encoder uses only training labels. The CPU TF-IDF models see longer narratives, so this is not a hardware-only comparison.

- train: 17,142/35,192 documents (48.71%) were truncated to the sequence limit.
- validation: 4,214/8,250 documents (51.08%) were truncated to the sequence limit.
- test: 5,291/10,400 documents (50.88%) were truncated to the sequence limit.

Label suggestions remain historical IPC labels. This experiment does not establish BNS support, current statutory validity or conversational question-answering capability. Rule-selected narratives and dataset labels may be noisy. The pretrained corpus was not independently audited for overlap with these cases. Validation is reused for checkpoint and threshold selection. No test result was used to choose epochs, thresholds or hyperparameters.

Model and tokenizer files are stored with SHA-256 integrity checks. The original CPU artifacts were verified unchanged. The running application still serves its original CPU model.

## Commands

```powershell
.\.venv\Scripts\python.exe -m legal_intelligence.cli train-gpu --epochs 3 --patience 2 --seed 42 --revision 12040accade4e8a0f71eabdb258fecc2e7e948be
.\.venv\Scripts\python.exe -m legal_intelligence.cli evaluate-gpu
.\.venv\Scripts\python.exe -m legal_intelligence.cli plot-gpu
```

Training creates a new experiment version. Evaluation refuses a repeat for an already evaluated version. Plotting and this report script only read saved measurements.
