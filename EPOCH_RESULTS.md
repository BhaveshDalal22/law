# CPU epoch experiment — actual results, 2 November 2026

Experiment **`ipc-sgd-epochs-7f670e8e6072`** was implemented and run on the real LeSICiN dataset. It completed **19 epochs**, stopped after five consecutive epochs without a new lowest validation loss, and restored **epoch 14**. No epoch history was reconstructed for the earlier baseline.

## Open the graphs and logs

- [Combined epoch overview](artifacts/experiments/ipc-sgd-epochs-7f670e8e6072/figures/epoch_overview.png)
- [Training and validation loss versus epoch](artifacts/experiments/ipc-sgd-epochs-7f670e8e6072/figures/loss_vs_epoch.png)
- [Micro and macro F1 versus epoch](artifacts/experiments/ipc-sgd-epochs-7f670e8e6072/figures/f1_vs_epoch.png)
- [Exact-match subset accuracy versus epoch](artifacts/experiments/ipc-sgd-epochs-7f670e8e6072/figures/subset_accuracy_vs_epoch.png)
- [All measured epochs — CSV](artifacts/experiments/ipc-sgd-epochs-7f670e8e6072/epoch_metrics.csv)
- [All measured epochs and metric definitions — JSON](artifacts/experiments/ipc-sgd-epochs-7f670e8e6072/epoch_history.json)

SVG versions of every graph are in the same figures folder. Graphs contain actual training and validation observations at fixed threshold **0.5**. No held-out test points appear in these curves.

## Data and training configuration

| Setting | Actual value |
| --- | --- |
| Eligible training records | 35,192 |
| Eligible validation records | 8,250 |
| Eligible final test records | 10,400 |
| Supported labels | 100 original IPC section identities |
| Features | Training-only TF-IDF, word unigrams/bigrams, 30,000 features, min_df=2 |
| Binary classifiers | One independent `SGDClassifier(loss="log_loss")` per label |
| Optimization | Public `partial_fit` on real minibatches, L2, alpha=0.00001, optimal learning-rate schedule |
| Batch size | 1,024; final smaller batch included |
| Seed | 42 |
| CPU resources | One worker, BLAS limited to one thread; no GPU |
| Maximum epochs | 20 |
| Completed epochs | **19** |
| Selected epoch | **14**, strictly lowest validation loss |
| Patience | Five consecutive epochs without improvement |
| Epoch classification threshold | 0.5 throughout |
| Final validation-selected threshold | **0.25** |
| Recorded training and selection time | 131.07 seconds; excludes final test evaluation and graph export |

The original supplied splits and preparation exclusions were reused. Every epoch visited all 35,192 eligible training records exactly once. TF-IDF was fitted once on training text; vocabulary and IDF weights remained fixed. Label encoding was fitted once on training labels. All 100 labels passed the minimum support and non-constant checks. No validation-only/test-only labels were found for this run.

Input selection and provision masking reuse the original baseline's rules. These are conservative narrative-selection rules, not a verified rhetorical-role model. Dataset section texts are historical excerpts with retained provenance; they are not asserted to be current law. No IPC identities were translated into BNS identities.

## Selected epoch, before threshold tuning

These metrics describe **epoch 14 at threshold 0.5**, as shown in the epoch graphs:

| Metric | Training | Validation |
| --- | ---: | ---: |
| Mean binary log loss | 0.065239 | **0.107522** |
| Micro precision | 0.904744 | 0.684304 |
| Micro recall | 0.382579 | 0.212925 |
| Micro F1 | 0.537761 | 0.324790 |
| Macro F1 | 0.332270 | 0.164664 |
| Exact-match subset accuracy | 0.147534 | 0.049697 |

Mean binary log loss is unregularized binary cross-entropy averaged over all supported document-label pairs, computed stably from logits. Lower is better. Checkpoint selection used this validation loss only; it did not select the largest F1 value. The saved epoch-14 weights were restored rather than using epoch 19.

Subset accuracy requires the **entire predicted label set** to equal the entire gold set for a document. It is not the fraction of individual binary label decisions that are correct.

## Validation-only threshold selection

After restoring the best checkpoint, the frozen model's validation scores were used to compare:

| Threshold | Validation micro F1 |
| --- | ---: |
| 0.15 | 0.444928 |
| **0.25** | **0.447249** |
| 0.35 | 0.407896 |
| 0.50 | 0.324790 |

No test data was read for checkpoint or threshold selection. The epoch curves remain at threshold 0.5; the following final evaluation uses the selected threshold 0.25.

## Final evaluation of the selected model

The experiment was evaluated **once** on 10,400 eligible test records after checkpoint and threshold selection were complete. The same fixed test split had previously been used to report the separate baseline's results; it was not read during this epoch experiment's training or selection.

| Metric | Validation at selected threshold | Held-out test at selected threshold |
| --- | ---: | ---: |
| Micro precision | 0.493907 | 0.491766 |
| Micro recall | 0.408645 | 0.398462 |
| Micro F1 | 0.447249 | **0.440225** |
| Macro F1 | 0.269156 | **0.257961** |
| Exact-match subset accuracy | 0.041091 | **0.039712** |
| Mean binary log loss | 0.107522 | 0.108380 |

Test exact-match subset accuracy is **3.97%**. Supported gold-label occurrence coverage is 1.0; that indicates supported-label coverage, not predictive accuracy. Full per-label precision, recall, F1 and support, plus top-k metrics and definitions, are in [test.json](artifacts/experiments/ipc-sgd-epochs-7f670e8e6072/test.json).

The earlier Logistic Regression baseline retains its recorded test micro F1 **0.439381** and macro F1 **0.264110**. The epoch experiment has slightly higher micro F1 and lower macro F1. No overall improvement or statistical significance is claimed. The original baseline remains the application's active model.

## Saved model and verification

The experiment directory is `artifacts/experiments/ipc-sgd-epochs-7f670e8e6072/`. It contains the restored `model.joblib`, `best_checkpoint.joblib`, integrity sidecars, label encoder, label order, thresholds, preprocessing, dataset checksums/provenance, configuration, package versions, and separate validation/test reports. Model scores are not calibrated confidence.

Completed verification:

- **35 tests passed**, including the original baseline tests and nine epoch-experiment tests.
- Ruff checks and formatting passed; dependency checking found no broken requirements.
- All 19 CSV rows match the corresponding JSON measurements. All 19 epochs contain exactly 35,192 training examples, and their seeded shuffle-order hashes are distinct.
- The model's vocabulary/IDF signature and label order match the recorded frozen signatures.
- Final classifier weights exactly match the saved best checkpoint. Every one of the 100 binary models records **492,688 training-example updates**, equal to 14 complete passes, confirming restoration of epoch 14.
- Original baseline model, metadata, evaluation reports and figures match their saved pre-experiment SHA-256 hashes.
- A persistent exclusive evaluation marker and test report record the single completed evaluation; repeated evaluation of this experiment is refused.
- Verification read the saved model/history/reports and did not read or re-score held-out examples.

Additional evidence is in [verification.json](artifacts/experiments/ipc-sgd-epochs-7f670e8e6072/verification.json) and [baseline_snapshot.json](artifacts/experiments/ipc-sgd-epochs-7f670e8e6072/baseline_snapshot.json). Tests use isolated artificial fixtures only; those measurements are not included in this report.

The test environment emits one upstream Starlette TestClient/httpx deprecation warning; checks passed. No baseline retraining, replacement or result modification was performed.

## Commands

To regenerate graphs from the completed run without training or evaluating again:

```powershell
.\.venv\Scripts\python.exe -m legal_intelligence.cli plot-epochs artifacts\experiments\ipc-sgd-epochs-7f670e8e6072
```

To start a **new** reproducible experiment and then evaluate its selected checkpoint once:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[reports]"
.\.venv\Scripts\python.exe -m legal_intelligence.cli train-epochs --epochs 20 --patience 5 --seed 42
.\.venv\Scripts\python.exe -m legal_intelligence.cli evaluate-epochs
```

Every training invocation creates a new experiment version. Validation is reused for checkpoint and threshold selection. Repeatedly trying configurations after inspecting test results would compromise the test set's role; these recorded results were not used for further tuning. Existing dataset noise, approximate related-case screening, class imbalance, missing statute dates, and uncalibrated scores remain limitations described in the README.
