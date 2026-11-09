# AI-Based Legal Case Intelligence

A local Python application for historical Indian Penal Code (IPC) label suggestions, passage retrieval, extractive summaries, and regex reference extraction. FastAPI serves the analysis; Streamlit provides the interface. The application and original baselines use CPU. A separate optional CUDA experiment fine-tunes DistilBERT. No Ollama, external LLM API, or generative model is used.

## Quick start — Windows PowerShell

Python 3.12 is the tested interpreter. Run these commands from this project directory:

```powershell
cd C:\Users\bhave\OneDrive\Desktop\law
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

The workspace created during implementation already contains this environment. No activation or PowerShell execution-policy change is necessary. For a fresh installation using the declared compatible dependency ranges instead of the tested lock:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Build the real dataset, model, final evaluation, and separate retrieval index:

```powershell
.\.venv\Scripts\python.exe -m legal_intelligence.cli download
.\.venv\Scripts\python.exe -m legal_intelligence.cli prepare
.\.venv\Scripts\python.exe -m legal_intelligence.cli train
.\.venv\Scripts\python.exe -m legal_intelligence.cli evaluate
.\.venv\Scripts\python.exe -m legal_intelligence.cli index
```

Start the backend in one terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn legal_intelligence.api:app --host 127.0.0.1 --port 8000
```

Start the frontend in a second terminal in the same directory:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open **http://127.0.0.1:8501**. Interactive API documentation is at **http://127.0.0.1:8000/docs**. Restart the backend after rebuilding models or indexes. Missing data/models do not prevent startup: summaries and regex extraction still work, while prediction/search show setup instructions. No mock predictions are generated.

On Linux/macOS, use `python3.12 -m venv .venv` and substitute `.venv/bin/python` for `.\.venv\Scripts\python.exe`.

## Dataset, access, and provenance

The integration uses only the public, pinned [LeSICiN release 6053791](https://zenodo.org/records/6053791), documented by [Law-AI/LeSICiN](https://github.com/Law-AI/LeSICiN). The release API declares open access and [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The downloader verifies those declarations, saves the actual metadata, checks the release's published MD5 values, and records local SHA-256 hashes and file sizes in `data/raw/lesicin/manifest.json`. It downloads only these five data files (~511 MB decimal); it does not download the upstream neural checkpoint or sentence vectors:

| File | Use |
| --- | --- |
| `train.jsonl` | Supplied training split |
| `dev.jsonl` | Supplied validation split |
| `test.jsonl` | Supplied held-out split |
| `secs.jsonl` | IPC statute identities and dataset statute excerpts |
| `label_vocab.json` | Original published label vocabulary; model encoding is still fitted on training labels |

Each case has `id`, `text: list[str]`, and `labels: list[str]`. Statute records have `id` and `text`; they do not require labels. All original Act/section identities, including subsections, are retained. Case titles, court, judgment dates, citations, original judgment URLs, statute titles, version dates, and effective dates are **null** when absent. Numeric document IDs are not converted into guessed links. Dataset URLs identify the dataset, not an original judgment.

If downloading is blocked, manually download the five exact files from the release page into a folder such as `C:\datasets\lesicin`, then run:

```powershell
.\.venv\Scripts\python.exe -m legal_intelligence.cli import-lesicin C:\datasets\lesicin
.\.venv\Scripts\python.exe -m legal_intelligence.cli prepare
```

Alternatively place them directly in `data\raw\lesicin\` and run `prepare`. Both paths verify the pinned official checksums; renamed, edited, incomplete, or other-version files are rejected. Failed downloads leave `.part` files and can be rerun. The public source URLs and expected hashes are in `src/legal_intelligence/data/download.py`.

Attribution: **Shounak Paul, Pawan Goyal, Saptarshi Ghosh**, “LeSICiN: A Heterogeneous Graph-based Approach for Automatic Legal Statute Identification from Indian Legal Documents,” AAAI 2022, DOI **10.5281/zenodo.6053791**. This project normalizes text, selects narrative features, masks references, and excludes detected duplicate records; these are modifications, not author-endorsed annotations. Retain this attribution and licence link when sharing derived artifacts. The upstream implementation's code has not been copied.

[India Code](https://indiacode.gov.in/) is the official statutory reference link. This project does not scrape it, infer effective dates, or represent the dataset excerpts as authoritative current legislation. IPC labels are never automatically translated into BNS labels. Optional future sources (legal_NER, rhetorical-role-baseline, CJPE) are not downloaded or integrated; their licences and schemas need separate review.

## Features and limits

- **Section prediction:** scikit-learn `Pipeline(FunctionTransformer → TfidfVectorizer → OneVsRestClassifier(LogisticRegression))`. Scores are uncalibrated. Only labels listed by `/model-info` are supported. No threshold crossing produces `insufficient_evidence`; separate below-threshold suggestions are explicitly marked.
- **Input features:** LeSICiN is described as facts/evidence data, but direct inspection found outcome and procedural language. A conservative rule selects up to 40 narrative sentences / 12,000 characters, skips procedural sentences, and stops before the first detected outcome. Documents with fewer than 80 selected characters are excluded from training/evaluation. The same selection rule is applied to user input before prediction. This is not trained rhetorical-role classification or a guarantee of perfectly isolated facts. Original excerpts and selected sentence numbers are retained.
- **Reference masking:** provisions (including lists and subsections) are masked for training and prediction. Negation, other numbers, dates, and paragraph boundaries are retained. Audit reports count explicit references before and after masking and reject recognized outcome markers remaining in selected features. Unrecognized or obliquely expressed labels and outcomes remain a limitation; do not interpret results as verified outcome-free legal reasoning.
- **Retrieval:** a separate sparse TF-IDF index ranks original dataset passages by cosine similarity, returning one supporting passage per case with file, line, dataset sentence range, and excerpt character offsets. LeSICiN sentence lists have no original PDF pages or paragraph numbers: these remain null. Input PDFs do have page mappings. Judgment completeness and original links are not inferred. No labelled retrieval evaluation exists, so no retrieval accuracy is reported.
- **Summary:** deterministic TF-IDF centroid sentence ranking with redundancy reduction. Output sentences are copied verbatim from the analyzed source text, returned in source order, with locations. It does not label facts, reasoning, or decision. Analysis ranks at most the first 1,200 source sentences to bound memory; original source passages are included in the response. PDF text is normalized before analysis, so PDF offsets refer to extracted page text, not binary file offsets.
- **Extraction:** regex provision, limited citation, and date-like mention extraction. Explicit `Case title:`, `Court:` / `IN THE … COURT …`, and `Date of Judgment:` headings can supply case details with matched-text evidence. Unlabelled dates are not assigned judgment/event roles. This is not trained NER. A `LegalNERAdapter` protocol is reserved for future separately trained models.
- **PDFs:** PyMuPDF extraction with page mappings, configurable repeated-margin removal, byte/page/text limits, invalid/encrypted file errors, and explicit `ocr_required` errors for insufficient text. OCR is not bundled. Mixed scanned/searchable PDFs warn about pages with little text.
- **Categories:** civil/criminal/category classification is disabled. `train_category_classifier` is an explicit extension interface that refuses training until a verified labelled adapter exists.

## Splits, leakage, and evaluation

`prepare` validates every raw record, checks labels against statute identities, reports missing metadata, generates stable source-record IDs and document hashes, and writes SQLite plus `artifacts/reports/dataset.json`.

Leakage checks compare original IDs, normalized full text, and masked feature text. An additional approximate related-text check uses 64-bit token-set SimHash (four bands), Hamming distance ≤3, and token Jaccard ≥0.8. Detected duplicate/related records remain in SQLite with their original splits and explicit exclusion reasons. Later records are excluded in train → validation → test priority order; splits are never reassigned. This excludes observed overlap and does not establish independence of every related proceeding: docket/family metadata is unavailable, and approximate screening can miss paraphrases. Within-split exact duplicates are also excluded.

Training fits vectorization and label encoding on eligible **training** records only. Labels with fewer than five positive training examples or no negative examples are excluded from supported predictions and reported. Validation/test-only labels are reported and counted as false negatives in full-universe evaluation. Training uses seed 42, `liblinear`, one worker and bounded BLAS threads. It compares C=1 and C=4, and thresholds 0.15, 0.25, 0.35, 0.5; the selection objective is validation micro F1 over supported labels. A global threshold is saved per label. No calibration has been evaluated.

Optional character n-gram comparison (3–5 character n-grams within word boundaries), selected using validation data only:

```powershell
.\.venv\Scripts\python.exe -m legal_intelligence.cli train --compare-char
```

Run `evaluate` only after selecting the configuration. It loads test rows for the final evaluation, records a report per model version, and refuses to overwrite that version's test result. Do not use test results to choose another model. Dataset checksums must match the trained release. Reports include micro precision/recall/F1, macro F1, per-label precision/recall/F1/support, label coverage, unseen labels, and precision@k / recall@k / hit-rate@k for k=1,3,5, with their exact definitions. Reported classification results apply to eligible rule-selected excerpts, not every uploaded judgment or the excluded records.

The model bundle includes preprocessing, pipeline, training label encoder, supported-label order, thresholds, seed/configuration, package versions, dataset checksums, and a model version. `artifacts/ipc_model.json` exposes its metadata without unpickling it.

The default retrieval index covers training documents only and caps at 100,000 passages. Its report states actual coverage and truncation. To index all supplied splits without a passage cap (higher memory use):

```powershell
.\.venv\Scripts\python.exe -m legal_intelligence.cli index --splits train validation test --max-passages 0
```

## Architecture and data flow

```text
Pinned public JSONL files + checked manifest
  → LeSICiN adapter → schema validation + normalization + leakage audit
  → SQLite cases / statutes / provenance (nulls preserved, splits retained)
      ├─ selected narrative → mask references → training-only TF-IDF → OVR logistic regression
      │    → validation selection → trusted local bundle → separate final test evaluation
      └─ original excerpt passages → separate sparse retrieval index

Text / PDF → source passages (pages where available)
  ├─ narrative selection + trained classifier → thresholded IPC suggestions
  ├─ regex extraction → detected mentions (separate from predictions)
  ├─ sparse cosine search → supporting passages + provenance
  └─ extractive summarizer → original sentences + locations
       → FastAPI → Streamlit
```

Code is separated into `data/`, `preprocessing.py`, `training.py`, `evaluation.py`, `retrieval.py`, `summarization.py`, `pdf_processing.py`, `service.py`, `api.py`, and `streamlit_app.py`. Training, downloading, and indexing are CLI-only operations. API endpoints are `GET /health`, `GET /model-info`, `POST /analyze-text`, `POST /analyze-pdf`, and `POST /search`.

## Configuration and security

See `.env.example`. Set variables in the shell; `.env` files are not loaded automatically:

```powershell
$env:LEGAL_DATA_DIR = 'D:\legal-data'
$env:LEGAL_ARTIFACT_DIR = 'D:\legal-models'
$env:LEGAL_MAX_UPLOAD_MB = '20'
$env:LEGAL_MAX_PDF_PAGES = '300'
$env:LEGAL_MAX_TEXT_CHARS = '200000'
$env:LEGAL_PDF_MIN_TEXT_CHARS = '80'
$env:LEGAL_REMOVE_PDF_MARGINS = 'true'
$env:LEGAL_API_URL = 'http://127.0.0.1:8000'
```

The JSON request schema also caps input text at 200,000 characters. Raising the environment limit alone does not raise that schema cap. Both processes should use the same environment and working directory. Raw files, SQLite databases, model binaries, and local configuration are ignored by git. Nothing is uploaded to a remote inference service.

Joblib/pickle is executable serialization. Only operator-controlled files at configured model/index paths are loaded; uploaded files can only enter PDF text extraction. Artifact SHA-256 sidecars detect corruption, **not** malicious replacement. Do not use artifacts from untrusted sources. Uploaded filenames are never used as filesystem paths, and database values use parameterized SQL. The service binds to localhost in these commands and has no authentication; review deployment controls before exposing it to a network.

## Verification and recorded results

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
```

With both servers running and the real artifacts built, verify an actual text/PDF request, source attribution, summary offsets, and the Streamlit Analyze action:

```powershell
.\.venv\Scripts\python.exe scripts\verify_runtime.py
```

This smoke check reads one real training-corpus narrative and writes `artifacts/reports/runtime.json`. It does not access the held-out test split.

Tests use artificial cases only in temporary test directories. They verify schema rejection, subsection identities, duplicate leakage detection, explicit-label masking, label alignment, training-only vocabulary, no test access during training, serialization consistency, unseen-label metrics, response schemas, missing-model behavior, below-threshold states, retrieval attribution, verbatim summary offsets, and invalid/scanned/oversized PDF handling. Test metrics are not production metrics.

Actual implementation verification and real-data results are recorded in [VALIDATION.md](VALIDATION.md). Full machine-readable reports live under `artifacts/reports/` locally. Known limitations remain documented above even when checks pass.

## Training and evaluation graphs

Open `artifacts/reports/figures/training_results.png` for the validation/test metric comparison, validation threshold search, held-out top-k metrics, and F1 of the most frequent labels. `per_label_f1.png` shows all 100 label scores. Both are also exported as SVG; `per_label_metrics.csv` contains the full precision/recall/F1/support table with original labels.

Regenerate these graphs from the saved reports, without training or evaluating the test set again:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[reports]"
.\.venv\Scripts\python.exe scripts\plot_results.py
```

These are recorded evaluation graphs. The logistic regression run did not record epoch-by-epoch loss or accuracy, so no such history is invented. Figure provenance and source-report checksums are in `artifacts/reports/figures/provenance.json`.

## Inspecting the additional Hugging Face dataset

The pandas snippets for `overthelex/indian-court-decisions` load one entire split at a time. Running both snippets assigns to the same `df`, replacing the first reference. They do not run training. This dataset provides outcome columns, whereas the implemented LeSICiN model predicts IPC section labels; its data cannot be passed directly to the existing training adapter. The presence of a column does not guarantee populated labels.

Install the optional loading dependencies and inspect a bounded Supreme Court training preview:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[hf-data]"
.\.venv\Scripts\python.exe scripts\inspect_court_decisions.py --config supreme_court --split train --rows 1000
```

The inspector checks the publisher's declared licence, pins the repository commit, records the original split, reads the actual Parquet schema, and saves a raw metadata preview plus provenance report. It uses Arrow batch iteration with pandas for the selected rows. Metadata columns are read by default; `--include-text` includes judgment text. Parquet readers may fetch complete column chunks, so a small requested row count does not guarantee a small download. The inspected Supreme Court training file has a single row group covering 40,044 records; reading its text column can transfer a substantial part of the 450 MB file. The High Courts training file is approximately 19.3 GB compressed and should not be loaded wholesale into this environment's pandas memory.

For file/schema metadata only, without reading preview rows:

```powershell
.\.venv\Scripts\python.exe scripts\inspect_court_decisions.py --config high_courts --split train --rows 0
```

Previews are isolated under `data/raw/indian_court_decisions/previews/<commit>/<config>/`; inspection reports are under `artifacts/reports/hf_dataset_inspection/<commit>/`. They are inspection samples, not complete training datasets. The report distinguishes publisher-provided full-file hashes from locally verified preview hashes. No existing model, database or retrieval index is modified. The declared CC BY 4.0 licence is recorded with attribution; metadata alignment, upstream rights, case-family leakage and outcome leakage need further audit before an outcome-training integration.

Actual inspection at commit `30e604222ae94c9881dd42626b2cd07294cf1e56` found that all 40,044 Supreme Court training records have empty `disposal_nature`, `disposal_nature_normalized`, and `case_type` values. That training split cannot currently support supervised outcome or case-type training without verified labels from another source. The complete training-label scan is recorded in `artifacts/reports/hf_dataset_inspection/30e604222ae94c9881dd42626b2cd07294cf1e56/supreme_court-train-label-coverage.json`. The first 1,000 High Courts training records contain 910 nonempty normalized outcome labels and 90 blanks; this prefix is not a representative sample or a full-split label audit. No judgment text or held-out examples were read for these label checks, and no model was trained on this additional dataset.

## Separate GPU transformer experiment

The GPU experiment fine-tunes the public [DistilBERT base encoder](https://huggingface.co/distilbert/distilbert-base-uncased) with a new multilabel IPC classification head. It preserves the original train/validation/test membership and eligibility rules. It is a classifier, not a chat model. The publisher declares Apache 2.0 for the base model; LeSICiN data provenance and CC BY 4.0 attribution remain separate. The pinned base commit and downloaded file hashes are recorded with each experiment.

Install the CUDA wheel first, followed by the optional project dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[gpu,reports]"
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

For the exact tested GPU additions, use `python -m pip install -r requirements-gpu.lock.txt` after the ordinary `requirements.lock.txt`. The GPU lock includes the CUDA wheel index and the compatible setuptools version; keep that installation order when recreating the environment.

Run training, final evaluation, or regenerate graphs:

```powershell
.\.venv\Scripts\python.exe -m legal_intelligence.cli train-gpu --epochs 3 --patience 2 --seed 42
.\.venv\Scripts\python.exe -m legal_intelligence.cli evaluate-gpu
.\.venv\Scripts\python.exe -m legal_intelligence.cli plot-gpu
```

Training requires a functioning CUDA GPU and never silently falls back to CPU. The initial configuration uses batch size 16, gradient accumulation of 2 batches, evaluation batch size 64, at most 256 tokens per document, AdamW at learning rate 0.00003, and mixed precision. `--batch-size`, `--eval-batch-size`, `--accumulation-steps`, `--max-length`, `--learning-rate`, and `--revision` are configurable. Reducing batch sizes is the first option if GPU memory is insufficient. The tokenizer vocabulary is frozen from the public pretrained model; task labels are derived from training data only. Truncation counts are measured and recorded because the transformer sees shorter narratives than the TF-IDF baselines.

One epoch visits every eligible training document once in a seed-42 shuffled order. After each complete epoch, full training and validation scores are measured in evaluation mode: mean unweighted binary log loss, micro precision/recall/F1, macro F1, and exact-match subset accuracy at a fixed threshold of 0.5. These are not per-label accuracy scores. The checkpoint with strictly lowest validation loss is saved; ties retain the earlier checkpoint, with early-stopping patience 2 by default. The best checkpoint is restored before selecting a global prediction threshold by validation micro F1. Held-out test examples are loaded only by `evaluate-gpu`, which permits one evaluation per experiment.

All GPU model weights (safetensors), tokenizer files, label order, thresholds, configuration, provenance, CSV/JSON history, plots and evaluation reports are isolated under `artifacts/gpu_experiments/ipc-distilbert-gpu-<version>/`. The latest pointer is `artifacts/gpu_experiments/latest.json`. CPU model and epoch artifacts are hashed before training and verified afterward. The running application continues to use its existing CPU model; GPU results must be reviewed before changing its serving model. Actual GPU results are documented in `GPU_RESULTS.md` once training and evaluation finish.

## Separate SGD epoch experiment

The CPU epoch experiment uses the same eligible LeSICiN training and validation records, narrative selection and reference masking. It writes a unique directory under `artifacts/experiments/ipc-sgd-epochs-<version>/`. It preserves the original `ipc_model.joblib`, baseline reports, and baseline figures; the API continues serving that original model.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[reports]"
.\.venv\Scripts\python.exe -m legal_intelligence.cli train-epochs --epochs 20 --patience 5 --seed 42
.\.venv\Scripts\python.exe -m legal_intelligence.cli evaluate-epochs
```

`train-epochs` trains and generates graphs; it never loads held-out test examples. `evaluate-epochs` performs the selected experiment's single final test evaluation. Without a directory argument, evaluation uses `artifacts/experiments/latest.json`. You can specify an experiment explicitly:

```powershell
.\.venv\Scripts\python.exe -m legal_intelligence.cli evaluate-epochs artifacts\experiments\ipc-sgd-epochs-YOUR_VERSION
.\.venv\Scripts\python.exe -m legal_intelligence.cli plot-epochs artifacts\experiments\ipc-sgd-epochs-YOUR_VERSION
```

`plot-epochs` reads saved history only. It does not train or evaluate anything. Install the `reports` extra to generate graphs. Available training flags also include `--batch-size` (default 1024), `--alpha` (default 0.00001), `--max-features` (default 30000), and `--min-label-support` (default 5).

The classifier is a one-vs-rest arrangement of independent `SGDClassifier(loss="log_loss")` binary models, each initialized with classes `[0, 1]` through public `partial_fit` calls. This handles labels absent from the first minibatch. The small `IncrementalOVRClassifier` adapter is necessary because the installed scikit-learn 1.9.1 `OneVsRestClassifier.partial_fit` raises a multilabel initialization error when directly given a fresh indicator matrix. No artificial initialization examples or private scikit-learn fitted attributes are used.

TF-IDF and label encoding are fitted once on training data. Vocabulary, IDF weights, label order, and preprocessing remain fixed. A seeded permutation visits each eligible training record exactly once per epoch; the final smaller minibatch is included. SGD uses L2 regularization, the optimal learning-rate schedule, one worker, one BLAS thread, and no internal shuffling. No GPU is used.

Every completed epoch records train and validation mean binary log loss, micro precision/recall/F1, macro F1, and exact-match subset accuracy at **fixed threshold 0.5**. Log loss is computed stably from logits and averaged over all supported document-label pairs; it excludes the L2 penalty. Unsupported gold labels count as false negatives in classification metrics but have no modeled score for loss. Subset accuracy requires the entire predicted label set to match the gold set. It is not per-label accuracy and can be much lower than micro F1.

The best checkpoint minimizes validation loss strictly; ties retain the earlier checkpoint. Training stops after five consecutive non-improving epochs, or the requested maximum. After restoring the saved best checkpoint, a global threshold is selected from 0.15, 0.25, 0.35 and 0.5 by validation micro F1. This selected threshold is used for final evaluation; the epoch curves remain at 0.5. Neither checkpoint nor threshold selection reads test data.

Each experiment contains:

| File | Contents |
| --- | --- |
| `epoch_history.json`, `epoch_metrics.csv` | Actual completed epochs, metrics, sample counts, shuffle-order hashes and timing |
| `best_checkpoint.joblib` + checksum | Lowest-validation-loss model with epoch comparison threshold 0.5 |
| `model.joblib` + checksum | Restored best model with validation-selected thresholds, preprocessing, labels and provenance |
| `metadata.json` | Configuration, versions, seeds, frozen-feature signature, selected epoch and actual completed count |
| `validation.json` | Threshold search and validation evaluation |
| `test.json` | Single final held-out evaluation, created only by `evaluate-epochs` |
| `baseline_snapshot.json` | Original model/report/figure hashes used to verify preservation |
| `figures/epoch_overview.png` | Combined loss, micro F1, macro F1 and subset accuracy history |
| `figures/loss_vs_epoch.png`, `f1_vs_epoch.png`, `subset_accuracy_vs_epoch.png` | Individual epoch graphs; SVG versions are also exported |

An exclusive `test_evaluation.started.json` marker prevents concurrent or repeated test evaluation. If evaluation fails after starting, retain the partial run for diagnosis; the command refuses an automatic rerun of that experiment. Model artifacts are trusted operator-controlled local files and must never be supplied through user uploads.

Epoch metrics are experiment measurements, not reconstructed baseline history. Class imbalance means an aggregate loss can improve while individual rare labels remain weak. Validation is reused for checkpoint and threshold selection; these are validation estimates, not an independent evaluation set. No calibration or improvement over the existing baseline is assumed. Actual run results and graph links are recorded in `EPOCH_RESULTS.md` after execution.
