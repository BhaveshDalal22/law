# Recorded validation — 2 November 2026

This file records the original CPU baseline checks and measurements. Separate GPU experiment measurements are recorded in `GPU_RESULTS.md`. No artificial fixture metrics are reported as model performance.

## Implementation and environment

- Project built from an initially empty workspace; no existing project code was replaced.
- Windows, Python **3.12.5**, CPU only. Installed environment is pinned in `requirements.lock.txt`.
- Training package versions: scikit-learn **1.9.1**, NumPy **2.5.4**, pandas **3.1.0**, SciPy **1.18.1**, spaCy **3.8.18**, joblib **1.6.0**, PyMuPDF **2.0**, FastAPI **0.162.0**.
- No external LLM, generative model, GPU, or upstream neural checkpoint was used for this original baseline. Optional GPU training is a separate experiment.

## Real dataset preparation

Downloaded the five public data files from [LeSICiN release 6053791](https://zenodo.org/records/6053791). Access and licence were inspected through its API: open access, CC BY 4.0. Every file matched the release's published MD5; SHA-256 hashes and source URLs are recorded in `data/raw/lesicin/manifest.json` and the training report. Dataset metadata was saved in `data/raw/lesicin/zenodo-record.json`.

| Split | Supplied records | Eligible after checks | Excluded, original split retained |
| --- | ---: | ---: | ---: |
| Train | 42,835 | 35,192 | 7,643 |
| Validation (`dev`) | 10,200 | 8,250 | 1,950 |
| Test | 13,039 | 10,400 | 2,639 |
| Total | 66,074 | 53,842 | 12,232 |

Final exclusion reasons: 9,163 insufficient rule-selected narrative records, 2,231 duplicate records, and 838 approximately related records. There were **15,326 flagged cross-split pairs** before exclusions; pairs and records are different counts, and many matches involve already unusable short excerpts. A direct check of the recorded pairs against eligible SQLite IDs found **zero retained flagged pairs**. This does not prove the absence of undiscovered related proceedings.

The source schema contains only `id`, `text`, and `labels` for cases. All 66,074 case records lack supplied title, court, judgment date, citation, and original judgment URL; those fields remain null. There are 100 statute identities, including the observed `Section 294(b)` and `Section 376(2)` identities, preserved without renaming. Statute version and effective dates are unavailable.

Provision-reference audit over selected narrative features, before final eligibility exclusions:

| Split | Documents with recognized provision mentions before masking | After masking |
| --- | ---: | ---: |
| Train | 16,444 | 0 |
| Validation | 3,760 | 0 |
| Test | 4,760 | 0 |

These counts describe the implemented regex, not a guarantee that every indirect or malformed reference has been removed. Recognized outcome/procedural markers were checked against selected training narratives. Source excerpts retain their original outcome language for search; it is not intentionally used as prediction features.

Full preparation report, including exact source hashes, exclusions and related-text screening: `artifacts/reports/dataset.json`.

## Trained model and final held-out evaluation

Model version: **`ipc-tfidf-4edb7524877a`**.

Used all 35,192 eligible training records and 8,250 eligible validation records. The 10,400 eligible test records were evaluated once after selection. TF-IDF and the label encoder were fitted on training data only. The supported-label order is saved with the model and checked against its training encoder. All **100** original IPC labels met the minimum support of five positive training examples and had negative examples; no labels were excluded as rare or constant for this run. No validation-only or test-only labels were found. Gold-label occurrence coverage is **1.0**, which means supported-label coverage, not predictive accuracy.

Pipeline: saved reference masking → word TF-IDF unigrams/bigrams, maximum 30,000 features, minimum document frequency 2, sublinear term frequency → One-vs-Rest Logistic Regression, `liblinear`, seed 42, one worker, maximum 1,000 iterations. No convergence failure was detected.

Validation compared C=1 and C=4 with thresholds 0.15, 0.25, 0.35 and 0.5. Selected **C=4, threshold=0.25** using validation micro F1 only. The optional character configuration is implemented but was not run in this baseline; no character-comparison result is claimed. Do not use the held-out results below to retune this model.

| Metric | Validation | Held-out test |
| --- | ---: | ---: |
| Micro precision | 0.491923 | 0.488597 |
| Micro recall | 0.410279 | 0.399172 |
| Micro F1 | 0.447407 | 0.439381 |
| Macro F1 | 0.278590 | 0.264110 |

Test ranking metrics, independent of decision thresholds:

| k | Precision@k | Recall@k | Hit rate@k |
| --- | ---: | ---: | ---: |
| 1 | 0.592308 | 0.191276 | 0.592308 |
| 3 | 0.457724 | 0.400140 | 0.785000 |
| 5 | 0.375385 | 0.522986 | 0.851442 |

Precision@k is the document-mean number of gold labels in the top k divided by k. Recall@k divides by that document's number of gold labels. Hit rate@k is the fraction of documents with at least one gold label in the top k. The machine-readable reports include the full per-label precision/recall/F1/support table, package versions, configuration, seed, original-label support, dataset hashes, and exact definitions:

- `artifacts/reports/training.json`
- `artifacts/reports/test-ipc-tfidf-4edb7524877a.json`
- `artifacts/ipc_model.json`

The test figures apply to the retained rule-selected excerpts and the supplied dataset labels. Macro F1 shows weaker performance across labels than the aggregate micro score. Scores are **not calibrated confidence**, legal certainty, or evidence that the system supports every legal domain.

## Retrieval and live application

Built a separate sparse index with **100,000 passages from 14,498 training documents**. The configured 100,000-passage cap was reached; this index does not cover the entire dataset. Coverage is reported by `/model-info`, the UI, and `artifacts/reports/retrieval.json`. Results identify source file, line, dataset sentence range, and excerpt character offsets. Original judgment URLs and PDF page numbers are not present in this source and are not invented. Cosine scores are retrieval similarity. No labelled query-relevance set was available, so no retrieval accuracy is reported.

Started both services and ran `scripts/verify_runtime.py` against them with a real training-corpus narrative. Actual results:

- API health: HTTP 200; trained model and retrieval index available; no artifact errors.
- Real text analysis: HTTP 200, real model version returned, prediction status `ok`.
- Summary sentences match the original input at their returned offsets.
- Retrieved passages have source filenames and line attribution.
- Search: HTTP 200 with results.
- Searchable PDF generated from that source text: HTTP 200 with page references.
- Streamlit health: HTTP 200.
- Streamlit's Analyze button completed through `streamlit.testing.v1.AppTest` with no UI exception and the real backend model version.
- The runtime smoke check did not access the held-out test split.

Saved result: `artifacts/reports/runtime.json`. Local services were confirmed at **http://127.0.0.1:8000** and **http://127.0.0.1:8501**. Startup commands are in the README if those processes have stopped.

## Automated checks

- `python -m pytest -q`: **26 passed**. Fixtures are artificial and isolated in pytest temporary directories; they never enter the real training pipeline or these performance figures.
- `python -m ruff check .`: passed.
- `python -m ruff format --check .`: 25 Python files already formatted.
- `python -m pip check`: no broken requirements.
- Python compilation check passed before runtime verification; subsequent code was imported by the passing tests and live smoke check.

Checks cover adapter correctness and subsection identities; malformed raw schemas; exact cross-split duplication; masking; label alignment and training-only vocabulary; test isolation; save/load consistency; unsupported and single-label metric handling; response schemas; threshold and out-of-vocabulary behavior; retrieval source attribution; deterministic verbatim summaries; missing/corrupt local models; invalid/image-only/oversized PDFs; page limits; mixed scans; page mapping; and explicit-heading case-detail evidence.

The test environment emits one upstream Starlette TestClient deprecation warning about its httpx transport. The AppTest harness emits its expected bare-mode ScriptRunContext warning. Neither prevented the checks from passing.

## Remaining limits

No remaining download or baseline-training step is required in this workspace. The system is a runnable research baseline, with these material limits:

- Narrative selection and outcome/reference filtering are conservative rules, not verified facts-only annotations; unrecognized language can survive and usable facts can be discarded.
- Duplicate screening is approximate for related cases and cannot substitute for unavailable docket-family metadata.
- Dataset labels and excerpt boundaries can be noisy; text completeness is unverified.
- The retrieval index is capped and covers only the recorded subset.
- Statute text/version applicability must be checked independently at [India Code](https://indiacode.gov.in/); no IPC-to-BNS translation or effective-date inference is performed.
- No trained category classifier, trained legal NER, OCR engine, probability calibration, or retrieval-relevance benchmark is included.
