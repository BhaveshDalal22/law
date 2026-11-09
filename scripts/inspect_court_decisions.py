"""Inspect pinned Hugging Face Parquet data in bounded batches; this does not train a model."""

import argparse
import hashlib
import json
from datetime import UTC, datetime

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from legal_intelligence.config import Settings

REPOSITORY = "overthelex/indian-court-decisions"
DATASET_URL = f"https://huggingface.co/datasets/{REPOSITORY}"


def inspect(config, split, rows, include_text, revision="main"):
    if not 0 <= rows <= 10000:
        raise ValueError("Use --rows between 0 (metadata only) and 10000; this script is an inspector.")
    # Public dataset only. No login, remote execution, or model download is involved.
    info = HfApi(token=False).dataset_info(REPOSITORY, revision=revision, files_metadata=True)
    license_id = info.card_data.get("license") if info.card_data else None
    if license_id != "cc-by-4.0":
        raise ValueError(f"Dataset licence declaration is {license_id!r}; review it before reading samples.")
    filename = f"{config}/{split}.parquet"
    upstream_file = next(item for item in info.siblings if item.rfilename == filename)
    resolved_revision = info.sha
    fs = HfFileSystem(token=False)
    source_path = f"datasets/{REPOSITORY}/{filename}"
    frames = []
    print(f"Inspecting {filename} at commit {resolved_revision}; requested preview rows: {rows}.", flush=True)
    with fs.open(source_path, "rb", revision=resolved_revision, cache_type="none") as stream:
        parquet = pq.ParquetFile(stream, pre_buffer=False)
        columns = [name for name in parquet.schema_arrow.names if include_text or name != "full_text"]
        schema = {field.name: str(field.type) for field in parquet.schema_arrow}
        total_rows, row_groups = parquet.metadata.num_rows, parquet.num_row_groups
        first_group_rows = parquet.metadata.row_group(0).num_rows if row_groups else 0
        remaining = rows
        if remaining:
            for batch in parquet.iter_batches(batch_size=min(rows, 256), columns=columns, use_threads=False):
                taken = batch.slice(0, min(remaining, batch.num_rows))
                frames.append(taken.to_pandas())
                remaining -= taken.num_rows
                if remaining == 0:
                    break
    settings = Settings()
    directory = settings.data_dir / "raw" / "indian_court_decisions" / "previews" / resolved_revision / config
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": REPOSITORY,
        "dataset_url": DATASET_URL,
        "revision": resolved_revision,
        "source_url": f"{DATASET_URL}/resolve/{resolved_revision}/{filename}",
        "config": config,
        "split": split,
        "inspected_at": datetime.now(UTC).isoformat(),
        "license_declared_by_publisher": license_id,
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Volodymyr Ovcharov, Indian Court Decisions / Multi-Legal-Bench (2026); dataset card credits Dattam Labs and original court sources.",
        "upstream_file_bytes": upstream_file.size,
        "upstream_lfs_sha256": upstream_file.lfs.sha256 if upstream_file.lfs else None,
        "whole_upstream_file_checksum_verified": False,
        "parquet_rows": total_rows,
        "row_groups": row_groups,
        "first_row_group_rows": first_group_rows,
        "schema": schema,
        "preview_columns": columns,
        "full_text_in_preview": include_text,
        "preview_rows_requested": rows,
        "preview_rows_read": 0,
        "training_performed": False,
        "existing_models_modified": False,
        "notes": [
            "Preview covers the first requested rows, not a random or representative sample.",
            "Parquet may fetch/decode whole column chunks; a row limit does not guarantee small network transfers.",
            "Full judgment text is excluded by default. --include-text can read a large text column chunk.",
            "Upstream LFS hash is publisher metadata, not a locally verified full-download checksum.",
            "Outcome labels are distinct from IPC section labels. No labels are inferred or renamed.",
            "Dataset card licence declaration is recorded; upstream rights and metadata alignment still need audit before training.",
        ],
    }
    if frames:
        import pandas as pd

        preview = pd.concat(frames, ignore_index=True)
        preview_path = (
            directory / f"{split}.first-{len(preview)}.{'with-text' if include_text else 'metadata'}.jsonl"
        )
        preview.to_json(preview_path, orient="records", lines=True, force_ascii=False)
        report.update(
            {
                "preview_rows_read": len(preview),
                "preview_path": str(preview_path),
                "preview_sha256": hashlib.sha256(preview_path.read_bytes()).hexdigest(),
                "missing_values_in_preview": {
                    name: int(
                        (
                            preview[name].isna()
                            | preview[name].astype("string").str.strip().eq("").fillna(False)
                        ).sum()
                    )
                    for name in columns
                },
                "duplicate_cnr_in_preview": int(preview["cnr"].duplicated().sum()),
                "outcome_counts_in_preview": {
                    str(label): int(count)
                    for label, count in preview["disposal_nature_normalized"]
                    .value_counts(dropna=False)
                    .items()
                },
            }
        )
    report_path = (
        settings.artifact_dir
        / "reports"
        / "hf_dataset_inspection"
        / resolved_revision
        / f"{config}-{split}-rows{rows}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("supreme_court", "high_courts"), default="supreme_court")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument(
        "--rows", type=int, default=1000, help="0 for file/schema metadata only; maximum 10000"
    )
    parser.add_argument(
        "--include-text", action="store_true", help="Include judgment text; remote column chunks can be large"
    )
    parser.add_argument("--revision", default="main", help="Commit/tag/branch resolved to a pinned commit")
    args = parser.parse_args()
    try:
        result = inspect(args.config, args.split, args.rows, args.include_text, args.revision)
    except (OSError, ValueError, StopIteration) as exc:
        parser.exit(1, f"Cannot inspect dataset: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
