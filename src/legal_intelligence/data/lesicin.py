import json
import re
from collections import Counter

import pandas as pd

from legal_intelligence.data.download import RECORD_URL, manifest
from legal_intelligence.data.leakage import fingerprint, leakage_audit
from legal_intelligence.data.store import connect, initialize, metadata
from legal_intelligence.preprocessing import (
    OUTCOME_RE,
    PROVISION_RE,
    mask_references,
    normalize_text,
    select_fact_narrative,
)

SPLITS = {"train.jsonl": "train", "dev.jsonl": "validation", "test.jsonl": "test"}
SECTION_ID = re.compile(r"^Section (\d+[A-Za-z]*(?:\([A-Za-z0-9]+\))*) in (The Indian Penal Code)$")


def read_jsonl(path, require_labels=True):
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{number}: invalid JSON") from exc
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("id"), str)
                or not record["id"].strip()
            ):
                raise ValueError(f"{path.name}:{number}: expected non-empty string id")
            if not isinstance(record.get("text"), list) or not all(
                isinstance(s, str) for s in record["text"]
            ):
                raise ValueError(f"{path.name}:{number}: text must be list[str]")
            if not any(s.strip() for s in record["text"]):
                raise ValueError(f"{path.name}:{number}: empty text")
            labels = record.get("labels")
            if require_labels and (
                not isinstance(labels, list) or not all(isinstance(s, str) for s in labels)
            ):
                raise ValueError(f"{path.name}:{number}: labels must be list[str]")
            yield number, record


def adapt_case(record, split, filename, line):
    original = [normalize_text(s) for s in record["text"]]
    full_text = "\n\n".join(original)
    selected = select_fact_narrative(original)
    return {
        "case_id": f"lesicin:{split}:{record['id']}",
        "original_id": record["id"],
        "document_id": fingerprint(full_text),
        "case_title": None,
        "court": None,
        "judgment_date": None,
        "citation": None,
        "facts_text": "\n\n".join(t for _, t in selected),
        "full_text": full_text,
        "act_section_labels": sorted(set(record["labels"])),
        "source_url": None,
        "source_dataset": "LeSICiN",
        "split": split,
        "source_file": filename,
        "source_line": line,
        "dataset_record_url": RECORD_URL,
        "text_kind": "dataset_excerpt_not_verified_full_judgment",
        "sentences": original,
        "fact_sentence_numbers": [n for n, _ in selected],
    }


def adapt_statute(record):
    match = SECTION_ID.fullmatch(record["id"])
    if not match:
        raise ValueError(f"Unexpected IPC section identity: {record['id']}")
    return {
        "statute_id": record["id"],
        "act_name": match[2],
        "section_number": match[1],
        "section_title": None,
        "text": "\n\n".join(record["text"]),
        "version_date": None,
        "effective_from": None,
        "effective_to": None,
        "source_url": None,
        "dataset_record_url": RECORD_URL,
    }


def prepare(raw_dir, db_path, report_path, verify=True):
    provenance = manifest(raw_dir) if verify else {"dataset": "TEST FIXTURE", "files": []}
    statutes = [
        adapt_statute(record) for _, record in read_jsonl(raw_dir / "secs.jsonl", require_labels=False)
    ]
    ids = {s["statute_id"] for s in statutes}
    if len(ids) != len(statutes):
        raise ValueError("Duplicate statute identities")
    cases, schema_keys, explicit, residual = [], {}, Counter(), Counter()
    for filename, split in SPLITS.items():
        keys = Counter()
        for line, raw in read_jsonl(raw_dir / filename):
            keys.update(raw.keys())
            unknown = set(raw["labels"]) - ids
            if unknown:
                raise ValueError(f"{filename}:{line}: labels absent from secs.jsonl: {sorted(unknown)}")
            row = adapt_case(raw, split, filename, line)
            explicit[split] += bool(PROVISION_RE.search(row["facts_text"]))
            residual[split] += bool(PROVISION_RE.search(mask_references(row["facts_text"])))
            cases.append(row)
        schema_keys[filename] = dict(keys)
    if len({r["case_id"] for r in cases}) != len(cases):
        raise ValueError("Duplicate id within a split; import aborted to avoid overwriting source records.")
    audit = leakage_audit(cases)
    missing = Counter()
    for row in cases:
        for field in ("case_title", "court", "judgment_date", "citation", "source_url"):
            missing[field] += row[field] is None
        row["eligible"] = 1
        row["exclusion_reason"] = None
        if not row["facts_text"] or len(row["facts_text"]) < 80:
            row["eligible"], row["exclusion_reason"] = 0, "insufficient_rule_selected_narrative"
        if row["case_id"] in audit["exclusions"]:
            row["eligible"], row["exclusion_reason"] = 0, audit["exclusions"][row["case_id"]]
        if OUTCOME_RE.search(row["facts_text"]):
            raise ValueError("Outcome/procedural marker survived facts selection")
    frame = pd.DataFrame([{k: r[k] for k in ("split", "eligible", "exclusion_reason")} for r in cases])
    report = {
        "provenance": provenance,
        "raw_schema_key_counts": schema_keys,
        "counts": dict(Counter(r["split"] for r in cases)),
        "missing_values": dict(missing),
        "eligibility": frame.groupby(["split", "eligible"])
        .size()
        .reset_index(name="count")
        .to_dict("records"),
        "exclusion_reasons": dict(
            Counter(r["exclusion_reason"].split(":")[0] for r in cases if r["exclusion_reason"])
        ),
        "explicit_reference_documents_before_masking": dict(explicit),
        "explicit_reference_documents_after_masking": dict(residual),
        "leakage": audit,
        "facts_policy": "Rule-selected narrative prefix, skip procedural sentences, stop before first outcome, max 40 sentences/12000 chars; not trained rhetorical-role annotation.",
        "statutes": len(statutes),
        "statute_version": None,
    }
    with connect(db_path) as db:
        initialize(db)
        # Atomic replacement only after full validation and audit, preserving old DB on failure.
        db.execute("DELETE FROM cases")
        db.execute("DELETE FROM statutes")
        for table, rows in [("cases", cases), ("statutes", statutes)]:
            for row in rows:
                values = {k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()}
                columns = list(values)
                db.execute(
                    f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
                    list(values.values()),
                )
        metadata(db, "preparation_report", report)
        metadata(db, "provenance", provenance)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
