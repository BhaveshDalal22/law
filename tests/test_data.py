import json

import pytest

from legal_intelligence.data.leakage import leakage_audit
from legal_intelligence.data.lesicin import adapt_case, adapt_statute, read_jsonl
from legal_intelligence.data.store import connect, load_cases
from legal_intelligence.preprocessing import PROVISION_RE, mask_references, normalize_text


def test_adapter_preserves_identity_and_unknown_metadata(prepared):
    settings, report, _ = prepared
    with connect(settings.db_path) as db:
        row = load_cases(db, "train")[0]
    assert row["case_title"] is None and row["source_url"] is None
    assert row["case_id"] == "lesicin:train:training-0"
    assert row["split"] == "train" and row["source_line"] == 1
    assert "convicted" not in row["facts_text"] and "convicted" in row["full_text"]
    assert report["counts"] == {"train": 10, "validation": 4, "test": 4}
    assert report["explicit_reference_documents_after_masking"] == {"train": 0, "validation": 0, "test": 0}


@pytest.mark.parametrize(
    "record",
    [
        {"id": "x", "text": "not a list", "labels": []},
        {"id": "x", "text": [], "labels": []},
        {"id": "x", "text": ["text"], "labels": None},
    ],
)
def test_bad_schema_fails_with_line_context(tmp_path, record):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        list(read_jsonl(path))


def test_cross_split_identity_and_masked_text_leakage():
    base = {
        "id": "same",
        "text": [
            "A witness described the incident at the factory in a detailed narrative without any judicial discussion."
        ],
        "labels": [],
    }
    first = adapt_case(base, "train", "train.jsonl", 1)
    second = adapt_case(base, "test", "test.jsonl", 1)
    audit = leakage_audit([first, second])
    assert audit["cross_split_pairs"] == 1
    assert second["case_id"] in audit["exclusions"]
    assert first["case_id"] not in audit["exclusions"]


@pytest.mark.parametrize(
    "reference",
    [
        "sections 302, 34 and 120-B IPC",
        "Section 120 B of the IPC",
        "IPC 302/34",
        "Article 21 of the Constitution",
    ],
)
def test_explicit_label_masking(reference):
    masked = mask_references(f"He did not leave. {reference} applies. The sum was 500 and the year was 2020.")
    assert not PROVISION_RE.search(masked)
    assert "302" not in masked and "120" not in masked
    assert "not" in masked and "500" in masked and "2020" in masked


def test_unicode_normalization_preserves_negation_and_paragraphs():
    assert normalize_text("No\t consent.\r\n\r\nAmount １２３.") == "No consent.\n\nAmount 123."


def test_subsection_id_is_preserved():
    row = adapt_statute({"id": "Section 294(b) in The Indian Penal Code", "text": ["Artificial test only."]})
    assert row["section_number"] == "294(b)"
    assert row["effective_from"] is None and row["version_date"] is None
