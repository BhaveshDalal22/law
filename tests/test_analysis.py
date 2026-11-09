import base64
from dataclasses import replace

import pymupdf
import pytest
from fastapi.testclient import TestClient

from legal_intelligence.api import create_app
from legal_intelligence.artifacts import load_bundle
from legal_intelligence.config import Settings
from legal_intelligence.data.store import connect, load_cases
from legal_intelligence.pdf_processing import PDFError, extract_pdf
from legal_intelligence.preprocessing import extract_case_details, text_passages
from legal_intelligence.retrieval import build_index, search
from legal_intelligence.service import IntelligenceService
from legal_intelligence.summarization import summarize


def test_missing_model_api_still_summarizes(tmp_path):
    client = TestClient(create_app(Settings(data_dir=tmp_path / "data", artifact_dir=tmp_path / "artifacts")))
    assert client.get("/health").status_code == 200
    assert client.get("/model-info").json()["supported_labels"] == []
    response = client.post(
        "/analyze-text",
        json={"text": "The witness did not consent. Section 34 IPC was mentioned on 12.04.2020."},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["prediction"]["status"] == "missing_model"
    assert result["prediction"]["suggestions"] == []
    assert result["detected_references"][0]["method"] == "regex"
    assert result["dates"][0]["text"] == "12.04.2020"
    assert result["summary"] and result["retrieval_status"] == "missing_index"
    assert client.post("/analyze-text", json={"text": "  "}).status_code == 422
    assert client.post("/search", json={"query": "witness"}).json()["status"] == "missing_index"
    assert client.post("/analyze-text", json={"text": "x", "summary_sentences": 100}).status_code == 422


def test_prediction_schema_and_below_threshold_are_separate(trained):
    service = IntelligenceService(trained[0])
    result = service.analyze_text(
        "A witness described an attack using a sharp weapon beside a house during the evening."
    )
    assert result.prediction.status in ("ok", "insufficient_evidence")
    assert all(s.above_threshold for s in result.prediction.suggestions)
    service.model["thresholds"] = [1.0] * len(service.model["labels"])
    result = service.analyze_text(
        "A witness described an attack using a sharp weapon beside a house during the evening."
    )
    assert result.prediction.status == "insufficient_evidence" and not result.prediction.suggestions
    assert all(not s.above_threshold for s in result.prediction.below_threshold)
    assert service.analyze_text("zxqvv qqqqq jjjjj wwww zzzz.").prediction.status == "insufficient_evidence"


def test_summary_is_verbatim_with_correct_offsets_and_order():
    text = "A witness saw the car. The driver did not stop.\n\nPolice found the car beside a river. The witness reported the incident."
    result = summarize(text_passages(text), 2)
    assert len(result) == 2
    assert all(text[s.start_char : s.end_char] == s.text for s in result)
    assert [s.start_char for s in result] == sorted(s.start_char for s in result)
    assert [s.text for s in result] == [s.text for s in summarize(text_passages(text), 2)]


def test_retrieval_source_attribution(prepared):
    settings, _, _ = prepared
    report = build_index(
        settings.db_path, settings.index_path, settings.artifact_dir / "reports", max_passages=0
    )
    bundle = load_bundle(settings.index_path, "retrieval")
    hits = search(bundle, "sharp weapon victim", 3)
    assert hits and report["accuracy"] is None
    with connect(settings.db_path) as db:
        rows = {r["case_id"]: r for r in load_cases(db, "train", eligible=False)}
    for hit in hits:
        passage = hit.passage
        assert rows[hit.case_id]["full_text"][passage.start_char : passage.end_char] == passage.text
        assert hit.source_url is None and hit.source_file == "train.jsonl" and hit.source_line >= 1
        assert passage.page is None and passage.paragraph is None and passage.sentence_start >= 1
    assert search(bundle, "zxqvvqqq", 3) == []


def pdf_bytes(text=None, image=False):
    with pymupdf.open() as document:
        page = document.new_page()
        if text:
            page.insert_text((60, 70), text)
        if image:
            png = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aD1sAAAAASUVORK5CYII="
            )
            page.insert_image(pymupdf.Rect(10, 10, 100, 100), stream=png)
        return document.tobytes()


def test_invalid_scanned_and_oversized_pdf(tmp_path):
    settings = Settings(data_dir=tmp_path, artifact_dir=tmp_path)
    with pytest.raises(PDFError, match="Invalid PDF"):
        extract_pdf(b"not a pdf", settings)
    with pytest.raises(PDFError, match="OCR required"):
        extract_pdf(pdf_bytes(image=True), settings)
    with pytest.raises(PDFError) as error:
        extract_pdf(pdf_bytes(), replace(settings, max_upload_bytes=10))
    assert error.value.status_code == 413
    client = TestClient(create_app(settings))
    response = client.post(
        "/analyze-pdf", files={"file": ("../../evil.joblib", b"not a PDF", "application/pdf")}
    )
    assert response.status_code == 422 and response.json()["detail"]["code"] == "invalid_pdf"
    scanned = client.post("/analyze-pdf", files={"file": ("scan.pdf", pdf_bytes(), "application/pdf")})
    assert scanned.status_code == 422 and scanned.json()["detail"]["code"] == "ocr_required"


def test_pdf_page_mapping(tmp_path):
    text = "The witness described the incident at the railway station. The driver did not stop beside the entrance."
    passages, _ = extract_pdf(pdf_bytes(text), Settings(data_dir=tmp_path, artifact_dir=tmp_path))
    assert passages and all(p.page == 1 for p in passages)
    summary = summarize(passages, 1)
    assert summary[0].page == 1 and summary[0].text in passages[0].text


def test_case_headings_require_explicit_date_role():
    text = (
        "Case title: Artificial A v Artificial B\nCourt: Example Test Court\nDate of Judgment: 12 April 2020"
    )
    result = extract_case_details(text_passages(text))
    assert result["case_title"] == "Artificial A v Artificial B"
    assert result["court"] == "Example Test Court" and result["judgment_date"] == "12 April 2020"
    for evidence in result["evidence"].values():
        assert text[evidence["start_char"] : evidence["end_char"]] == evidence["text"]
    assert (
        extract_case_details(text_passages("The incident occurred on 12 April 2020."))["judgment_date"]
        is None
    )


def test_readable_pdf_endpoint_responds_with_page_references(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", artifact_dir=tmp_path / "artifacts")
    client = TestClient(create_app(settings))
    text = "The witness did not consent to the entry. The guard described the vehicle beside the railway station."
    response = client.post("/analyze-pdf", files={"file": ("input.pdf", pdf_bytes(text), "application/pdf")})
    assert response.status_code == 200
    assert response.json()["summary"][0]["page"] == 1
    assert response.json()["prediction"]["status"] == "missing_model"


def test_pdf_page_limit_and_mixed_scans(tmp_path):
    settings = Settings(data_dir=tmp_path, artifact_dir=tmp_path)
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text(
            (40, 50),
            "The witness described the incident at the factory. The guard did not leave the gate during the evening.",
        )
        document.new_page()
        content = document.tobytes()
    with pytest.raises(PDFError) as error:
        extract_pdf(content, replace(settings, max_pdf_pages=1))
    assert error.value.code == "too_many_pages"
    passages, warnings = extract_pdf(content, settings)
    assert passages[0].page == 1 and any("OCR" in w and "2" in w for w in warnings)


def test_corrupt_local_model_keeps_app_available(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", artifact_dir=tmp_path / "artifacts")
    settings.artifact_dir.mkdir()
    settings.model_path.write_bytes(b"broken")
    settings.model_path.with_suffix(".sha256").write_text("wrong", encoding="ascii")
    service = IntelligenceService(settings)
    assert service.model_info()["artifact_errors"]
    assert service.analyze_text("The witness reported the incident.").prediction.status == "unavailable_model"
