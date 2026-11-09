"""Verify local servers with real training-corpus text; never read the held-out test split."""

import json
import os
from pathlib import Path

import httpx
import pymupdf
from streamlit.testing.v1 import AppTest

from legal_intelligence.artifacts import write_json
from legal_intelligence.config import Settings
from legal_intelligence.data.store import connect


def main():
    settings = Settings()
    api_url = os.getenv("LEGAL_API_URL", "http://127.0.0.1:8000").rstrip("/")
    with connect(settings.db_path) as db:
        row = db.execute(
            "SELECT facts_text FROM cases WHERE split = ? AND eligible = 1 ORDER BY source_line LIMIT 1",
            ("train",),
        ).fetchone()
    if not row:
        raise RuntimeError("Prepare the real dataset before running runtime verification.")
    text = row[0]
    with httpx.Client(base_url=api_url, timeout=180) as client:
        health = client.get("/health")
        health.raise_for_status()
        info = client.get("/model-info").json()
        assert info["model_available"] and info["retrieval_available"]
        response = client.post("/analyze-text", json={"text": text, "summary_sentences": 3, "top_k": 3})
        response.raise_for_status()
        analysis = response.json()
        assert analysis["prediction"]["model_version"] == info["metadata"]["model_version"]
        assert all(text[s["start_char"] : s["end_char"]] == s["text"] for s in analysis["summary"])
        assert analysis["similar_judgments"]
        assert all(h["source_file"] and h["source_line"] for h in analysis["similar_judgments"])
        search = client.post("/search", json={"query": text[:1000], "top_k": 3})
        search.raise_for_status()
        assert search.json()["status"] == "ok"
        with pymupdf.open() as document:
            page = document.new_page()
            page.insert_textbox(pymupdf.Rect(40, 40, 550, 780), text[:1500], fontsize=10)
            content = document.tobytes()
        pdf = client.post("/analyze-pdf", files={"file": ("runtime-source.pdf", content, "application/pdf")})
        pdf.raise_for_status()
        assert pdf.json()["source_passages"][0]["page"] == 1
    ui_health = httpx.get("http://127.0.0.1:8501/_stcore/health", timeout=30)
    ui_health.raise_for_status()
    ui = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"))
    ui.run(timeout=60)
    assert not ui.exception, str(ui.exception)
    ui.text_area[0].set_value(text[:1500])
    next(button for button in ui.button if button.label == "Analyze").click()
    ui.run(timeout=60)
    assert not ui.exception, str(ui.exception)
    assert "analysis" in ui.session_state
    assert ui.session_state["analysis"]["prediction"]["model_version"] == info["metadata"]["model_version"]
    result = {
        "api_health": health.json(),
        "model_version": info["metadata"]["model_version"],
        "supported_labels": len(info["supported_labels"]),
        "real_text_analysis": response.status_code,
        "real_text_prediction_status": analysis["prediction"]["status"],
        "summary_verbatim_offsets": True,
        "retrieval_source_attribution": True,
        "pdf_analysis": pdf.status_code,
        "streamlit_health": ui_health.status_code,
        "streamlit_analyze_action": "passed",
        "test_split_accessed": False,
    }
    write_json(settings.artifact_dir / "reports" / "runtime.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
