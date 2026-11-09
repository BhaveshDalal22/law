"""Run with: python -m streamlit run streamlit_app.py"""

import os

import httpx
import streamlit as st

st.set_page_config(page_title="Legal Case Intelligence", page_icon="⚖️", layout="wide")
API_URL = os.getenv("LEGAL_API_URL", "http://127.0.0.1:8000").rstrip("/")


def api(method, path, **kwargs):
    try:
        response = httpx.request(method, API_URL + path, timeout=180, **kwargs)
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            st.error(detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail))
            return None
        return response.json()
    except httpx.HTTPError:
        st.error(
            "Backend unavailable. Start: python -m uvicorn legal_intelligence.api:app --host 127.0.0.1 --port 8000"
        )
        return None


def location(passage):
    values = []
    if passage.get("page") is not None:
        values.append(f"page {passage['page']}")
    if passage.get("paragraph") is not None:
        values.append(f"paragraph {passage['paragraph']}")
    if passage.get("sentence_start") is not None:
        values.append(f"dataset sentences {passage['sentence_start']}–{passage['sentence_end']}")
    values.append(f"characters {passage['start_char']}–{passage['end_char']}")
    return ", ".join(values)


def render_hits(hits, status):
    if not hits:
        if status in ("missing_index", "unavailable_index"):
            st.info("Search index unavailable. Run: legal-intelligence index, then restart the backend.")
        else:
            st.info("No matching judgments found.")
    for hit in hits:
        with st.expander(
            f"{hit.get('case_title') or hit['case_id']} · similarity {hit['retrieval_similarity']:.3f}"
        ):
            st.write(hit["passage"]["text"])
            st.caption(location(hit["passage"]))
            st.caption(f"{hit['source_dataset']} · {hit['source_file']}, line {hit['source_line']}")
            st.caption(
                f"Court: {hit.get('court') or 'unavailable'} · Judgment date: {hit.get('judgment_date') or 'unavailable'}"
            )
            if hit.get("source_url"):
                st.link_button("Original judgment", hit["source_url"])
            else:
                st.caption("Original judgment source link unavailable.")
            if hit.get("dataset_record_url"):
                st.link_button("Dataset provenance", hit["dataset_record_url"])
            st.caption(
                "Dataset excerpt; completeness as a full judgment is unverified. Cosine similarity is not legal relevance probability."
            )


st.title("AI-Based Legal Case Intelligence")
st.write("Explore IPC section suggestions, source passages, and extractive summaries with local CPU models.")
st.caption(
    "Research aid. IPC labels remain IPC labels. Verify statutory applicability and text at India Code."
)
info = api("GET", "/model-info")
with st.sidebar:
    st.header("Model & coverage")
    if info:
        metadata = info.get("metadata") or {}
        st.write("Model: " + metadata.get("model_version", "not trained"))
        st.write(f"Supported labels: {len(info['supported_labels'])}")
        st.caption("Statute version and effective dates: unavailable in this release.")
        with st.expander("Supported IPC labels"):
            for label in info["supported_labels"]:
                st.text(label)
        st.checkbox(
            "Civil/criminal category classifier",
            value=False,
            disabled=True,
            help="Disabled until a suitable labelled dataset and real trained model exist.",
        )
        st.link_button("Official statutory reference", "https://indiacode.gov.in/")
        if not info["model_available"] or not info["dataset_available"]:
            st.info("Dataset/model setup required. Text extraction and summaries remain available.")
            st.code(
                "legal-intelligence download\nlegal-intelligence prepare\nlegal-intelligence train\nlegal-intelligence evaluate\nlegal-intelligence index"
            )
            st.caption("Restart the backend after building artifacts.")
        if info.get("artifact_errors"):
            st.error(info["artifact_errors"])
        with st.expander("Retrieval corpus"):
            st.json(info.get("retrieval_metadata") or {"status": "Index missing"})
    summary_length = st.slider("Summary sentences", 1, 20, 5)
    top_k = st.slider("Similar judgments", 1, 10, 5)

analysis_tab, search_tab = st.tabs(["Analyze case facts or PDF", "Search judgments"])
with analysis_tab:
    mode = st.radio("Input", ["Case facts", "Judgment PDF"], horizontal=True)
    text, upload = "", None
    if mode == "Case facts":
        text = st.text_area(
            "Case facts", height=230, placeholder="Enter the factual narrative to analyze…", max_chars=200000
        )
    else:
        upload = st.file_uploader("Upload a searchable judgment PDF", type=["pdf"])
        st.caption("Scanned PDFs need OCR. Upload limit defaults to 20 MB; analysis stays on your backend.")
    if st.button("Analyze", type="primary"):
        if mode == "Case facts" and not text.strip():
            st.warning("Enter case facts first.")
        elif mode == "Judgment PDF" and upload is None:
            st.warning("Select a PDF first.")
        else:
            with st.spinner("Analyzing source text…"):
                if mode == "Case facts":
                    result = api(
                        "POST",
                        "/analyze-text",
                        json={"text": text, "summary_sentences": summary_length, "top_k": top_k},
                    )
                else:
                    result = api(
                        "POST",
                        "/analyze-pdf",
                        params={"summary_sentences": summary_length, "top_k": top_k},
                        files={"file": ("judgment.pdf", upload.getvalue(), "application/pdf")},
                    )
            if result:
                st.session_state["analysis"] = result
    result = st.session_state.get("analysis")
    if result:
        st.subheader("Predicted section suggestions")
        prediction = result["prediction"]
        if prediction["status"] != "ok":
            st.info(prediction.get("message") or prediction["status"])
        if prediction["suggestions"]:
            st.dataframe(prediction["suggestions"], hide_index=True)
        st.caption(prediction["score_meaning"])
        if prediction["below_threshold"]:
            with st.expander("Below-threshold ranking — insufficient evidence for these labels"):
                st.dataframe(prediction["below_threshold"], hide_index=True)
        st.subheader("Extractive summary")
        for sentence in result["summary"]:
            st.write(sentence["text"])
            st.caption(location(sentence))
        left, right = st.columns(2)
        with left:
            st.subheader("Detected provision references")
            st.caption("Regex mentions in the input, separate from ML suggestions.")
            if result["detected_references"]:
                st.dataframe(result["detected_references"], hide_index=True)
            else:
                st.write("No references detected.")
        with right:
            st.subheader("Date-like mentions & case details")
            if result["dates"]:
                st.dataframe(result["dates"], hide_index=True)
            st.json(result["case_details"])
        st.subheader("Similar judgment excerpts")
        render_hits(result["similar_judgments"], result["retrieval_status"])
        with st.expander("Source text and analysis limits"):
            for warning in result["warnings"]:
                st.write(warning)
            st.json(result["source_passages"])
        st.download_button(
            "Download analysis JSON",
            data=__import__("json").dumps(result, indent=2),
            file_name="legal-analysis.json",
            mime="application/json",
        )

with search_tab:
    query = st.text_input("Search available judgment passages")
    if st.button("Search"):
        if not query.strip():
            st.warning("Enter a search query.")
        else:
            result = api("POST", "/search", json={"query": query, "top_k": top_k})
            if result:
                render_hits(result["results"], result["status"])
