import logging
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from legal_intelligence.config import Settings
from legal_intelligence.pdf_processing import PDFError, extract_pdf
from legal_intelligence.schemas import AnalysisResponse, SearchRequest, TextRequest
from legal_intelligence.service import IntelligenceService

logger = logging.getLogger(__name__)


def create_app(settings=None):
    settings = settings or Settings()
    application = FastAPI(title="AI-Based Legal Case Intelligence", version="0.1.0")
    application.state.service = IntelligenceService(settings)

    @application.get("/health")
    def health():
        service = application.state.service
        return {
            "status": "ok",
            "model_available": service.model is not None,
            "retrieval_available": service.index is not None,
            "artifact_errors": service.errors,
        }

    @application.get("/model-info")
    def model_info():
        return application.state.service.model_info()

    @application.post("/analyze-text", response_model=AnalysisResponse)
    def analyze_text(request: TextRequest):
        if not request.text.strip():
            raise HTTPException(422, detail={"code": "empty_text", "message": "Enter non-empty case facts."})
        if len(request.text) > settings.max_text_chars:
            raise HTTPException(
                413, detail={"code": "oversized_text", "message": "Text exceeds configured limit."}
            )
        return application.state.service.analyze_text(request.text, request.summary_sentences, request.top_k)

    @application.post("/analyze-pdf", response_model=AnalysisResponse)
    def analyze_pdf(
        file: Annotated[UploadFile, File()],
        summary_sentences: Annotated[int, Query(ge=1, le=20)] = 5,
        top_k: Annotated[int, Query(ge=1, le=20)] = 5,
    ):
        # Read at most limit+1; no client-supplied filename is used as a filesystem path.
        try:
            content = file.file.read(settings.max_upload_bytes + 1)
            passages, warnings = extract_pdf(content, settings)
            return application.state.service.analyze(passages, summary_sentences, top_k, warnings)
        except PDFError as exc:
            raise HTTPException(exc.status_code, detail={"code": exc.code, "message": str(exc)}) from exc
        finally:
            file.file.close()

    @application.post("/search")
    def search(request: SearchRequest):
        if not request.query.strip():
            raise HTTPException(422, detail={"code": "empty_query", "message": "Enter a search query."})
        hits, status = application.state.service.search(request.query, request.top_k)
        return {
            "status": status,
            "results": hits,
            "score_meaning": "TF-IDF cosine retrieval similarity; not legal relevance probability.",
        }

    return application


app = create_app()
