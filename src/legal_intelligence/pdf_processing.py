import hashlib
import re
from collections import Counter

import pymupdf

from legal_intelligence.preprocessing import normalize_text, text_passages


class PDFError(ValueError):
    def __init__(self, message, code="invalid_pdf", status_code=422):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def extract_pdf(content, settings):
    if len(content) > settings.max_upload_bytes:
        raise PDFError("PDF exceeds configured upload limit.", "oversized_upload", 413)
    if not content.startswith(b"%PDF-"):
        raise PDFError("Invalid PDF header.")
    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            if document.needs_pass:
                raise PDFError("Password-protected PDFs are not supported.")
            if len(document) > settings.max_pdf_pages:
                raise PDFError("PDF exceeds configured page limit.", "too_many_pages", 413)
            pages = [page.get_text("text", sort=True) for page in document]
    except PDFError:
        raise
    except Exception as exc:
        raise PDFError("The upload could not be read as a PDF.") from exc
    repeated = set()
    if settings.remove_pdf_margins and len(pages) >= 3:
        counts = Counter()
        for text in pages:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            counts.update(set(lines[:2] + lines[-2:]))
        repeated = {line for line, n in counts.items() if n >= max(3, len(pages) * 0.6)}
    cleaned = []
    for text in pages:
        lines = text.splitlines()
        nonempty = [i for i, line in enumerate(lines) if line.strip()]
        edges = set(nonempty[:2] + nonempty[-2:])
        # Remove repeated margin lines only at page edges, not matching body text.
        cleaned.append(
            normalize_text(
                "\n".join(
                    line for i, line in enumerate(lines) if not (i in edges and line.strip() in repeated)
                )
            )
        )
    if sum(len(re.sub(r"\s", "", t)) for t in cleaned) < settings.pdf_min_text_chars:
        raise PDFError(
            "OCR required: insufficient extractable text. Run OCR locally and upload the searchable PDF.",
            "ocr_required",
        )
    if sum(map(len, cleaned)) > settings.max_text_chars:
        raise PDFError("Extracted PDF text exceeds configured analysis limit.", "oversized_text", 413)
    source_id = "upload:" + hashlib.sha256(content).hexdigest()[:20]
    passages = [
        p
        for number, text in enumerate(cleaned, 1)
        for p in text_passages(text, source_id=source_id, page=number)
    ]
    warnings = []
    empty_pages = [i for i, text in enumerate(cleaned, 1) if len(re.sub(r"\s", "", text)) < 30]
    if empty_pages:
        warnings.append(f"OCR may be required on pages {empty_pages}; only extractable text is analyzed.")
    if repeated:
        warnings.append(f"Removed {len(repeated)} repeated margin strings; this is configurable.")
    return passages, warnings
