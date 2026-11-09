import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("LEGAL_DATA_DIR", "data")))
    artifact_dir: Path = field(default_factory=lambda: Path(os.getenv("LEGAL_ARTIFACT_DIR", "artifacts")))
    max_upload_bytes: int = field(
        default_factory=lambda: int(os.getenv("LEGAL_MAX_UPLOAD_MB", "20")) * 1048576
    )
    max_pdf_pages: int = field(default_factory=lambda: int(os.getenv("LEGAL_MAX_PDF_PAGES", "300")))
    max_text_chars: int = field(default_factory=lambda: int(os.getenv("LEGAL_MAX_TEXT_CHARS", "200000")))
    pdf_min_text_chars: int = field(default_factory=lambda: int(os.getenv("LEGAL_PDF_MIN_TEXT_CHARS", "80")))
    remove_pdf_margins: bool = field(
        default_factory=lambda: os.getenv("LEGAL_REMOVE_PDF_MARGINS", "true").lower() == "true"
    )

    @property
    def db_path(self):
        return self.data_dir / "cases.sqlite3"

    @property
    def model_path(self):
        return self.artifact_dir / "ipc_model.joblib"

    @property
    def index_path(self):
        return self.artifact_dir / "retrieval.joblib"
