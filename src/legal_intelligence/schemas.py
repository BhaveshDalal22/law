from typing import Literal

from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200000)
    summary_sentences: int = Field(default=5, ge=1, le=20)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20000)
    top_k: int = Field(default=5, ge=1, le=20)


class SourcePassage(BaseModel):
    text: str
    source_id: str
    page: int | None = None
    paragraph: int | None = None
    start_char: int = 0
    end_char: int = 0
    sentence_start: int | None = None
    sentence_end: int | None = None


class Reference(BaseModel):
    text: str
    kind: Literal["provision", "date", "case_citation"]
    method: str = "regex"
    page: int | None = None
    paragraph: int | None = None
    start_char: int
    end_char: int


class SectionSuggestion(BaseModel):
    label: str
    score: float
    threshold: float
    above_threshold: bool
    act_name: str = "The Indian Penal Code"
    section_number: str | None = None
    statute_version: str | None = None
    effective_from: str | None = None
    effective_to: str | None = None


class Prediction(BaseModel):
    status: Literal["ok", "insufficient_evidence", "missing_model", "unavailable_model"]
    suggestions: list[SectionSuggestion] = Field(default_factory=list)
    below_threshold: list[SectionSuggestion] = Field(default_factory=list)
    model_version: str | None = None
    score_meaning: str = "Uncalibrated model score; not legal certainty or a calibrated confidence."
    message: str | None = None


class SearchHit(BaseModel):
    case_id: str
    case_title: str | None = None
    court: str | None = None
    judgment_date: str | None = None
    citation: str | None = None
    source_url: str | None = None
    source_dataset: str
    dataset_record_url: str | None = None
    source_file: str | None = None
    source_line: int | None = None
    text_kind: str
    passage: SourcePassage
    retrieval_similarity: float
    source_link_status: str


class AnalysisResponse(BaseModel):
    prediction: Prediction
    detected_references: list[Reference]
    dates: list[Reference]
    case_details: dict
    similar_judgments: list[SearchHit]
    retrieval_status: str
    summary: list[SourcePassage]
    source_passages: list[SourcePassage]
    warnings: list[str]
