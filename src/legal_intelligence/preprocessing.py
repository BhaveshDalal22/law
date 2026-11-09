"""Deterministic normalization and regex extraction; this is not trained NER."""

import re
import unicodedata
from functools import lru_cache
from typing import Protocol

import spacy

from legal_intelligence.schemas import Reference, SourcePassage

SECTION_NUMBER = r"\d{1,3}[A-Z]?(?:\s*[- ]\s*[A-Z]\b)?(?:\s*\([a-zA-Z0-9]+\))*"
SECTION_LIST = rf"{SECTION_NUMBER}(?:\s*(?:,|/|&|\band\b|\br\.?\s*/\s*w\.?|\bto\b|[-–])\s*{SECTION_NUMBER})*"
PROVISION_RE = re.compile(
    rf"\b(?:sections?|secs?\.?|s\.|articles?|arts?\.?)\s*{SECTION_LIST}"
    rf"(?:\s+(?:of\s+(?:the\s+)?)?(?:Indian Penal Code|I\.?\s*P\.?\s*C\.?|"
    rf"Bharatiya Nyaya Sanhita|BNS|Cr\.?\s*P\.?\s*C\.?|Constitution(?:\s+of\s+India)?))?"
    rf"|\b(?:IPC|BNS|CrPC)\s*{SECTION_LIST}",
    re.IGNORECASE,
)
DATE_RE = re.compile(
    r"\b(?:\d{1,2}[./-]\d{1,2}[./-](?:\d{4}|\d{2})|\d{4}-\d{2}-\d{2}|"
    r"\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|"
    r"July?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s*,?\s*\d{4})\b",
    re.IGNORECASE,
)
CITATION_RE = re.compile(
    r"\b(?:AIR\s+\d{4}\s+[A-Z][A-Za-z ]{1,20}\s+\d+|\(?\d{4}\)?\s*\(\d+\)\s*SCC\s*\d+)\b"
)
# Exclude judicial outcomes and procedural discussion from model features. This is a conservative
# rule, not a rhetorical-role model. Raw excerpts remain available only to retrieval/summary.
OUTCOME_RE = re.compile(
    r"\b(?:convict\w*|acquit\w*|sentenced?|sentencing|verdict|judgment|judgement|"
    r"appeal\w*|petition\w*|court|judge|held|holding|findings?|dismiss\w*|allowed|ordered|"
    r"guilty|not\s+guilty|punish\w*)\b",
    re.IGNORECASE,
)
DECISIVE_RE = re.compile(
    r"\b(?:convict\w*|acquit\w*|sentenced?|sentencing|verdict|"
    r"appeal\s+(?:is\s+)?(?:allowed|dismissed)|we\s+(?:hold|find)|"
    r"held\s+that|found\s+guilty)\b",
    re.IGNORECASE,
)


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def mask_references(text):
    # Mask complete reference spans, including lists, so explicit labels cannot be copied.
    return PROVISION_RE.sub(" legal_provision ", normalize_text(text))


def preprocess_texts(texts, reference_pattern=None, reference_flags=0):
    pattern = re.compile(reference_pattern, reference_flags) if reference_pattern else PROVISION_RE
    return [pattern.sub(" legal_provision ", normalize_text(t)) for t in texts]


def select_fact_narrative(sentences, max_sentences=40, max_chars=12000):
    selected = []
    for i, sentence in enumerate(sentences):
        if DECISIVE_RE.search(sentence):
            break
        if OUTCOME_RE.search(sentence):
            continue
        text = normalize_text(sentence)
        if len(text) >= 20:
            selected.append((i + 1, text))
        if len(selected) >= max_sentences:
            break
    # Truncate at a complete sentence boundary, keeping provenance of every selected sentence.
    kept, length = [], 0
    for number, text in selected:
        if length + len(text) > max_chars:
            break
        kept.append((number, text))
        length += len(text) + 2
    return kept


@lru_cache(maxsize=1)
def sentence_pipeline():
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    nlp.max_length = 1000000
    return nlp


def text_passages(text, source_id="input", page=None):
    passages = []
    for number, match in enumerate(re.finditer(r"\S[\s\S]*?(?=\n\s*\n|\Z)", text), 1):
        end = match.end()
        while end > match.start() and text[end - 1].isspace():
            end -= 1
        passages.append(
            SourcePassage(
                text=text[match.start() : end],
                source_id=source_id,
                page=page,
                paragraph=number,
                start_char=match.start(),
                end_char=end,
            )
        )
    return passages


def sentences(passages):
    result = []
    nlp = sentence_pipeline()
    for passage in passages:
        for sentence in nlp(passage.text).sents:
            start, end = sentence.start_char, sentence.end_char
            while start < end and passage.text[start].isspace():
                start += 1
            while end > start and passage.text[end - 1].isspace():
                end -= 1
            if end > start:
                result.append(
                    passage.model_copy(
                        update={
                            "text": passage.text[start:end],
                            "start_char": passage.start_char + start,
                            "end_char": passage.start_char + end,
                        }
                    )
                )
    return result


def extract_references(passages):
    extracted = []
    for passage in passages:
        for kind, pattern in [("provision", PROVISION_RE), ("date", DATE_RE), ("case_citation", CITATION_RE)]:
            for match in pattern.finditer(passage.text):
                extracted.append(
                    Reference(
                        text=match.group(),
                        kind=kind,
                        page=passage.page,
                        paragraph=passage.paragraph,
                        start_char=passage.start_char + match.start(),
                        end_char=passage.start_char + match.end(),
                    )
                )
    return extracted


def extract_case_details(passages):
    """Read explicit document headings only; never infer a date's role from an unlabelled mention."""
    result = {"case_title": None, "court": None, "judgment_date": None, "evidence": {}}
    patterns = {
        "case_title": re.compile(r"^\s*case\s+title\s*:\s*(?P<value>[^\n]+)", re.IGNORECASE | re.MULTILINE),
        "court": re.compile(
            r"^\s*(?:court\s*:\s*(?P<label>[^\n]+)|(?P<heading>IN THE [^\n]*COURT[^\n]*))",
            re.IGNORECASE | re.MULTILINE,
        ),
        "judgment_date": re.compile(
            r"^\s*(?:date\s+of\s+judg(?:e)?ment|judg(?:e)?ment\s+date)\s*:\s*(?P<value>[^\n]+)",
            re.IGNORECASE | re.MULTILINE,
        ),
    }
    for passage in passages:
        for field, pattern in patterns.items():
            if result[field] is not None:
                continue
            match = pattern.search(passage.text)
            if not match:
                continue
            value = (
                match.groupdict().get("value")
                or match.groupdict().get("label")
                or match.groupdict().get("heading")
            ).strip()
            if field == "judgment_date":
                date = DATE_RE.fullmatch(value)
                if not date:
                    continue
            result[field] = value
            result["evidence"][field] = {
                "text": match.group(),
                "page": passage.page,
                "paragraph": passage.paragraph,
                "start_char": passage.start_char + match.start(),
                "end_char": passage.start_char + match.end(),
                "method": "explicit_heading_regex",
            }
    result["extraction_method"] = "Explicit-heading and limited citation regex; unavailable metadata is null."
    return result


class LegalNERAdapter(Protocol):
    """Extension point for an independently trained NER model; not instantiated by default."""

    def extract(self, passages: list[SourcePassage]) -> list[dict]: ...
