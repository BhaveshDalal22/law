import logging
import re

from legal_intelligence.artifacts import load_bundle
from legal_intelligence.config import Settings
from legal_intelligence.preprocessing import (
    extract_case_details,
    extract_references,
    select_fact_narrative,
    sentences,
    text_passages,
)
from legal_intelligence.retrieval import search
from legal_intelligence.schemas import AnalysisResponse, Prediction, SectionSuggestion
from legal_intelligence.summarization import summarize
from legal_intelligence.training import predict_scores

SETUP = "Run: legal-intelligence download; legal-intelligence prepare; legal-intelligence train; legal-intelligence index."


class IntelligenceService:
    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self.model, self.index, self.errors = None, None, {}
        for attribute, path, kind in [
            ("model", self.settings.model_path, "ipc_classifier"),
            ("index", self.settings.index_path, "retrieval"),
        ]:
            if path.exists():
                try:
                    setattr(self, attribute, load_bundle(path, kind))
                except Exception:
                    logging.getLogger(__name__).exception("Failed to load local %s artifact", attribute)
                    self.errors[attribute] = (
                        "Local artifact unreadable or incompatible. Rebuild it using the CLI."
                    )

    def model_info(self):
        return {
            "name": "AI-Based Legal Case Intelligence",
            "model_available": self.model is not None,
            "dataset_available": self.settings.db_path.exists(),
            "retrieval_available": self.index is not None,
            "supported_labels": self.model["labels"] if self.model else [],
            "metadata": self.model["metadata"] if self.model else None,
            "retrieval_metadata": self.index["metadata"] if self.index else None,
            "category_classifier_available": False,
            "trained_ner_available": False,
            "statutory_reference": "https://indiacode.gov.in/",
            "setup_guidance": SETUP,
            "artifact_errors": self.errors,
        }

    def predict(self, passages):
        if self.model is None:
            unavailable = "model" in self.errors
            return Prediction(
                status="unavailable_model" if unavailable else "missing_model",
                message=self.errors.get("model", SETUP),
            )
        selected = select_fact_narrative([p.text for p in sentences(passages)])
        features = "\n\n".join(text for _, text in selected)
        version = self.model["metadata"]["model_version"]
        if not features or self.model["pipeline"][:-1].transform([features]).nnz == 0:
            return Prediction(
                status="insufficient_evidence",
                model_version=version,
                message="No usable narrative or in-vocabulary evidence after preprocessing.",
            )
        values = predict_scores(self.model["pipeline"], [features], self.model["labels"])[0]
        above, below = [], []
        for i in values.argsort()[::-1]:
            label, score, threshold = (
                self.model["labels"][i],
                float(values[i]),
                float(self.model["thresholds"][i]),
            )
            match = re.match(r"Section (\S+) in", label)
            item = SectionSuggestion(
                label=label,
                score=score,
                threshold=threshold,
                above_threshold=score >= threshold,
                section_number=match[1] if match else None,
            )
            (above if item.above_threshold else below).append(item)
        return Prediction(
            status="ok" if above else "insufficient_evidence",
            suggestions=above,
            below_threshold=below[:3],
            model_version=version,
            message=None if above else "No section passes its validation-selected threshold.",
        )

    def search(self, query, top_k=5):
        if self.index is None:
            return [], "unavailable_index" if "index" in self.errors else "missing_index"
        results = search(self.index, query, top_k)
        return results, "ok" if results else "no_results"

    def analyze(self, passages, summary_length=5, top_k=5, warnings=None):
        references = extract_references(passages)
        query = "\n\n".join(p.text for p in passages)
        similar, status = self.search(query, top_k)
        notices = list(warnings or []) + [
            "Historical IPC dataset labels only; no automatic IPC-to-BNS conversion.",
            "Predictions use a rule-selected narrative and are research suggestions, not legal advice.",
            "Regex detects mentions and date-like strings; dates are not verified event or judgment dates.",
            "Summary sentences are copied from the input; no rhetorical-role classification is performed.",
        ]
        if self.index and self.index["metadata"].get("truncated"):
            notices.append(
                "Retrieval index is limited to the recorded passage cap; see model-info for corpus coverage."
            )
        return AnalysisResponse(
            prediction=self.predict(passages),
            detected_references=[r for r in references if r.kind == "provision"],
            dates=[r for r in references if r.kind == "date"],
            case_details={
                **extract_case_details(passages),
                "citations": [r.model_dump() for r in references if r.kind == "case_citation"],
            },
            similar_judgments=similar,
            retrieval_status=status,
            summary=summarize(passages, summary_length),
            source_passages=passages,
            warnings=notices,
        )

    def analyze_text(self, text, summary_length=5, top_k=5):
        return self.analyze(text_passages(text), summary_length, top_k)
