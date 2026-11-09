from datetime import UTC, datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from legal_intelligence.artifacts import save_bundle, write_json
from legal_intelligence.data.store import connect, iter_cases, load_metadata
from legal_intelligence.preprocessing import normalize_text
from legal_intelligence.schemas import SearchHit, SourcePassage


def case_passages(row, target_chars=1200):
    offset, start, first, parts = 0, 0, 1, []
    for number, sentence in enumerate(row["sentences"], 1):
        if parts and sum(map(len, parts)) + len(sentence) > target_chars:
            text = "\n\n".join(parts)
            yield SourcePassage(
                text=text,
                source_id=row["case_id"],
                start_char=start,
                end_char=start + len(text),
                sentence_start=first,
                sentence_end=number - 1,
            )
            parts, start, first = [], offset, number
        if not parts:
            start, first = offset, number
        parts.append(sentence)
        offset += len(sentence) + 2
    if parts:
        text = "\n\n".join(parts)
        yield SourcePassage(
            text=text,
            source_id=row["case_id"],
            start_char=start,
            end_char=start + len(text),
            sentence_start=first,
            sentence_end=len(row["sentences"]),
        )


def build_index(db_path, index_path, report_dir, splits=("train",), max_passages=100000):
    passages, cases = [], {}
    truncated = False
    fields = (
        "case_id",
        "case_title",
        "court",
        "judgment_date",
        "citation",
        "source_url",
        "source_dataset",
        "dataset_record_url",
        "source_file",
        "source_line",
        "text_kind",
    )
    with connect(db_path) as db:
        provenance = load_metadata(db, "provenance")
        rows = (
            row
            for split in splits
            for row in iter_cases(db, split, eligible=False, fields=fields + ("sentences",))
        )
        for row in rows:
            for passage in case_passages(row):
                if max_passages and len(passages) >= max_passages:
                    truncated = True
                    break
                if len(passage.text.strip()) < 30:
                    continue
                passages.append(passage.model_dump())
                cases[row["case_id"]] = {field: row[field] for field in fields}
            if truncated:
                break
    if not passages:
        raise ValueError("No judgment passages available; run prepare first.")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=40000,
        dtype=np.float32,
        preprocessor=normalize_text,
    )
    matrix = vectorizer.fit_transform([p["text"] for p in passages]).tocsr()
    info = {
        "created_at": datetime.now(UTC).isoformat(),
        "passages": len(passages),
        "cases": len(cases),
        "splits": splits,
        "max_passages": max_passages,
        "truncated": truncated,
        "dataset_provenance": provenance,
        "sparse_shape": list(matrix.shape),
        "score_meaning": "TF-IDF cosine retrieval similarity; not legal relevance probability.",
        "accuracy": None,
        "accuracy_note": "No labelled query-relevance evaluation set.",
    }
    save_bundle(
        {
            "kind": "retrieval",
            "format_version": 1,
            "vectorizer": vectorizer,
            "matrix": matrix,
            "passages": passages,
            "cases": cases,
            "metadata": info,
        },
        index_path,
    )
    write_json(report_dir / "retrieval.json", info)
    return info


def search(bundle, query, top_k=5):
    query_vector = bundle["vectorizer"].transform([query])
    if query_vector.nnz == 0:
        return []
    similarities = (bundle["matrix"] @ query_vector.T).tocoo()
    order = np.argsort(-similarities.data, kind="stable")
    results, seen = [], set()
    for position in order:
        score = float(similarities.data[position])
        if score <= 0:
            continue
        passage = bundle["passages"][int(similarities.row[position])]
        case_id = passage["source_id"]
        if case_id in seen:
            continue
        seen.add(case_id)
        case = bundle["cases"][case_id]
        results.append(
            SearchHit(
                **case,
                passage=SourcePassage(**passage),
                retrieval_similarity=score,
                source_link_status="available"
                if case["source_url"]
                else "Original judgment source link unavailable",
            )
        )
        if len(results) >= top_k:
            break
    return results
