import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from legal_intelligence.preprocessing import sentences


def summarize(passages, length=5, max_candidates=1200):
    candidates = sentences(passages)[:max_candidates]
    if len(candidates) <= length:
        return candidates
    try:
        matrix = TfidfVectorizer(ngram_range=(1, 1), max_features=30000).fit_transform(
            [sentence.text for sentence in candidates]
        )
    except ValueError:
        return candidates[:length]
    # Only a vocabulary-length centroid is dense, never an all-sentence similarity matrix.
    centroid = np.asarray(matrix.mean(axis=0)).ravel()
    norm = np.linalg.norm(centroid)
    scores = np.asarray(matrix @ (centroid / max(norm, 1e-12))).ravel()
    chosen, redundancy = [], np.zeros(len(candidates))
    for _ in range(min(length, len(candidates))):
        ranking = 0.75 * scores - 0.25 * redundancy
        ranking[chosen] = -np.inf
        index = int(np.argmax(ranking))
        chosen.append(index)
        redundancy = np.maximum(redundancy, (matrix @ matrix[index].T).toarray().ravel())
    return [candidates[i] for i in sorted(chosen)]
