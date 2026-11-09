import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def evaluate_labels(gold, scores, supported_labels, thresholds, top_k=(1, 3, 5)):
    """Full-universe metrics: unsupported/unseen gold labels count as false negatives."""
    universe = sorted(set(supported_labels) | {label for labels in gold for label in labels})
    positions = {label: i for i, label in enumerate(universe)}
    truth = np.zeros((len(gold), len(universe)), dtype=np.uint8)
    predicted = np.zeros_like(truth)
    for i, labels in enumerate(gold):
        for label in labels:
            truth[i, positions[label]] = 1
    decisions = scores >= np.asarray(thresholds)
    for j, label in enumerate(supported_labels):
        predicted[:, positions[label]] = decisions[:, j]
    if len(universe) == 1:
        p, r, f, support = precision_recall_fscore_support(
            truth.ravel(), predicted.ravel(), labels=[1], average=None, zero_division=0
        )
        micro, macro_f1 = (p[0], r[0], f[0]), f[0]
    else:
        micro = precision_recall_fscore_support(truth, predicted, average="micro", zero_division=0)
        macro_f1 = precision_recall_fscore_support(truth, predicted, average="macro", zero_division=0)[2]
        p, r, f, support = precision_recall_fscore_support(truth, predicted, average=None, zero_division=0)
    ranking = np.argsort(-scores, axis=1, kind="stable")
    top = {}
    for requested in top_k:
        k = min(requested, len(supported_labels))
        hits = [
            len({supported_labels[j] for j in ranking[i, :k]} & set(labels)) for i, labels in enumerate(gold)
        ]
        top[str(requested)] = {
            "effective_k": k,
            "precision_at_k": float(np.mean(np.asarray(hits) / max(k, 1))),
            "recall_at_k": float(np.mean([h / max(len(set(labels)), 1) for h, labels in zip(hits, gold)])),
            "hit_rate_at_k": float(np.mean(np.asarray(hits) > 0)),
        }
    occurrences = sum(len(set(labels)) for labels in gold)
    covered = sum(len(set(labels) & set(supported_labels)) for labels in gold)
    return {
        "documents": len(gold),
        "micro_precision": float(micro[0]),
        "micro_recall": float(micro[1]),
        "micro_f1": float(micro[2]),
        "macro_f1": float(macro_f1),
        "per_label": {
            label: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(support[i]),
                "supported": label in supported_labels,
            }
            for i, label in enumerate(universe)
        },
        "top_k": top,
        "gold_label_occurrence_coverage": covered / max(occurrences, 1),
        "unsupported_gold_labels": sorted(set(universe) - set(supported_labels)),
        "definitions": {
            "micro": "Aggregate TP, FP, FN across documents and full label universe.",
            "macro_f1": "Unweighted label F1 mean, including unsupported gold labels.",
            "precision_at_k": "Mean number of true labels in top k / effective k.",
            "recall_at_k": "Mean number of true labels in top k / number of gold labels; empty gold contributes zero.",
            "hit_rate_at_k": "Fraction of documents with at least one true label in top k.",
            "ranking": "Ranks supported labels regardless of decision thresholds.",
        },
    }
