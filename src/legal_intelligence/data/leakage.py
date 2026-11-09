"""Exact identity/text checks plus bounded near-duplicate screening across supplied splits."""

import hashlib
import re
from collections import defaultdict

import numpy as np

from legal_intelligence.preprocessing import mask_references, normalize_text


def fingerprint(text):
    return hashlib.sha256(normalize_text(text).lower().encode()).hexdigest()


def simhash(text):
    tokens = sorted(set(re.findall(r"\w+", mask_references(text).lower())))
    if not tokens:
        return 0
    values = np.fromiter(
        (int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), "big") for t in tokens),
        dtype=np.uint64,
    )
    bits = ((values[:, None] >> np.arange(64, dtype=np.uint64)) & 1).astype(np.int32)
    signs = bits.sum(axis=0) >= len(tokens) / 2
    return sum(int(value) << i for i, value in enumerate(signs))


def leakage_audit(rows):
    exact = defaultdict(list)
    bands = defaultdict(list)
    pairs, exclusions, seen_pairs = [], {}, set()
    ordered = sorted(
        rows, key=lambda r: ({"train": 0, "validation": 1, "test": 2}[r["split"]], r["source_line"])
    )
    known = {}
    for row in ordered:
        identity = row["case_id"]
        keys = [("id", row["original_id"]), ("text", fingerprint(row["full_text"]))]
        if row["facts_text"]:
            keys.append(("masked_facts", fingerprint(mask_references(row["facts_text"]))))
        for key in keys:
            for previous in exact[key]:
                if previous["case_id"] == identity:
                    continue
                pair = (previous["case_id"], identity)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    pairs.append(
                        {
                            "first": pair[0],
                            "second": pair[1],
                            "kind": key[0],
                            "cross_split": previous["split"] != row["split"],
                        }
                    )
                exclusions[identity] = f"duplicate:{previous['case_id']}"
            exact[key].append(row)
        signature = simhash(row["facts_text"] or row["full_text"][:12000])
        candidates = set()
        for band in range(4):
            candidates.update(bands[(band, (signature >> (16 * band)) & 65535)])
        tokens = None
        for previous_id in candidates:
            previous, old_signature = known[previous_id]
            if previous["split"] == row["split"] or (signature ^ old_signature).bit_count() > 3:
                continue
            if tokens is None:
                tokens = set(
                    re.findall(r"\w+", mask_references(row["facts_text"] or row["full_text"][:12000]).lower())
                )
            old = set(
                re.findall(
                    r"\w+", mask_references(previous["facts_text"] or previous["full_text"][:12000]).lower()
                )
            )
            similarity = len(tokens & old) / max(len(tokens | old), 1)
            pair = (previous_id, identity)
            if similarity >= 0.8 and pair not in seen_pairs:
                seen_pairs.add(pair)
                pairs.append(
                    {
                        "first": previous_id,
                        "second": identity,
                        "kind": "near_duplicate",
                        "cross_split": True,
                        "token_jaccard": similarity,
                    }
                )
                exclusions[identity] = f"related:{previous_id}"
        known[identity] = (row, signature)
        for band in range(4):
            bands[(band, (signature >> (16 * band)) & 65535)].append(identity)
    return {
        "pairs": pairs,
        "exclusions": exclusions,
        "cross_split_pairs": sum(p["cross_split"] for p in pairs),
        "related_check": "64-bit token-set SimHash, 4 bands, Hamming <=3, token Jaccard >=0.8",
        "limitations": "Approximate text screening cannot identify all related proceedings. No docket/group metadata supplied.",
        "policy": "Keep original split. Exclude later duplicate records in train, validation, test priority order; no reassignment.",
    }
