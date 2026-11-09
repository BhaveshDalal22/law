"""Artificial fixtures are isolated under pytest temp directories; never shipped as training data."""

import json

import pytest

from legal_intelligence.config import Settings
from legal_intelligence.data.lesicin import prepare
from legal_intelligence.training import TrainConfig, train

A = "Section 34 in The Indian Penal Code"
B = "Section 302 in The Indian Penal Code"
RARE = "Section 120B in The Indian Penal Code"
UNSEEN = "Section 395 in The Indian Penal Code"


@pytest.fixture(scope="session")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("artificial-test-fixtures-only")
    raw = root / "raw"
    raw.mkdir()
    texts = [
        "A group planned together and shared the journey carrying tools towards the warehouse beside the old railway.",
        "A person attacked the victim using a sharp weapon while witnesses watched from a nearby house that evening.",
    ]
    vocab = [
        "amber",
        "bronze",
        "cobalt",
        "denim",
        "emerald",
        "fuchsia",
        "golden",
        "hazel",
        "indigo",
        "jade",
        "khaki",
        "lilac",
        "magenta",
        "navy",
        "ochre",
        "pearl",
        "quartz",
        "ruby",
        "silver",
        "teal",
    ]
    for filename, count, prefix, start in [
        ("train.jsonl", 10, "training", 0),
        ("dev.jsonl", 4, "validation", 10),
        ("test.jsonl", 4, "testing", 14),
    ]:
        rows = []
        for i in range(count):
            text = (
                texts[i % 2]
                + f" The {vocab[start + i]} witness described a distinct location {start + i} beside {prefix}."
            )
            labels = [A if i % 2 == 0 else B]
            if prefix == "training" and i == 0:
                labels.append(RARE)
            if prefix == "testing" and i == 0:
                labels.append(UNSEEN)
            rows.append(
                {
                    "id": f"{prefix}-{i}",
                    "text": [
                        text,
                        "Section 34 IPC was mentioned in the report.",
                        "The court convicted the accused and sentenced him.",
                    ],
                    "labels": labels,
                }
            )
        (raw / filename).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (raw / "secs.jsonl").write_text(
        "\n".join(
            json.dumps({"id": label, "text": ["Artificial statute test text only."]})
            for label in (A, B, RARE, UNSEEN)
        ),
        encoding="utf-8",
    )
    settings = Settings(data_dir=root / "data", artifact_dir=root / "artifacts")
    report = prepare(raw, settings.db_path, root / "dataset.json", verify=False)
    return settings, report, raw


@pytest.fixture(scope="session")
def trained(prepared):
    settings, _report, _raw = prepared
    result = train(
        settings.db_path,
        settings.model_path,
        settings.artifact_dir / "reports",
        TrainConfig(min_label_support=2, min_df=1, max_features=1000, c_values=(1.0,), thresholds=(0.3, 0.5)),
    )
    return settings, result
