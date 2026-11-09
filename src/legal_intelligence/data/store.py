import json
import sqlite3
from contextlib import contextmanager


@contextmanager
def connect(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    try:
        yield db
        db.commit()
    finally:
        db.close()


def initialize(db):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS cases (
            case_id TEXT PRIMARY KEY, original_id TEXT NOT NULL, document_id TEXT NOT NULL,
            case_title TEXT, court TEXT, judgment_date TEXT, citation TEXT,
            facts_text TEXT NOT NULL, full_text TEXT NOT NULL, act_section_labels TEXT NOT NULL,
            source_url TEXT, source_dataset TEXT NOT NULL, split TEXT NOT NULL,
            source_file TEXT NOT NULL, source_line INTEGER NOT NULL,
            dataset_record_url TEXT NOT NULL, text_kind TEXT NOT NULL,
            sentences TEXT NOT NULL, fact_sentence_numbers TEXT NOT NULL,
            eligible INTEGER NOT NULL DEFAULT 1, exclusion_reason TEXT
        );
        CREATE INDEX IF NOT EXISTS split_index ON cases(split, eligible);
        CREATE TABLE IF NOT EXISTS statutes (
            statute_id TEXT PRIMARY KEY, act_name TEXT NOT NULL, section_number TEXT NOT NULL,
            section_title TEXT, text TEXT NOT NULL, version_date TEXT,
            effective_from TEXT, effective_to TEXT, source_url TEXT,
            dataset_record_url TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)


def metadata(db, key, value):
    db.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", (key, json.dumps(value)))


def load_metadata(db, key):
    row = db.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def iter_cases(db, split=None, eligible=True, fields=None):
    clauses, values = [], []
    if split is not None:
        clauses.append("split = ?")
        values.append(split)
    if eligible:
        clauses.append("eligible = 1")
    allowed = {row[1] for row in db.execute("PRAGMA table_info(cases)")}
    if fields is not None and not set(fields) <= allowed:
        raise ValueError("Unknown internal case field")
    selected = ", ".join(fields) if fields else "*"
    query = (
        f"SELECT {selected} FROM cases"
        + (" WHERE " + " AND ".join(clauses) if clauses else "")
        + " ORDER BY source_line"
    )
    for raw in db.execute(query, values):
        row = dict(raw)
        for field in ("act_section_labels", "sentences", "fact_sentence_numbers"):
            if field in row:
                row[field] = json.loads(row[field])
        yield row


def load_cases(db, split=None, eligible=True, fields=None):
    return list(iter_cases(db, split, eligible, fields))
