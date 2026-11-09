"""Only the operator-controlled local artifact directory is a trust boundary."""

import hashlib
import json
from pathlib import Path

import joblib


def save_bundle(bundle, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    joblib.dump(bundle, temporary, compress=3)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    temporary.replace(path)
    path.with_suffix(".sha256").write_text(digest, encoding="ascii")


def load_bundle(path, kind):
    path = Path(path)
    # A checksum detects corruption, not malicious artifacts. Never accept an upload here.
    expected = path.with_suffix(".sha256").read_text(encoding="ascii").strip()
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Local artifact checksum mismatch; rebuild the artifact.")
    bundle = joblib.load(path)
    if bundle.get("kind") != kind or bundle.get("format_version") != 1:
        raise ValueError("Unsupported local artifact format")
    return bundle


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
