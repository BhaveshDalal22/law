"""Public, pinned LeSICiN release. Downloads data only, never pretrained binaries."""

import hashlib
import json
import shutil
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

RECORD_URL = "https://zenodo.org/records/6053791"
API_URL = "https://zenodo.org/api/records/6053791"
FILES = {
    "train.jsonl": "a1994d1aaa8ae7a987fcb1c2e2dc52f1",
    "dev.jsonl": "237b1776f5a535eb3f079bfd0ea0b200",
    "test.jsonl": "09a554405680830c6d3818320eb055af",
    "secs.jsonl": "b2768e06377ab032df2d8f8c323886ff",
    "label_vocab.json": "5758e3117c37788045308e03a531a03d",
}


def hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(chunk)
            sha.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def file_url(name):
    return f"{API_URL}/files/{name}/content"


def manifest(raw_dir):
    entries = []
    for name, expected in FILES.items():
        path = raw_dir / name
        if not path.exists():
            raise ValueError(f"Missing {path}; use download or import-lesicin.")
        checksums = hashes(path)
        if checksums["md5"] != expected:
            raise ValueError(f"Checksum mismatch for {path}; expected official release 6053791.")
        entries.append({"filename": name, "url": file_url(name), "bytes": path.stat().st_size, **checksums})
    result = {
        "dataset": "LeSICiN",
        "record_url": RECORD_URL,
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Shounak Paul, Pawan Goyal, Saptarshi Ghosh; LeSICiN, AAAI 2022",
        "verified_at": datetime.now(UTC).isoformat(),
        "files": entries,
    }
    (raw_dir / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def download(raw_dir):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Persist the rights/access metadata actually checked at download time.
    with urllib.request.urlopen(API_URL, timeout=60) as response:
        metadata = json.load(response)
    if (
        metadata["metadata"].get("access_right") != "open"
        or metadata["metadata"].get("license", {}).get("id") != "cc-by-4.0"
    ):
        raise ValueError("Dataset access or licence changed. Review the release before downloading.")
    (raw_dir / "zenodo-record.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for name, expected in FILES.items():
        target = raw_dir / name
        if target.exists() and hashes(target)["md5"] == expected:
            print(f"Verified existing {name}", flush=True)
            continue
        partial = target.with_suffix(".part")
        for attempt in range(3):
            try:
                print(f"Downloading {name} (attempt {attempt + 1})", flush=True)
                with (
                    urllib.request.urlopen(file_url(name), timeout=90) as response,
                    partial.open("wb") as out,
                ):
                    shutil.copyfileobj(response, out, length=1024 * 1024)
                if hashes(partial)["md5"] != expected:
                    raise ValueError(f"Checksum mismatch: {name}")
                partial.replace(target)
                break
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    return manifest(raw_dir)


def import_local(source, raw_dir):
    source, raw_dir = Path(source), Path(raw_dir)
    # Validate before replacing an existing local release.
    manifest(source)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if (source / name).resolve() != (raw_dir / name).resolve():
            shutil.copy2(source / name, raw_dir / name)
    return manifest(raw_dir)
