"""Publish validated dataset v3 files to immutable GCS locations."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

FINAL_FILES = ("manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl", "quality_report.md")
DRAFT_FILES = (
    "manifest.json",
    "drafts.jsonl",
    "review_queue.jsonl",
    "review_template.jsonl",
    "synthetic_briefs.jsonl",
    "brief_manifest.json",
    "HUMAN_REVIEW.md",
)


def _gcloud(*args: str, allow_missing: bool = False) -> bytes | None:
    result = subprocess.run(("gcloud", "storage", *args), capture_output=True, check=False)
    if result.returncode:
        if allow_missing and result.returncode == 1:
            return None
        raise RuntimeError(
            f"gcloud storage {' '.join(args[:1])} failed: {result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def _put_immutable(path: Path, uri: str) -> None:
    existing = _gcloud("cat", uri, allow_missing=True)
    if existing is not None:
        if hashlib.sha256(existing).digest() != hashlib.sha256(path.read_bytes()).digest():
            raise ValueError(f"Remote object differs; refusing to overwrite: {uri}")
        return
    _gcloud("cp", "--if-generation-match=0", str(path), uri)
    uploaded = _gcloud("cat", uri)
    if hashlib.sha256(uploaded or b"").digest() != hashlib.sha256(path.read_bytes()).digest():
        raise RuntimeError(f"Upload verification failed: {uri}")


def _bucket(project_id: str) -> str:
    if not project_id or not all(c.isalnum() or c == "-" for c in project_id):
        raise ValueError("A valid GCP project ID is required for automatic dataset upload")
    return f"gs://{project_id}-dataset"


def upload_draft(output_dir: Path, project_id: str) -> str:
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    draft_hash = manifest["drafts_sha256"]
    if hashlib.sha256((output_dir / "drafts.jsonl").read_bytes()).hexdigest() != draft_hash:
        raise ValueError("Draft hash differs from manifest")
    prefix = f"{_bucket(project_id)}/staging/{draft_hash}"
    for name in DRAFT_FILES:
        path = output_dir / name
        if path.exists():
            _put_immutable(path, f"{prefix}/{name}")
    return prefix


def upload_final(output_dir: Path, project_id: str) -> str:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_validation") != "passed" or manifest.get("split_counts") != {
        "train": 168,
        "validation": 24,
        "test": 48,
    }:
        raise ValueError("Dataset has not passed schema and split validation")
    if not (manifest.get("human_semantic_review") is True and manifest.get("review_status") == "reviewed"):
        if not (
            manifest.get("training_eligibility") == "schema_only_experiment"
            and manifest.get("review_status") == "pending"
        ):
            raise ValueError("Dataset is neither reviewed nor an explicit schema-only experiment")
    hashes = manifest.get("training_files_sha256") or manifest.get("files_sha256") or {}
    for split in ("train", "validation", "test"):
        path = output_dir / f"{split}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes.get(split):
            raise ValueError(f"{split}.jsonl hash differs from manifest")
        if sum(1 for line in path.open(encoding="utf-8") if line.strip()) != manifest["split_counts"][split]:
            raise ValueError(f"{split}.jsonl count differs from manifest")
    if (
        manifest.get("quality_report_sha256")
        and hashlib.sha256((output_dir / "quality_report.md").read_bytes()).hexdigest()
        != manifest["quality_report_sha256"]
    ):
        raise ValueError("Quality report hash differs from manifest")
    version = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    prefix = f"{_bucket(project_id)}/versions/{version}"
    for name in FINAL_FILES:
        _put_immutable(output_dir / name, f"{prefix}/{name}")
    if manifest.get("drafts_sha256"):
        staging = f"{_bucket(project_id)}/staging/{manifest['drafts_sha256']}"
        if _gcloud("ls", f"{staging}/", allow_missing=True) is not None:
            _gcloud("rm", "--recursive", staging)
    return prefix
