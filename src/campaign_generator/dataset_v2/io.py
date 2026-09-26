"""JSONL input validation and immutable draft output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import BriefSeed, DraftRecord


def canonical_json(value: Any) -> str:
    """Serialize model data deterministically for hashing and JSONL."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    """Hash a file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class JsonlBriefReader:
    """Reject malformed or repeated briefs before the first API call."""

    def read(self, path: Path) -> list[tuple[int, BriefSeed]]:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows: list[tuple[int, BriefSeed]] = []
        seen_ids: set[str] = set()
        seen_briefs: set[str] = set()
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    seed = BriefSeed.model_validate(value)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{path}:{line_number}: invalid brief: {exc}") from exc
                brief_hash = hashlib.sha256(
                    canonical_json(seed.brief.model_dump(mode="json")).encode("utf-8")
                ).hexdigest()
                if seed.id in seen_ids:
                    raise ValueError(f"{path}:{line_number}: duplicate id: {seed.id}")
                if brief_hash in seen_briefs:
                    raise ValueError(f"{path}:{line_number}: duplicate brief")
                seen_ids.add(seed.id)
                seen_briefs.add(brief_hash)
                rows.append((line_number, seed))
        if not rows:
            raise ValueError(f"{path}: no briefs found")
        return rows


class JsonlDraftWriter:
    """Write only to a new run directory so earlier datasets are preserved."""

    def write(self, path: Path, rows: list[DraftRecord], manifest: dict[str, Any]) -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir(exist_ok=False)
        drafts_path = path / "drafts.jsonl"
        with drafts_path.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(canonical_json(row.model_dump(mode="json")) + "\n")
        completed = {**manifest, "drafts_sha256": sha256_file(drafts_path)}
        (path / "manifest.json").write_text(
            json.dumps(completed, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return completed
