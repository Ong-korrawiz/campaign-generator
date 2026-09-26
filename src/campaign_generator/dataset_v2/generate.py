"""Generate reviewable campaign directions from first-party brief JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .contracts import (
    BriefReader,
    CampaignTeacher,
    DraftProvenance,
    DraftRecord,
    DraftWriter,
    TeacherCache,
    TeacherResult,
)
from .io import JsonlBriefReader, JsonlDraftWriter, canonical_json, sha256_file
from .teacher import OpenAICampaignTeacher


class SqliteTeacherCache:
    """Persist each successful response so a failed run can resume without paying twice."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS teacher_results "
            "(cache_key TEXT PRIMARY KEY, result_json TEXT NOT NULL)"
        )
        self._connection.commit()

    def get(self, key: str) -> TeacherResult | None:
        row = self._connection.execute(
            "SELECT result_json FROM teacher_results WHERE cache_key = ?", (key,)
        ).fetchone()
        return TeacherResult.model_validate_json(row[0]) if row else None

    def put(self, key: str, result: TeacherResult) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO teacher_results VALUES (?, ?)",
            (key, result.model_dump_json()),
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()


class DatasetGenerator:
    """Coordinate independent input, teacher, cache, and output components."""

    def __init__(
        self,
        reader: BriefReader,
        teacher: CampaignTeacher,
        cache: TeacherCache,
        writer: DraftWriter,
    ) -> None:
        self._reader = reader
        self._teacher = teacher
        self._cache = cache
        self._writer = writer

    def run(self, input_path: Path, output_dir: Path, limit: int | None = None) -> dict[str, Any]:
        if output_dir.exists():
            raise FileExistsError(output_dir)
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        seeds = self._reader.read(input_path)
        drafts: list[DraftRecord] = []
        cache_hits = 0
        api_input_tokens = 0
        api_output_tokens = 0
        for line_number, seed in seeds[:limit]:
            cache_payload = {
                "brief": seed.brief.model_dump(mode="json"),
                "model": self._teacher.model_id,
                "prompt_sha256": self._teacher.prompt_sha256,
            }
            key = hashlib.sha256(canonical_json(cache_payload).encode("utf-8")).hexdigest()
            result = self._cache.get(key)
            hit = result is not None
            if result is None:
                result = self._teacher.generate(seed.brief)
                self._cache.put(key, result)
                api_input_tokens += result.input_tokens
                api_output_tokens += result.output_tokens
            else:
                cache_hits += 1
            drafts.append(
                DraftRecord(
                    id=seed.id,
                    input=seed.brief,
                    output=result.output,
                    provenance=DraftProvenance(
                        model=self._teacher.model_id,
                        prompt_version=self._teacher.prompt_version,
                        prompt_sha256=self._teacher.prompt_sha256,
                        source_line=line_number,
                        response_id=result.response_id,
                        generated_at=result.generated_at,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        cache_hit=hit,
                    ),
                )
            )
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_file": str(input_path),
            "input_sha256": sha256_file(input_path),
            "model": self._teacher.model_id,
            "prompt_version": self._teacher.prompt_version,
            "prompt_sha256": self._teacher.prompt_sha256,
            "draft_count": len(drafts),
            "cache_hits": cache_hits,
            "api_input_tokens": api_input_tokens,
            "api_output_tokens": api_output_tokens,
            "review_status": "pending",
        }
        return self._writer.write(output_dir, drafts, manifest)


def main() -> None:
    """Run GPT-6 Luna on validated JSONL briefs; no web search is enabled."""
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/cache/brief-teacher.sqlite3"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your_openai_api_key_here":
        parser.error("Set OPENAI_API_KEY before generating drafts")
    cache = SqliteTeacherCache(args.cache)
    try:
        generator = DatasetGenerator(
            reader=JsonlBriefReader(),
            teacher=OpenAICampaignTeacher(OpenAI(api_key=api_key)),
            cache=cache,
            writer=JsonlDraftWriter(),
        )
        result = generator.run(args.input_file, args.output_dir, args.limit)
    finally:
        cache.close()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
