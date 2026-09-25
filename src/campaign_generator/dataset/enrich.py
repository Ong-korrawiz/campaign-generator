"""Create resumable, OpenAI-generated brainstorming targets from normalized Zarn rows.

Run with: python -m campaign_generator.dataset.enrich --input-dir data/processed/day1-refactor
          --output-dir data/processed/zarn-brainstorm-v1
"""

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

from ..constants import ENRICHMENT_PROMPT_VERSION, SPLITS
from ..prompts.enrichment import (
    ENRICHMENT_SYSTEM_PROMPT,
    format_brainstorming_response,
    render_enrichment_prompt,
)
from ..schemas import BrainstormingSet, BrainstormingTrainingRecord, TrainingRecord
from .pipeline import canonical_json, read_jsonl, sha256_file, write_jsonl


def prompt_fingerprint() -> str:
    """Hash the exact system instruction and prompt version for provenance."""
    content = f"{ENRICHMENT_PROMPT_VERSION}\n{ENRICHMENT_SYSTEM_PROMPT}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def cache_key(record: dict[str, Any], model: str, prompt_hash: str) -> str:
    """Key cached targets by source row, selected model, and prompt revision."""
    payload = {"record": record, "model": model, "prompt_sha256": prompt_hash}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def initialize_cache(connection: sqlite3.Connection) -> None:
    """Create the local cache table if it does not yet exist."""
    connection.execute(
        "CREATE TABLE IF NOT EXISTS generations ("
        "cache_key TEXT PRIMARY KEY, output_json TEXT NOT NULL, usage_json TEXT NOT NULL, "
        "created_at TEXT NOT NULL)"
    )
    connection.commit()


def read_cached(connection: sqlite3.Connection, key: str) -> tuple[dict[str, Any], dict[str, int]] | None:
    """Return a cached target and token usage, if present."""
    row = connection.execute(
        "SELECT output_json, usage_json FROM generations WHERE cache_key = ?", (key,)
    ).fetchone()
    if row is None:
        return None
    return json.loads(row[0]), json.loads(row[1])


def save_cached(
    connection: sqlite3.Connection,
    key: str,
    output: dict[str, Any],
    usage: dict[str, int],
) -> None:
    """Persist a successful generation immediately for restart-safe reuse."""
    connection.execute(
        "INSERT OR REPLACE INTO generations VALUES (?, ?, ?, ?)",
        (
            key,
            canonical_json(output),
            canonical_json(usage),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    connection.commit()


def generate_target(
    client: OpenAI,
    record: dict[str, Any],
    *,
    model: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Ask the Responses API for one schema-constrained concept and token usage."""
    response = client.responses.parse(
        model=model,
        store=False,
        input=[
            {"role": "system", "content": ENRICHMENT_SYSTEM_PROMPT},
            {"role": "user", "content": render_enrichment_prompt(record)},
        ],
        text_format=BrainstormingSet,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise ValueError(f"OpenAI returned no parsed target for row {record['id']}")
    output = parsed.model_dump(mode="json")
    usage = {
        "input_tokens": int(response.usage.input_tokens) if response.usage else 0,
        "output_tokens": int(response.usage.output_tokens) if response.usage else 0,
    }
    return output, usage


def enrich_record(
    record: dict[str, Any],
    output: dict[str, Any],
    *,
    model: str,
    prompt_hash: str,
) -> dict[str, Any]:
    """Replace the source-style target with the generated concept and trace it."""
    enriched = dict(record)
    enriched["output"] = BrainstormingSet.model_validate(output).model_dump(mode="json")
    enriched["messages"] = [
        {"role": "user", "content": render_enrichment_prompt(enriched)},
        {
            "role": "assistant",
            "content": format_brainstorming_response(enriched["output"]),
        },
    ]
    enriched["transformation"] = {
        "method": "openai_responses_structured_output",
        "model": model,
        "prompt_version": ENRICHMENT_PROMPT_VERSION,
        "prompt_sha256": prompt_hash,
    }
    return BrainstormingTrainingRecord.model_validate(enriched).model_dump(mode="json")


def run_enrichment(
    input_dir: Path,
    output_dir: Path,
    cache_path: Path,
    *,
    model: str,
    limit_per_split: int | None = None,
    selected_splits: tuple[str, ...] = SPLITS,
) -> dict[str, Any]:
    """Enrich each split, reusing cache entries and writing a new immutable run."""
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}. Choose a new run directory.")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your_openai_api_key_here":
        raise RuntimeError("Set OPENAI_API_KEY in .env before running enrichment.")

    client = OpenAI(api_key=api_key)
    prompt_hash = prompt_fingerprint()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(cache_path)
    initialize_cache(connection)
    results: dict[str, list[dict[str, Any]]] = {}
    split_usage: dict[str, dict[str, int]] = {}
    cache_hits: dict[str, int] = {}
    input_hashes: dict[str, str] = {}

    try:
        for split in selected_splits:
            input_path = input_dir / f"{split}.jsonl"
            if not input_path.is_file():
                raise FileNotFoundError(f"Missing input split: {input_path}")
            input_hashes[split] = sha256_file(input_path)
            records: list[dict[str, Any]] = []
            usage_total = {
                "api_input_tokens": 0,
                "api_output_tokens": 0,
                "cached_input_tokens": 0,
                "cached_output_tokens": 0,
            }
            cache_hits[split] = 0
            for _, record in read_jsonl(input_path):
                if limit_per_split is not None and len(records) >= limit_per_split:
                    break
                TrainingRecord.model_validate(record)
                key = cache_key(record, model, prompt_hash)
                cached = read_cached(connection, key)
                if cached is None:
                    output, usage = generate_target(client, record, model=model)
                    BrainstormingSet.model_validate(output)
                    save_cached(connection, key, output, usage)
                    usage_prefix = "api"
                else:
                    output, usage = cached
                    cache_hits[split] += 1
                    usage_prefix = "cached"
                records.append(enrich_record(record, output, model=model, prompt_hash=prompt_hash))
                for token_name in ("input_tokens", "output_tokens"):
                    usage_total[f"{usage_prefix}_{token_name}"] += usage.get(token_name, 0)
            results[split] = records
            split_usage[split] = usage_total
    finally:
        connection.close()

    output_dir.mkdir(parents=True, exist_ok=False)
    output_hashes: dict[str, str] = {}
    for split in selected_splits:
        output_path = output_dir / f"{split}.jsonl"
        write_jsonl(output_path, results[split])
        output_hashes[split] = sha256_file(output_path)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_input_dir": str(input_dir),
        "input_sha256": input_hashes,
        "model": model,
        "selected_splits": list(selected_splits),
        "prompt_version": ENRICHMENT_PROMPT_VERSION,
        "prompt_sha256": prompt_hash,
        "split_counts": {split: len(results[split]) for split in selected_splits},
        "cache_hits": cache_hits,
        "token_usage": split_usage,
        "outputs_sha256": output_hashes,
        "label_notice": "Targets are OpenAI-generated synthetic transformations of Zarn references; they are not human-verified labels.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    """Load .env, parse CLI options, and run the enrichment pipeline."""
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/cache/openai-enrichment.sqlite3"))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--split", choices=SPLITS, help="Process only this dataset split.")
    parser.add_argument("--limit-per-split", type=int)
    args = parser.parse_args()
    if args.limit_per_split is not None and args.limit_per_split < 1:
        parser.error("--limit-per-split must be at least 1")
    result = run_enrichment(
        args.input_dir,
        args.output_dir,
        args.cache,
        model=args.model,
        limit_per_split=args.limit_per_split,
        selected_splits=(args.split,) if args.split else SPLITS,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
