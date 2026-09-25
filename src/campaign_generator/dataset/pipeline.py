"""Download pinned sources, build one-concept SFT rows, and audit quality.

Run with: python -m campaign_generator.dataset.pipeline --config configs/day1.json
            --output-dir data/processed/day1-run1
The output directory must not exist, preventing accidental replacement of a run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from ..constants import (
    DATA_USER_AGENT,
    DEDUP_PRIORITY,
    INFLUENCER_SUGGESTION,
    NO_PAID_INFLUENCER,
    PROCESS_WORDS,
    SPLITS,
)
from ..prompts.campaign import make_messages
from ..schemas import clean_text, validate_training_record


def canonical_json(value: Any) -> str:
    """Serialize values with stable ordering for hashing and JSONL output."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    """Hash a file in chunks so source downloads need not fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_url(repo_id: str, revision: str, filename: str) -> str:
    """Address a source file at a pinned Hugging Face revision."""
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{quote(filename, safe='/')}?download=true"


def download_pinned(repo_id: str, revision: str, filename: str, raw_dir: Path) -> Path:
    """Reuse a cached source or download it atomically into the raw cache."""
    destination = raw_dir / f"{repo_id.replace('/', '--')}--{revision}" / filename
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(source_url(repo_id, revision, filename), headers={"User-Agent": DATA_USER_AGENT})
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as temp:
        temporary = Path(temp.name)
        try:
            with urlopen(request, timeout=120) as response:
                shutil.copyfileobj(response, temp)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(destination)
    return destination


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield numbered JSON objects and reject non-object source rows."""
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield line_number, value


def _string_list(values: Any) -> list[str]:
    """Normalize a source list while omitting blank or non-string values."""
    if not isinstance(values, list):
        return []
    return [text for item in values if (text := clean_text(item))]


def convert_zarn(
    row: dict[str, Any], *, revision: str, license_name: str, split: str
) -> tuple[dict[str, Any] | None, list[str]]:
    """Map source fields only; never synthesize budget, KPI, or creative claims."""
    creative = row.get("creative_brief") or {}
    brand = row.get("brand_context") or {}
    audience = row.get("audience_profile") or {}
    reference = row.get("reference_brief_package") or {}
    hierarchy = row.get("message_hierarchy") or []
    strategy = row.get("channel_strategy") or []
    reference_output = row.get("reference_output") or {}

    if not all(isinstance(part, dict) for part in (creative, brand, audience, reference, reference_output)):
        return None, ["invalid_nested_source_object"]
    deliverables = reference_output.get("deliverables") or []

    first_message = hierarchy[0] if isinstance(hierarchy, list) and hierarchy else {}
    if not isinstance(first_message, dict):
        first_message = {}
    brief: dict[str, Any] = {
        "industry": clean_text(brand.get("vertical")),
        "target_audience": clean_text(creative.get("audience")) or clean_text(audience.get("primary")),
        "objective": clean_text(creative.get("business_goal")),
        "brand": clean_text(creative.get("brand")),
        "brand_context": clean_text(creative.get("project_background")),
        "channels": _string_list(creative.get("channels")),
        "constraints": list(
            dict.fromkeys(
                _string_list(creative.get("non_goals"))
                + _string_list(brand.get("must_avoid"))
                + _string_list(row.get("constraints"))
            )
        ),
        "proof_point": clean_text(creative.get("must_keep_proof")),
    }
    channel_plan = []
    if isinstance(strategy, list):
        for item in strategy:
            if isinstance(item, dict):
                channel_plan.append(
                    {
                        "channel": clean_text(item.get("channel")),
                        "role": clean_text(item.get("job")),
                        "format_note": clean_text(item.get("format_note")),
                    }
                )
    asset_plan = []
    if isinstance(deliverables, list):
        for item in deliverables:
            if isinstance(item, dict):
                asset_plan.append(
                    {
                        "priority": item.get("priority"),
                        "asset": clean_text(item.get("asset")),
                        "channel": clean_text(item.get("channel")),
                        "reason": clean_text(item.get("reason")),
                    }
                )
    output = {
        "campaign_direction": clean_text(reference.get("hero_angle")),
        "audience_insight": clean_text(audience.get("trust_builder")),
        "key_message": clean_text(first_message.get("message")),
        "channel_plan": channel_plan,
        "asset_plan": asset_plan,
    }
    source_id = clean_text(row.get("example_id"))
    if not source_id:
        return None, ["missing_example_id"]
    record = {
        "id": source_id,
        "source": {
            "dataset": "zarnite/zarn-creative-brief-to-asset-plan",
            "revision": revision,
            "split": split,
            "source_id": source_id,
            "license": license_name,
            "annotation_status": clean_text(row.get("annotation_status")),
        },
        "input": brief,
        "output": output,
        "messages": make_messages(brief, output),
    }
    errors = validate_training_record(record)
    return (None, errors) if errors else (record, [])


def brief_fingerprint(record: dict[str, Any]) -> str:
    """Hash only the normalized input to detect exact brief overlap."""
    return hashlib.sha256(canonical_json(record["input"]).encode("utf-8")).hexdigest()


def deduplicate_splits(
    rows_by_split: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Keep test first, then validation, then train if briefs are identical."""
    seen: set[str] = set()
    kept: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    removed = {split: 0 for split in SPLITS}
    for split in DEDUP_PRIORITY:
        for row in rows_by_split[split]:
            fingerprint = brief_fingerprint(row)
            if fingerprint in seen:
                removed[split] += 1
                continue
            seen.add(fingerprint)
            kept[split].append(row)
    return kept, removed


def audit_brainstorming(path: Path, minimum_unique: int) -> dict[str, Any]:
    """Count distinct marketing cases. Its list-of-ten label is not mapped to Zarn's one-concept schema."""
    total_rows = 0
    marketing_rows = 0
    unique_pairs: set[str] = set()
    unique_prompts: set[str] = set()
    all_pairs: set[str] = set()
    all_prompts: set[str] = set()
    malformed = 0
    obvious_conflicts = 0
    for _, row in read_jsonl(path):
        total_rows += 1
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict) or metadata.get("domain") != "marketing_campaigns":
            continue
        marketing_rows += 1
        conversations = row.get("conversations") or []
        if not isinstance(conversations, list) or len(conversations) != 2:
            malformed += 1
            continue
        prompt = clean_text(conversations[0].get("value")) if isinstance(conversations[0], dict) else ""
        answer = clean_text(conversations[1].get("value")) if isinstance(conversations[1], dict) else ""
        if not prompt or not answer:
            malformed += 1
            continue
        prompt_hash = hashlib.sha256(prompt.casefold().encode()).hexdigest()
        pair_hash = hashlib.sha256((prompt + "\n" + answer).casefold().encode()).hexdigest()
        all_prompts.add(prompt_hash)
        all_pairs.add(pair_hash)
        if NO_PAID_INFLUENCER.search(prompt) and INFLUENCER_SUGGESTION.search(answer):
            obvious_conflicts += 1
            continue
        unique_prompts.add(prompt_hash)
        unique_pairs.add(pair_hash)
    # Input diversity is the deciding factor; repeated labels do not add scenarios.
    usable = len(unique_prompts) >= minimum_unique
    return {
        "total_rows": total_rows,
        "marketing_rows": marketing_rows,
        "unique_marketing_prompts_before_filter": len(all_prompts),
        "unique_marketing_pairs_before_filter": len(all_pairs),
        "unique_marketing_prompts_after_simple_filter": len(unique_prompts),
        "unique_marketing_pairs_after_simple_filter": len(unique_pairs),
        "malformed_marketing_rows": malformed,
        "obvious_constraint_conflicts": obvious_conflicts,
        "minimum_unique_cases": minimum_unique,
        "decision": "candidate_requires_schema_conversion"
        if usable
        else "exclude_from_training_insufficient_diversity",
        "note": "Only an obvious influencer contradiction is checked automatically; this is not a full semantic safety audit.",
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write unchanged normalized dictionaries as deterministic JSONL."""
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")


def build_report(manifest: dict[str, Any], samples: list[dict[str, Any]]) -> str:
    """Summarize dataset quality and five deterministic mapping examples."""
    counts = manifest["zarn"]["kept_counts"]
    audit = manifest["brainstorming"]
    lines = [
        "# Day 1 data quality report",
        "",
        "## Result",
        "",
        f"- Zarn rows kept: train {counts['train']}, validation {counts['validation']}, test {counts['test']}.",
        f"- Exact duplicate briefs removed: {manifest['zarn']['duplicate_briefs_removed']}.",
        f"- Rejected source rows: {manifest['zarn']['rejection_reasons']}.",
        f"- Distinct industries: {manifest['zarn']['distinct_industries']}.",
        f"- Distinct train campaign directions: {manifest['zarn']['distinct_train_directions']}; distinct train key messages: {manifest['zarn']['distinct_train_key_messages']}.",
        f"- Train/test campaign direction overlap: {manifest['zarn']['train_test_direction_overlap_count']} distinct values.",
        f"- Directions containing process/asset words (heuristic): {manifest['zarn']['process_word_direction_count']}/{sum(counts.values())}.",
        f"- Brainstorming marketing rows: {audit['marketing_rows']}; unique prompts before/after simple filter: {audit['unique_marketing_prompts_before_filter']}/{audit['unique_marketing_prompts_after_simple_filter']}; decision: {audit['decision']}.",
        f"- **Fine-tuning readiness: {manifest['trainability']['status']}** — {manifest['trainability']['reason']}.",
        "",
        "## Limitations",
        "",
        "Zarn is authored as a creative-brief-to-asset-plan benchmark. A hero angle may describe an asset workflow rather than an original campaign idea. The process-word count and five samples below make this visible but do not establish creative quality. No numeric budget or KPI labels were added. Exact brief hashes detect literal duplication, not semantic leakage.",
        "",
        "## Five deterministic mapping samples (pipeline sanity check, not gold labels)",
        "",
    ]
    for sample in samples:
        lines.extend(
            [
                f"- `{sample['id']}` ({sample['input']['industry']}): {sample['output']['campaign_direction']}",
                f"  - Key message: {sample['output']['key_message']}",
            ]
        )
    return "\n".join(lines) + "\n"


def run(config_path: Path, raw_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Build a new pinned data run without replacing an existing output."""
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}. Choose a new run directory.")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    zarn = config["sources"]["zarn"]
    brainstorm = config["sources"]["brainstorming"]
    source_files = {
        split: download_pinned(zarn["repo_id"], zarn["revision"], zarn["files"][split], raw_dir)
        for split in SPLITS
    }
    brainstorming_file = download_pinned(
        brainstorm["repo_id"], brainstorm["revision"], brainstorm["file"], raw_dir
    )

    raw_counts: dict[str, int] = {}
    rejection_counts: Counter[str] = Counter()
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        rows: list[dict[str, Any]] = []
        raw_counts[split] = 0
        for _, source_row in read_jsonl(source_files[split]):
            raw_counts[split] += 1
            record, errors = convert_zarn(
                source_row, revision=zarn["revision"], license_name=zarn["license"], split=split
            )
            if record is None:
                rejection_counts.update(errors)
            else:
                rows.append(record)
        rows_by_split[split] = rows
    kept, duplicate_counts = deduplicate_splits(rows_by_split)
    brainstorm_audit = audit_brainstorming(brainstorming_file, brainstorm["min_unique_marketing_cases"])
    industries = sorted({row["input"]["industry"] for rows in kept.values() for row in rows})
    process_count = sum(
        bool(PROCESS_WORDS.search(row["output"]["campaign_direction"]))
        for rows in kept.values()
        for row in rows
    )
    train_directions = {row["output"]["campaign_direction"] for row in kept["train"]}
    test_directions = {row["output"]["campaign_direction"] for row in kept["test"]}
    train_messages = {row["output"]["key_message"] for row in kept["train"]}
    minimum_directions = config["minimum_distinct_train_directions"]
    trainability = {
        "status": "ready_for_review"
        if len(train_directions) >= minimum_directions
        else "blocked_low_target_diversity",
        "minimum_distinct_train_directions": minimum_directions,
        "reason": "campaign directions vary enough for a training review"
        if len(train_directions) >= minimum_directions
        else "the target campaign direction repeats too often; do not claim campaign-ideation SFT readiness",
    }

    manifest: dict[str, Any] = {
        "schema_version": config["schema_version"],
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "seed": config["seed"],
        "sources": {
            alias: {
                "repo_id": source["repo_id"],
                "revision": source["revision"],
                "license": source["license"],
                "files_sha256": {split: sha256_file(path) for split, path in source_files.items()}
                if alias == "zarn"
                else {"jsonl": sha256_file(brainstorming_file)},
            }
            for alias, source in (("zarn", zarn), ("brainstorming", brainstorm))
        },
        "zarn": {
            "raw_counts": raw_counts,
            "kept_counts": {split: len(kept[split]) for split in SPLITS},
            "duplicate_briefs_removed": duplicate_counts,
            "rejection_reasons": dict(sorted(rejection_counts.items())),
            "distinct_industries": len(industries),
            "industries": industries,
            "distinct_train_directions": len(train_directions),
            "distinct_train_key_messages": len(train_messages),
            "train_test_direction_overlap_count": len(train_directions & test_directions),
            "process_word_direction_count": process_count,
        },
        "brainstorming": brainstorm_audit,
        "trainability": trainability,
        "outputs_sha256": {},
    }
    output_dir.mkdir(parents=True)
    try:
        for split in SPLITS:
            file = output_dir / f"{split}.jsonl"
            write_jsonl(file, kept[split])
            manifest["outputs_sha256"][split] = sha256_file(file)
        train_rows = kept["train"]
        samples = random.Random(config["seed"]).sample(train_rows, min(5, len(train_rows)))
        (output_dir / "quality_report.md").write_text(build_report(manifest, samples), encoding="utf-8")
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except BaseException:
        shutil.rmtree(output_dir)
        raise
    return manifest


def main() -> None:
    """Parse the data CLI arguments and print a compact run summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/day1.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.raw_dir, args.output_dir)
    print(
        json.dumps(
            {
                "kept_counts": result["zarn"]["kept_counts"],
                "brainstorming_decision": result["brainstorming"]["decision"],
                "trainability": result["trainability"]["status"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
