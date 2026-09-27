"""Generate, review, and assemble the description-aware 240-row dataset.

Run `generate` to create pending candidates, review every row into a JSONL file,
then run `finalize`. This module never marks teacher output human-approved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import local
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from ..prompts.campaign import SYSTEM_PROMPT_V2, make_messages_v2
from ..schemas import (
    CampaignBrief,
    CampaignDirectionV2,
    TrainingRecordV2,
)
from .gcs import upload_draft, upload_final

MODEL_ID = "gpt-6-luna"
PROMPT_VERSION = "brief-to-campaign-description-v2"
PROMPT_SHA256 = hashlib.sha256(f"{PROMPT_VERSION}\n{SYSTEM_PROMPT_V2}".encode()).hexdigest()
DATASET_NAME = "baseline-v3-description-seed42"
SEED = 42

INDUSTRIES = (
    "food and beverage",
    "beauty",
    "consumer technology",
    "creator education",
    "fintech",
    "health technology",
    "media tools",
    "productivity software",
    "travel",
    "wellness",
)
OBJECTIVES = (
    "increase qualified trial sign-ups",
    "introduce a new product feature",
    "improve repeat visits",
    "increase first purchases",
    "explain a product benefit",
    "grow event registrations",
    "encourage product consideration",
    "increase newsletter subscriptions",
    "support a seasonal launch",
    "drive demo requests",
    "build awareness for a new offer",
    "encourage app activation",
)
AUDIENCES = (
    "busy parents",
    "independent shop owners",
    "early-career professionals",
    "small marketing teams",
    "university students",
    "frequent travelers",
    "first-time buyers",
    "local community organizers",
    "team leads",
    "independent creators",
    "health-conscious adults",
    "time-constrained commuters",
)
CHANNEL_SETS = (
    ("Instagram", "Email"),
    ("TikTok", "Instagram Reels"),
    ("LinkedIn", "Email"),
    ("YouTube Shorts", "Landing page"),
    ("Facebook", "Email"),
    ("Search", "Landing page"),
)
PROOF_POINTS = (
    "The product includes a guided setup checklist.",
    "The service shows the full price before checkout.",
    "The package includes a printed use guide.",
    "Customers can compare the available options on one page.",
    "The app lets users save a draft and return to it later.",
    "The event page lists the schedule and venue details.",
)


def synthetic_briefs() -> list[dict[str, Any]]:
    """Create 120 fictional, grounded briefs with balanced industries and scenarios."""
    rows: list[dict[str, Any]] = []
    for index in range(120):
        industry = INDUSTRIES[index // 12]
        objective = OBJECTIVES[index % len(OBJECTIVES)]
        audience = AUDIENCES[(index * 5 + index // 10) % len(AUDIENCES)]
        channels = CHANNEL_SETS[(index * 7 + index // 12) % len(CHANNEL_SETS)]
        product = f"{industry.title()} offer {index + 1:03d}"
        rows.append(
            {
                "id": f"synthetic-v3-{index + 1:03d}",
                "brief": {
                    "industry": industry,
                    "brand": f"Studio {index + 1:03d}",
                    "brand_context": f"A fictional {industry} brand preparing a focused campaign for a new audience.",
                    "product": product,
                    "target_audience": audience,
                    "objective": objective,
                    "channels": list(channels),
                    "constraints": [
                        "Use only the supplied product information.",
                        "Avoid guaranteed outcomes.",
                    ],
                    "proof_point": PROOF_POINTS[index % len(PROOF_POINTS)],
                    "market": "Thailand",
                    "duration_days": 30,
                },
            }
        )
    return rows


def assign_new_splits(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Assign the 120 synthetic briefs deterministically to 72/12/36 splits."""
    if len(rows) != 120:
        raise ValueError(f"Expected 120 synthetic briefs, found {len(rows)}")
    shuffled = rows.copy()
    random.Random(SEED).shuffle(shuffled)
    return {
        row["id"]: split
        for split, group in (
            ("train", shuffled[:72]),
            ("validation", shuffled[72:84]),
            ("test", shuffled[84:]),
        )
        for row in group
    }


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class Teacher:
    def __init__(self, client: OpenAI, cache_path: Path) -> None:
        self.client = client
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS outputs (cache_key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.db.commit()

    def generate(self, brief: dict[str, Any]) -> tuple[dict[str, Any], str, int, int, bool]:
        key = canonical_hash({"brief": brief, "model": MODEL_ID, "prompt": PROMPT_SHA256})
        cached = self.db.execute("SELECT value FROM outputs WHERE cache_key=?", (key,)).fetchone()
        if cached:
            parsed = json.loads(cached[0])
            return (
                parsed["output"],
                parsed["response_id"],
                parsed["input_tokens"],
                parsed["output_tokens"],
                True,
            )
        response = self.client.responses.parse(
            model=MODEL_ID,
            store=False,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT_V2},
                {
                    "role": "user",
                    "content": "Create one campaign direction for this brief. Return JSON only.\n"
                    + json.dumps(brief, ensure_ascii=False, sort_keys=True),
                },
            ],
            text_format=CampaignDirectionV2,
        )
        if response.output_parsed is None:
            raise ValueError("teacher_returned_no_parsed_output")
        output = CampaignDirectionV2.model_validate(response.output_parsed).model_dump(mode="json")
        info = {
            "output": output,
            "response_id": response.id,
            "input_tokens": response.usage.input_tokens if response.usage else 0,
            "output_tokens": response.usage.output_tokens if response.usage else 0,
        }
        self.db.execute("INSERT OR REPLACE INTO outputs VALUES (?, ?)", (key, json.dumps(info)))
        self.db.commit()
        return output, info["response_id"], info["input_tokens"], info["output_tokens"], False

    def close(self) -> None:
        self.db.close()


def _candidate(
    row_id: str, brief: dict[str, Any], split: str, source: dict[str, str], teacher: Teacher
) -> dict[str, Any]:
    output, response_id, input_tokens, output_tokens, cache_hit = teacher.generate(brief)
    output = CampaignDirectionV2.model_validate(output).model_dump(mode="json")
    return {
        "id": row_id,
        "input": brief,
        "output": output,
        "assigned_split": split,
        "review_status": "pending",
        "source": source,
        "transformation": {
            "method": "openai_responses_structured_output",
            "model": MODEL_ID,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": PROMPT_SHA256,
            "response_id": response_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_hit": cache_hit,
        },
    }


def generate(args: argparse.Namespace) -> None:
    if (args.output_dir / "manifest.json").exists() or (args.output_dir / "drafts.jsonl").exists():
        raise FileExistsError(f"Dataset run already exists: {args.output_dir}")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError("Set OPENAI_API_KEY before generating dataset v3 drafts")
    source_file = args.source_briefs
    old_rows: list[tuple[str, dict[str, Any], str, str]] = []
    for row in read_jsonl(source_file):
        old_rows.append((row["id"], row["brief"], row["split"], row["source_id"]))
    if len(old_rows) != 120 or {
        s: sum(row[2] == s for row in old_rows) for s in ("train", "validation", "test")
    } != {"train": 96, "validation": 12, "test": 12}:
        raise ValueError("Expected the original reviewed seed 42 splits to contain 96/12/12 rows")
    new_rows = synthetic_briefs()
    new_split_by_id = assign_new_splits(new_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "synthetic_briefs.jsonl", new_rows)
    jobs: list[tuple[str, dict[str, Any], str, dict[str, str]]] = []
    for row_id, brief, split, source_id in old_rows:
        source = {
            "dataset": "zarnite/zarn-creative-brief-to-asset-plan",
            "revision": "4e2c4d87937dcda9604e2e99fb98715da51fab89",
            "source_id": source_id,
            "license": "apache-2.0",
        }
        jobs.append((row_id, brief, split, source))
    for row in new_rows:
        source = {
            "dataset": "campaign-generator-synthetic-briefs-v1",
            "revision": canonical_hash(new_rows),
            "source_id": row["id"],
            "license": "internal-synthetic",
        }
        jobs.append((row["id"], row["brief"], new_split_by_id[row["id"]], source))
    thread_state = local()

    def generate_one(job: tuple[str, dict[str, Any], str, dict[str, str]]) -> dict[str, Any]:
        if not hasattr(thread_state, "teacher"):
            thread_state.teacher = Teacher(OpenAI(api_key=api_key), args.cache)
        row_id, brief, split, source = job
        return _candidate(row_id, brief, split, source, thread_state.teacher)

    candidates: list[dict[str, Any] | None] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(generate_one, job): index for index, job in enumerate(jobs)}
        for completed, future in enumerate(as_completed(futures), 1):
            candidates[futures[future]] = future.result()
            if completed % 20 == 0 or completed == len(jobs):
                print(f"Teacher drafts: {completed}/{len(jobs)}", flush=True)
    candidates = [candidate for candidate in candidates if candidate is not None]
    cache_hits = sum(item["transformation"]["cache_hit"] for item in candidates)
    input_tokens = sum(item["transformation"]["input_tokens"] for item in candidates)
    output_tokens = sum(item["transformation"]["output_tokens"] for item in candidates)
    write_jsonl(args.output_dir / "drafts.jsonl", candidates)
    write_jsonl(args.output_dir / "review_queue.jsonl", candidates)
    write_jsonl(
        args.output_dir / "review_template.jsonl",
        [
            {"id": item["id"], "reviewer": "", "status": "", "notes": "", "corrected_output": None}
            for item in candidates
        ],
    )
    (args.output_dir / "HUMAN_REVIEW.md").write_text(
        "# Human review instructions\n\n"
        "Review every row in review_queue.jsonl. Save one decision per candidate in a separate JSONL file using "
        "review_template.jsonl as the shape. Set reviewer and status (`approved` or `rejected`). Check grounding, "
        "constraints, distinctness, and a 2–3 sentence English campaign_description. Put a corrected CampaignDirectionV2 "
        "in corrected_output when editing an approved label. Rejected rows need a replacement candidate object with a "
        "new id, the same assigned_split, complete input/source/transformation data, and a human-approved output.\n",
        encoding="utf-8",
    )
    manifest = {
        "dataset": DATASET_NAME,
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "source_dataset": "zarnite/zarn-creative-brief-to-asset-plan + campaign-generator-synthetic-briefs-v1",
        "source_revision": "4e2c4d87937dcda9604e2e99fb98715da51fab89",
        "model": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "candidate_count": len(candidates),
        "split_counts": {
            split: sum(item["assigned_split"] == split for item in candidates)
            for split in ("train", "validation", "test")
        },
        "human_semantic_review": False,
        "review_status": "pending",
        "cache_hits": cache_hits,
        "api_input_tokens": input_tokens,
        "api_output_tokens": output_tokens,
        "synthetic_briefs_sha256": hashlib.sha256(
            (args.output_dir / "synthetic_briefs.jsonl").read_bytes()
        ).hexdigest(),
        "drafts_sha256": hashlib.sha256((args.output_dir / "drafts.jsonl").read_bytes()).hexdigest(),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.no_upload:
        print(f"Drafts uploaded: {upload_draft(args.output_dir, args.project_id)}")
    print(json.dumps(manifest, indent=2))


def _assert_no_split_leakage(records: list[dict[str, Any]]) -> None:
    seen: dict[str, str] = {}
    for row in records:
        brief = row["input"]
        group = (brief.get("brand", "").casefold(), brief.get("product", "").casefold())
        if group == ("", ""):
            group = ("brief", canonical_hash(brief))
        previous = seen.setdefault("|".join(group), row["assigned_split"])
        if previous != row["assigned_split"]:
            raise ValueError(f"brand/product leakage across splits for {row['id']}")
    brief_hashes: dict[str, str] = {}
    for row in records:
        key = canonical_hash(row["input"])
        split = row["assigned_split"]
        previous = brief_hashes.setdefault(key, split)
        if previous != split:
            raise ValueError(f"duplicate brief leakage across splits for {row['id']}")


def finalize(args: argparse.Namespace) -> None:
    candidates = read_jsonl(args.output_dir / "drafts.jsonl")
    decisions = read_jsonl(args.reviews)
    decision_by_id = {item.get("id"): item for item in decisions}
    if len(decision_by_id) != len(decisions):
        raise ValueError("Review file contains duplicate ids")
    candidate_ids = {item["id"] for item in candidates}
    if set(decision_by_id) != candidate_ids:
        raise ValueError("Review file must contain exactly one human decision for every candidate")
    approved = []
    approved_ids: set[str] = set()
    for row in candidates:
        decision = decision_by_id[row["id"]]
        if not decision.get("reviewer") or decision.get("status") not in ("approved", "rejected"):
            raise ValueError(f"{row['id']}: reviewer and approved/rejected status are required")
        if decision["status"] == "rejected":
            replacement = decision.get("replacement")
            if not isinstance(replacement, dict):
                raise ValueError(f"{row['id']}: rejected rows require a human-reviewed replacement object")
            row = replacement
            if (
                row.get("assigned_split")
                != candidates[[item["id"] for item in candidates].index(decision["id"])]["assigned_split"]
            ):
                raise ValueError("Replacement must retain the rejected row's assigned split")
            row_id = row.get("id")
            if not row_id or row_id in candidate_ids or row_id in approved_ids:
                raise ValueError("Replacement id must be new and unique")
            if not decision.get("replacement_approved"):
                raise ValueError(f"{decision['id']}: replacement_approved must be true after human review")
        row_id = row["id"]
        if row_id in approved_ids:
            raise ValueError(f"Duplicate approved row id: {row_id}")
        approved_ids.add(row_id)
        output = decision.get("corrected_output") or row["output"]
        row["output"] = CampaignDirectionV2.model_validate(output).model_dump(mode="json")
        sentence_count = len(
            [
                part
                for part in re.split(r"(?<=[.!?])\s+", row["output"]["campaign_description"].strip())
                if part
            ]
        )
        if sentence_count < 2 or sentence_count > 3:
            raise ValueError(f"{row_id}: campaign_description must contain 2–3 sentences")
        row["review_status"] = "approved"
        row["reviewer"] = decision["reviewer"]
        row["review_notes"] = decision.get("notes", "")
        record = {
            "id": row["id"],
            "source": {
                "dataset": row["source"]["dataset"],
                "revision": row["source"]["revision"],
                "split": row["assigned_split"],
                "source_id": row["source"]["source_id"],
                "license": row["source"].get("license", ""),
                "annotation_status": "human_semantic_review_approved",
            },
            "input": row["input"],
            "output": row["output"],
            "messages": make_messages_v2(row["input"], row["output"]),
            "transformation": {
                "method": "openai_responses_structured_output",
                "model": row["transformation"]["model"],
                "prompt_version": row["transformation"]["prompt_version"],
                "prompt_sha256": row["transformation"]["prompt_sha256"],
            },
        }
        approved.append(TrainingRecordV2.model_validate(record).model_dump(mode="json"))
    reviewed_candidates = []
    for item in candidates:
        if decision_by_id[item["id"]]["status"] == "rejected":
            replacement = dict(decision_by_id[item["id"]]["replacement"])
            replacement["assigned_split"] = item["assigned_split"]
            reviewed_candidates.append(replacement)
        else:
            reviewed_candidates.append(item)
    _assert_no_split_leakage(reviewed_candidates)
    expected = {"train": 168, "validation": 24, "test": 48}
    for split, count in expected.items():
        rows = [row for row in approved if row["source"]["split"] == split]
        if len(rows) != count:
            raise ValueError(f"Expected {count} approved {split} rows, found {len(rows)}")
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)
    write_jsonl(args.output_dir / "reviewed.jsonl", reviewed_candidates)
    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "review_status": "reviewed",
            "schema_validation": "passed",
            "human_semantic_review": True,
            "reviewed_count": len(approved),
            "split_counts": expected,
            "reviewed_sha256": hashlib.sha256((args.output_dir / "reviewed.jsonl").read_bytes()).hexdigest(),
            "files_sha256": {
                split: hashlib.sha256((args.output_dir / f"{split}.jsonl").read_bytes()).hexdigest()
                for split in expected
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = (
        f"# {DATASET_NAME}\n\n- Rows: 240 (train 168, validation 24, test 48)\n"
        "- Human semantic review: 240/240 approved; see reviewed.jsonl for reviewer records.\n"
        "- Primary promotion holdout: the 36 synthetic briefs in test; legacy test rows are secondary.\n"
        "- Teacher labels are proposals and do not contain KPI or budget labels.\n"
    )
    (args.output_dir / "quality_report.md").write_text(report, encoding="utf-8")
    manifest["quality_report_sha256"] = hashlib.sha256(
        (args.output_dir / "quality_report.md").read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.no_upload:
        print(f"Dataset uploaded: {upload_final(args.output_dir, args.project_id)}")
    print(json.dumps(manifest, indent=2))


def prepare_schema_only_training(args: argparse.Namespace) -> None:
    """Build explicitly unreviewed training splits after strict schema validation."""
    candidates = read_jsonl(args.output_dir / "drafts.jsonl")
    expected = {"train": 168, "validation": 24, "test": 48}
    if len(candidates) != sum(expected.values()):
        raise ValueError(f"Expected 240 candidates, found {len(candidates)}")
    records: dict[str, list[dict[str, Any]]] = {split: [] for split in expected}
    ids: set[str] = set()
    for candidate in candidates:
        row_id = candidate.get("id")
        split = candidate.get("assigned_split")
        if not row_id or row_id in ids:
            raise ValueError(f"Missing or duplicate candidate id: {row_id}")
        ids.add(row_id)
        if split not in records:
            raise ValueError(f"{row_id}: invalid assigned_split {split!r}")
        brief = CampaignBrief.model_validate(candidate["input"]).model_dump(mode="json")
        output = CampaignDirectionV2.model_validate(candidate["output"]).model_dump(mode="json")
        source = candidate["source"]
        transformation = candidate["transformation"]
        record = {
            "id": row_id,
            "source": {
                "dataset": source["dataset"],
                "revision": source["revision"],
                "split": split,
                "source_id": source["source_id"],
                "license": source.get("license", ""),
                "annotation_status": "teacher_schema_validated_unreviewed",
            },
            "input": brief,
            "output": output,
            "messages": make_messages_v2(brief, output),
            "transformation": {
                "method": transformation["method"],
                "model": transformation["model"],
                "prompt_version": transformation["prompt_version"],
                "prompt_sha256": transformation["prompt_sha256"],
            },
        }
        records[split].append(TrainingRecordV2.model_validate(record).model_dump(mode="json"))
    if {split: len(rows) for split, rows in records.items()} != expected:
        raise ValueError(
            f"Split counts do not match v3 contract: { {k: len(v) for k, v in records.items()} }"
        )
    for split, rows in records.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)
    hashes = {
        split: hashlib.sha256((args.output_dir / f"{split}.jsonl").read_bytes()).hexdigest()
        for split in expected
    }
    quality_report = (
        f"# {DATASET_NAME} schema-only experiment\n\n"
        "- Candidate rows: 240; all outputs validated against CampaignDirectionV2 and all training rows against TrainingRecordV2.\n"
        "- Split counts: train 168, validation 24, test 48.\n"
        "- Schema validation: passed.\n"
        "- Human semantic review: pending; no semantic quality or factual grounding claim is made.\n"
        "- Use: experimental LoRA training only; not eligible for model promotion.\n"
    )
    report_path = args.output_dir / "quality_report.md"
    report_path.write_text(quality_report, encoding="utf-8")
    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "training_eligibility": "schema_only_experiment",
            "schema_validation": "passed",
            "human_semantic_review": False,
            "review_status": "pending",
            "split_counts": expected,
            "training_files_sha256": hashes,
            "quality_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.no_upload:
        print(f"Dataset uploaded: {upload_final(args.output_dir, args.project_id)}")
    print(
        json.dumps(
            {
                "schema_validation": "passed",
                "human_semantic_review": False,
                "split_counts": expected,
                "training_files_sha256": hashes,
            },
            indent=2,
        )
    )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--source-briefs", type=Path, default=Path("dataset/source_briefs.jsonl"))
    generate_parser.add_argument("--output-dir", type=Path, default=Path("data/processed/dataset-v3"))
    generate_parser.add_argument(
        "--cache", type=Path, default=Path("data/cache/baseline-v3-description.sqlite3")
    )
    prepare_parser = commands.add_parser("prepare-briefs")
    prepare_parser.add_argument("--output-dir", type=Path, default=Path("data/processed/dataset-v3"))
    finalize_parser = commands.add_parser("finalize")
    finalize_parser.add_argument("--output-dir", type=Path, default=Path("data/processed/dataset-v3"))
    finalize_parser.add_argument("--reviews", type=Path, required=True)
    schema_train_parser = commands.add_parser("prepare-schema-only-training")
    schema_train_parser.add_argument("--output-dir", type=Path, default=Path("data/processed/dataset-v3"))
    sync_parser = commands.add_parser("sync", help="Retry GCS upload after a network failure")
    sync_parser.add_argument("--output-dir", type=Path, default=Path("data/processed/dataset-v3"))
    for command in (generate_parser, finalize_parser, schema_train_parser, sync_parser):
        command.add_argument(
            "--project-id", default=os.getenv("CAMPAIGN_GCP_PROJECT", "campaign-generator-509812")
        )
        command.add_argument("--no-upload", action="store_true", help="Keep generated files local")
    args = parser.parse_args()
    if args.command == "generate":
        generate(args)
    elif args.command == "prepare-briefs":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        rows = synthetic_briefs()
        write_jsonl(args.output_dir / "synthetic_briefs.jsonl", rows)
        split_map = assign_new_splits(rows)
        split_manifest = {
            "dataset": DATASET_NAME,
            "seed": SEED,
            "counts": {
                split: sum(value == split for value in split_map.values())
                for split in ("train", "validation", "test")
            },
            "synthetic_briefs_sha256": hashlib.sha256(
                (args.output_dir / "synthetic_briefs.jsonl").read_bytes()
            ).hexdigest(),
            "human_semantic_review": False,
            "status": "briefs_only",
        }
        (args.output_dir / "brief_manifest.json").write_text(
            json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(split_manifest, indent=2))
    elif args.command == "prepare-schema-only-training":
        prepare_schema_only_training(args)
    elif args.command == "sync":
        manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_validation") == "passed":
            print(f"Dataset uploaded: {upload_final(args.output_dir, args.project_id)}")
        else:
            print(f"Drafts uploaded: {upload_draft(args.output_dir, args.project_id)}")
    else:
        finalize(args)


if __name__ == "__main__":
    main()
