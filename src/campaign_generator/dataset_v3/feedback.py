"""Freeze public thumb-up feedback into an immutable training dataset version."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any

from ..prompts.campaign import make_messages_v2
from ..schemas import CampaignBrief, CampaignDirectionV2, TrainingRecordV2


def _encode(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_encode(value)).hexdigest()


def _rows(content: bytes) -> list[dict[str, Any]]:
    return [
        TrainingRecordV2.model_validate_json(line).model_dump(mode="json")
        for line in content.splitlines()
        if line
    ]


def _brief_group(brief: dict[str, Any]) -> str:
    brand = brief.get("brand", "").casefold()
    product = (brief.get("product") or "").casefold()
    return f"{brand}|{product}" if brand or product else _digest(brief)


def build_version(
    base_uri: str,
    base_files: dict[str, bytes],
    votes: list[tuple[str, int, dict[str, Any], dict[str, Any]]],
) -> dict[str, bytes] | None:
    """Return complete version files, or None when no new up-voted row exists."""
    base_manifest = json.loads(base_files["manifest.json"])
    splits = {split: _rows(base_files[f"{split}.jsonl"]) for split in ("train", "validation", "test")}
    base_train_count = len(splits["train"])
    for split, rows in splits.items():
        content = base_files[f"{split}.jsonl"]
        expected_hash = (
            base_manifest.get("training_files_sha256") or base_manifest.get("files_sha256") or {}
        ).get(split)
        if hashlib.sha256(content).hexdigest() != expected_hash:
            raise ValueError(f"Base {split}.jsonl hash differs from manifest")
        if len(rows) != base_manifest["split_counts"][split]:
            raise ValueError(f"Base {split}.jsonl count differs from manifest")

    holdout_briefs = {_digest(row["input"]) for split in ("validation", "test") for row in splits[split]}
    holdout_groups = {_brief_group(row["input"]) for split in ("validation", "test") for row in splits[split]}
    seen = {
        _digest({"input": row["input"], "output": row["output"]}) for rows in splits.values() for row in rows
    }
    added = 0
    for generation_id, _, feedback, generation in sorted(votes):
        brief = CampaignBrief.model_validate(
            {key: value for key, value in generation["brief"].items() if key in CampaignBrief.model_fields}
        ).model_dump(mode="json")
        if _digest(brief) in holdout_briefs or _brief_group(brief) in holdout_groups:
            continue
        for index_text, rating in sorted(feedback["ratings"].items()):
            if rating != "up":
                continue
            index = int(index_text)
            if not 0 <= index < 3:
                raise ValueError("Invalid concept index in feedback")
            source_concept = generation["concepts"][index]
            output = CampaignDirectionV2.model_validate(
                {
                    key: value
                    for key, value in source_concept.items()
                    if key in CampaignDirectionV2.model_fields
                }
            ).model_dump(mode="json")
            key = _digest({"input": brief, "output": output})
            if key in seen:
                continue
            seen.add(key)
            record = TrainingRecordV2.model_validate(
                {
                    "id": f"feedback-{key[:32]}",
                    "source": {
                        "dataset": "campaign-public-feedback",
                        "revision": generation_id,
                        "split": "train",
                        "source_id": f"{generation_id}:{index}",
                        "annotation_status": "public_thumb_up_schema_validated_unreviewed",
                    },
                    "input": brief,
                    "output": output,
                    "messages": make_messages_v2(brief, output),
                }
            ).model_dump(mode="json")
            splits["train"].append(record)
            added += 1
    if not added:
        return None

    result = {
        "train.jsonl": base_files["train.jsonl"]
        + (b"" if base_files["train.jsonl"].endswith(b"\n") else b"\n")
        + b"".join(_encode(row) + b"\n" for row in splits["train"][base_train_count:]),
        "validation.jsonl": base_files["validation.jsonl"],
        "test.jsonl": base_files["test.jsonl"],
    }
    report = (
        f"# Feedback training snapshot\n\nBase dataset: {base_uri}\n\n"
        f"Added public thumb-up concepts: {added}. Schema validated; human semantic review not performed.\n"
    )
    result["quality_report.md"] = report.encode("utf-8")
    manifest = {
        "dataset": "campaign-feedback-training",
        "schema_version": 2,
        "base_dataset_uri": base_uri,
        "split_counts": {split: len(rows) for split, rows in splits.items()},
        "feedback_added_count": added,
        "feedback_snapshot": [
            {"generation_id": generation_id, "object_generation": object_generation}
            for generation_id, object_generation, _, _ in sorted(votes)
        ],
        "training_files_sha256": {
            split: hashlib.sha256(result[f"{split}.jsonl"]).hexdigest() for split in splits
        },
        "quality_report_sha256": hashlib.sha256(result["quality_report.md"]).hexdigest(),
        "schema_validation": "passed",
        "human_semantic_review": False,
        "review_status": "pending",
        "training_eligibility": "schema_only_experiment",
    }
    result["manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode() + b"\n"
    )
    return result


def prepare_feedback_version(base_uri: str, feedback_bucket: str, project_id: str | None = None) -> str:
    from google.cloud import storage
    from google.oauth2.credentials import Credentials

    if not base_uri.startswith("gs://") or "/" not in base_uri[5:]:
        raise ValueError("Base dataset must be a gs:// prefix")
    base_bucket_name, base_prefix = base_uri[5:].split("/", 1)
    access_token = os.getenv("GOOGLE_OAUTH_ACCESS_TOKEN")
    credentials = Credentials(token=access_token) if access_token else None
    client = storage.Client(project=project_id, credentials=credentials)
    base_bucket = client.bucket(base_bucket_name)
    base_files = {
        name: base_bucket.blob(f"{base_prefix.rstrip('/')}/{name}").download_as_bytes()
        for name in ("manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl")
    }
    bucket = client.bucket(feedback_bucket)
    votes = []
    for listed in bucket.list_blobs(prefix="feedback/votes/"):
        generation_id = listed.name.removeprefix("feedback/votes/").removesuffix(".json")
        object_generation = int(listed.generation)
        feedback = json.loads(bucket.blob(listed.name, generation=object_generation).download_as_bytes())
        if feedback.get("generation_id") != generation_id:
            raise ValueError("Feedback generation ID mismatch")
        generation = json.loads(bucket.blob(f"feedback/generations/{generation_id}.json").download_as_bytes())
        votes.append((generation_id, object_generation, feedback, generation))
    result = build_version(base_uri, base_files, votes)
    if result is None:
        return base_uri
    version = hashlib.sha256(result["manifest.json"]).hexdigest()
    prefix = f"versions/{version}"
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "quality_report.md", "manifest.json"):
        blob = base_bucket.blob(f"{prefix}/{name}")
        try:
            blob.upload_from_string(result[name], if_generation_match=0)
        except Exception as exc:
            from google.api_core.exceptions import PreconditionFailed

            if not isinstance(exc, PreconditionFailed) or blob.download_as_bytes() != result[name]:
                raise
    return f"gs://{base_bucket_name}/{prefix}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-uri", required=True)
    parser.add_argument("--feedback-bucket", required=True)
    parser.add_argument("--project-id", required=True)
    args = parser.parse_args()
    print(prepare_feedback_version(args.base_uri, args.feedback_bucket, args.project_id))


if __name__ == "__main__":
    main()
