"""Run a reproducible public API check on diverse held-out campaign briefs."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import httpx

from ..api.service import _similar_name
from ..config import (
    API_DESCRIPTION_MAX_SENTENCES,
    API_DESCRIPTION_MIN_SENTENCES,
    EVAL_LIVE_BUDGET_CURRENCY,
    EVAL_LIVE_BUDGET_MAX,
    EVAL_LIVE_BUDGET_MIN,
    EVAL_LIVE_LIMIT,
    EVAL_LIVE_TIMEOUT_SECONDS,
)
from ..io import read_jsonl
from ..schemas import CampaignGenerationResponse


def select_rows(path: Path, limit: int) -> list[dict]:
    by_industry: dict[str, list[dict]] = defaultdict(list)
    for _, row in read_jsonl(path):
        if str(row.get("id", "")).startswith("synthetic-v3-"):
            by_industry[row["input"]["industry"]].append(row)
    selected: list[dict] = []
    while len(selected) < limit and any(by_industry.values()):
        for industry in sorted(by_industry):
            if by_industry[industry] and len(selected) < limit:
                selected.append(by_industry[industry].pop(0))
    if len(selected) != limit:
        raise ValueError(f"Requested {limit} synthetic holdout briefs, found {len(selected)}")
    return selected


def check_response(data: dict, brief: dict) -> list[str]:
    response = CampaignGenerationResponse.model_validate(data)
    errors: list[str] = []
    prior_names: list[str] = []
    for concept in response.concepts:
        if _similar_name(concept.campaign_direction, prior_names):
            errors.append("similar_campaign_direction")
        prior_names.append(concept.campaign_direction.casefold().strip())
        sentences = [
            part
            for part in re.split(r"[.!?]+(?:\s+|$)", concept.campaign_description.strip())
            if part.strip()
        ]
        if not API_DESCRIPTION_MIN_SENTENCES <= len(sentences) <= API_DESCRIPTION_MAX_SENTENCES:
            errors.append("description_sentence_count")
        channels = {item.channel for item in concept.channel_plan} | {
            item.channel for item in concept.asset_plan
        }
        if not channels.issubset(set(brief["channels"])):
            errors.append("unapproved_channel")
        allocation = sum(item.amount for item in concept.budget_allocation)
        if "budget_min" in brief:
            if not brief["budget_min"] <= allocation <= brief["budget_max"]:
                errors.append("budget_outside_range")
        elif concept.budget_allocation:
            errors.append("unexpected_budget_allocation")
    return sorted(set(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--url", required=True, help="Public API base URL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=EVAL_LIVE_LIMIT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Pass --execute to call the public API")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if args.limit < 1:
        parser.error("--limit must be positive")
    results = []
    with httpx.Client(timeout=httpx.Timeout(EVAL_LIVE_TIMEOUT_SECONDS)) as client:
        for index, row in enumerate(select_rows(args.test_file, args.limit)):
            brief = dict(row["input"])
            if index % 2:
                brief.update(
                    budget_min=EVAL_LIVE_BUDGET_MIN,
                    budget_max=EVAL_LIVE_BUDGET_MAX,
                    currency=EVAL_LIVE_BUDGET_CURRENCY,
                )
            started = time.perf_counter()
            try:
                response = client.post(args.url.rstrip("/") + "/generate", json=brief)
                payload = response.json()
                errors = (
                    check_response(payload, brief)
                    if response.status_code == 200
                    else [f"http_{response.status_code}"]
                )
                valid = response.status_code == 200 and not errors
                result = {
                    "source_id": row["id"],
                    "industry": brief["industry"],
                    "budget_supplied": bool(index % 2),
                    "status": response.status_code,
                    "valid": valid,
                    "errors": errors,
                    "latency_seconds": round(time.perf_counter() - started, 3),
                    "output": payload if valid else None,
                }
            except (httpx.HTTPError, ValueError) as exc:
                result = {
                    "source_id": row["id"],
                    "industry": brief["industry"],
                    "budget_supplied": bool(index % 2),
                    "status": 0,
                    "valid": False,
                    "errors": [type(exc).__name__],
                    "latency_seconds": round(time.perf_counter() - started, 3),
                    "output": None,
                }
            results.append(result)
            print(
                json.dumps(
                    {
                        "completed": len(results),
                        "valid": sum(r["valid"] for r in results),
                        "last_errors": result["errors"],
                    }
                ),
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "total": len(results),
                "valid": sum(r["valid"] for r in results),
                "industries": len({r["industry"] for r in results}),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
