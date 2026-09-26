"""Compare base and fine-tuned campaign brainstorms with an OpenAI judge.

Prediction JSONL rows contain source_id (or id) and one of output, response,
or raw_response. Test rows contain id and a structured input campaign brief.
Run with --execute only after code review because the command makes paid calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict

from ..dataset.pipeline import read_jsonl, sha256_file
from ..prompts.judge import JUDGE_SYSTEM_PROMPT, JUDGE_USER_PROMPT_TEMPLATE

Vote = Literal["A", "B", "tie"]
CRITERIA = ("goal_alignment", "groundedness", "idea_distinctness", "execution_fit", "overall")


class StrictJudgeModel(BaseModel):
    """Reject extra fields in machine-readable judge results."""

    model_config = ConfigDict(extra="forbid")


class CriterionJudgment(StrictJudgeModel):
    """Record a preference and concise evidence for one evaluation criterion."""

    winner: Vote
    evidence: str
    reasoning: str


class PairwiseJudgment(StrictJudgeModel):
    """Structured result for one ordering of a pair of candidate responses."""

    goal_alignment: CriterionJudgment
    groundedness: CriterionJudgment
    idea_distinctness: CriterionJudgment
    execution_fit: CriterionJudgment
    overall: CriterionJudgment


class Prediction(StrictJudgeModel):
    """One generated answer linked to its held-out brief."""

    source_id: str
    content: Any
    latency_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


def load_jsonl_map(path: Path, *, id_fields: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Read JSONL objects indexed by the first available stable identifier."""
    records: dict[str, dict[str, Any]] = {}
    for line_number, row in read_jsonl(path):
        record_id = next((str(row[field]) for field in id_fields if row.get(field)), None)
        if record_id is None:
            raise ValueError(f"{path}:{line_number} has no identifier in {id_fields}.")
        if record_id in records:
            raise ValueError(f"{path}:{line_number} repeats identifier {record_id!r}.")
        records[record_id] = row
    return records


def prediction_from_row(row: dict[str, Any], *, source_id: str) -> Prediction:
    """Normalize supported inference-result fields into one prediction."""
    content = next((row[key] for key in ("output", "response", "raw_response") if key in row), None)
    if content is None:
        raise ValueError(f"Prediction for {source_id!r} needs output, response, or raw_response.")
    return Prediction(
        source_id=source_id,
        content=content,
        latency_seconds=row.get("latency_seconds"),
        input_tokens=row.get("input_tokens"),
        output_tokens=row.get("output_tokens"),
    )


def render_content(content: Any) -> str:
    """Render structured or textual predictions consistently for the judge."""
    if isinstance(content, str):
        return content.strip()
    return json.dumps(content, ensure_ascii=False, sort_keys=True, indent=2)


def automatic_checks(content: Any, *, expected_ideas: int) -> dict[str, Any]:
    """Measure idea count, unique names, and prioritization notes deterministically."""
    parsed = content
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get("ideas"), list):
        ideas = parsed["ideas"]
        names = [str(item.get("name", "")).strip().casefold() for item in ideas if isinstance(item, dict)]
        return {
            "format": "structured_ideas",
            "idea_count": len(ideas),
            "expected_idea_count": expected_ideas,
            "idea_count_valid": len(ideas) == expected_ideas,
            "unique_nonempty_names": bool(names)
            and len(names) == len(ideas)
            and all(names)
            and len(set(names)) == len(names),
            "prioritization_notes_present": bool(str(parsed.get("prioritization_notes", "")).strip()),
        }

    text = render_content(content)
    numbered = re.findall(r"(?m)^\s*(\d{1,2})[.)]\s+", text)
    bold_names = re.findall(r"(?m)^\s*\d{1,2}[.)]\s+\*\*(.+?)\*\*", text)
    normalized_names = [name.strip().casefold() for name in bold_names]
    return {
        "format": "numbered_text",
        "idea_count": len(numbered),
        "expected_idea_count": expected_ideas,
        "idea_count_valid": len(numbered) == expected_ideas,
        "unique_nonempty_names": bool(normalized_names)
        and len(normalized_names) == len(numbered)
        and all(normalized_names)
        and len(set(normalized_names)) == len(normalized_names),
        "prioritization_notes_present": bool(re.search(r"priorit", text, flags=re.IGNORECASE)),
    }


def score_one_order(
    client: OpenAI, *, model: str, brief: dict[str, Any], response_a: str, response_b: str
) -> tuple[PairwiseJudgment, dict[str, int]]:
    """Ask the configured OpenAI model to judge one answer ordering."""
    response = client.responses.parse(
        model=model,
        store=False,
        input=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": JUDGE_USER_PROMPT_TEMPLATE.format(
                    brief=json.dumps(brief, ensure_ascii=False, sort_keys=True, indent=2),
                    response_a=response_a,
                    response_b=response_b,
                ),
            },
        ],
        text_format=PairwiseJudgment,
    )
    if response.output_parsed is None:
        raise RuntimeError("OpenAI judge returned no parsed result.")
    usage = {
        "input_tokens": int(response.usage.input_tokens) if response.usage else 0,
        "output_tokens": int(response.usage.output_tokens) if response.usage else 0,
    }
    return response.output_parsed, usage


def normalize_vote(vote: Vote, *, reversed_order: bool) -> str:
    """Map a judge letter to the original base or tuned model identity."""
    if vote == "tie":
        return "tie"
    if reversed_order:
        vote = "B" if vote == "A" else "A"
    return "base" if vote == "A" else "tuned"


def resolve_ordered_votes(first: Vote, reversed_vote: Vote) -> dict[str, Any]:
    """Resolve swapped-order judgments and flag order-sensitive decisions."""
    first_identity = normalize_vote(first, reversed_order=False)
    reversed_identity = normalize_vote(reversed_vote, reversed_order=True)
    return {
        "winner": first_identity if first_identity == reversed_identity else None,
        "order_disagreement": first_identity != reversed_identity,
        "base_first_vote": first_identity,
        "tuned_first_vote_normalized": reversed_identity,
    }


def stratified_ids(tests: dict[str, dict[str, Any]], limit: int | None) -> list[str]:
    """Select test examples round-robin across industries."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for source_id, row in sorted(tests.items()):
        industry = str(row.get("input", {}).get("industry", ""))
        grouped[industry].append(source_id)
    industries = sorted(grouped)
    if limit is None:
        return [source_id for industry in industries for source_id in grouped[industry]]
    selected: list[str] = []
    while len(selected) < limit and any(grouped.values()):
        for industry in industries:
            if grouped[industry] and len(selected) < limit:
                selected.append(grouped[industry].pop(0))
    return selected


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize preferences and order disagreement per criterion."""
    summaries: dict[str, Any] = {}
    for criterion in CRITERIA:
        resolved = [row["judgments"][criterion]["winner"] for row in rows]
        resolved = [winner for winner in resolved if winner is not None]
        counts = Counter(resolved)
        disagreements = sum(row["judgments"][criterion]["order_disagreement"] for row in rows)
        summaries[criterion] = {
            "resolved_count": len(resolved),
            "base_wins": counts["base"],
            "tuned_wins": counts["tuned"],
            "ties": counts["tie"],
            "order_disagreements": disagreements,
            "order_disagreement_rate": disagreements / len(rows) if rows else None,
        }
    return summaries


def prompt_fingerprint() -> str:
    """Hash exact judge prompt text for reproducibility."""
    content = JUDGE_SYSTEM_PROMPT + "\n" + JUDGE_USER_PROMPT_TEMPLATE
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def run_evaluation(
    *,
    test_file: Path,
    base_file: Path,
    tuned_file: Path,
    output_file: Path,
    model: str,
    expected_ideas: int,
    limit: int | None,
) -> dict[str, Any]:
    """Run deterministic checks and two order-swapped judgments per example."""
    if output_file.exists():
        raise FileExistsError(f"Output already exists: {output_file}")
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError("Set OPENAI_API_KEY in .env before running the judge.")

    tests = load_jsonl_map(test_file, id_fields=("id",))
    base = load_jsonl_map(base_file, id_fields=("source_id", "id"))
    tuned = load_jsonl_map(tuned_file, id_fields=("source_id", "id"))
    base_ids = set(base)
    tuned_ids = set(tuned)
    unknown_ids = (base_ids | tuned_ids) - set(tests)
    if base_ids != tuned_ids or unknown_ids:
        raise ValueError(
            "Base and tuned predictions must cover the same IDs from the test file. "
            f"base_missing={sorted(tuned_ids - base_ids)}; tuned_missing={sorted(base_ids - tuned_ids)}; "
            f"unknown_ids={sorted(unknown_ids)}"
        )
    matched_tests = {source_id: tests[source_id] for source_id in base_ids}
    source_ids = stratified_ids(matched_tests, limit)
    if not source_ids:
        raise ValueError("No test examples are available.")

    client = OpenAI()
    rows: list[dict[str, Any]] = []
    token_usage = Counter()
    for source_id in source_ids:
        brief = tests[source_id].get("input")
        if not isinstance(brief, dict):
            raise ValueError(f"Test example {source_id!r} has no object-valued input brief.")
        base_prediction = prediction_from_row(base[source_id], source_id=source_id)
        tuned_prediction = prediction_from_row(tuned[source_id], source_id=source_id)
        base_text = render_content(base_prediction.content)
        tuned_text = render_content(tuned_prediction.content)

        forward, usage = score_one_order(
            client, model=model, brief=brief, response_a=base_text, response_b=tuned_text
        )
        token_usage.update(usage)
        reverse, usage = score_one_order(
            client, model=model, brief=brief, response_a=tuned_text, response_b=base_text
        )
        token_usage.update(usage)
        judgments = {
            criterion: resolve_ordered_votes(
                getattr(forward, criterion).winner, getattr(reverse, criterion).winner
            )
            for criterion in CRITERIA
        }
        rows.append(
            {
                "source_id": source_id,
                "industry": brief.get("industry"),
                "automatic": {
                    "base": automatic_checks(base_prediction.content, expected_ideas=expected_ideas),
                    "tuned": automatic_checks(tuned_prediction.content, expected_ideas=expected_ideas),
                },
                "judgments": judgments,
                "judge_evidence": {
                    "base_first": forward.model_dump(mode="json"),
                    "tuned_first": reverse.model_dump(mode="json"),
                },
                "model_runtime": {
                    "base_latency_seconds": base_prediction.latency_seconds,
                    "tuned_latency_seconds": tuned_prediction.latency_seconds,
                    "base_input_tokens": base_prediction.input_tokens,
                    "base_output_tokens": base_prediction.output_tokens,
                    "tuned_input_tokens": tuned_prediction.input_tokens,
                    "tuned_output_tokens": tuned_prediction.output_tokens,
                },
            }
        )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "judge": {"provider": "openai", "model": model, "store": False},
        "task": {"name": "brainstorming_pairwise", "expected_ideas": expected_ideas},
        "inputs": {
            "test_file": str(test_file),
            "test_sha256": sha256_file(test_file),
            "base_predictions_file": str(base_file),
            "base_predictions_sha256": sha256_file(base_file),
            "tuned_predictions_file": str(tuned_file),
            "tuned_predictions_sha256": sha256_file(tuned_file),
        },
        "judge_prompt_sha256": prompt_fingerprint(),
        "sample_count": len(rows),
        "judge_calls": len(rows) * 2,
        "judge_token_usage": dict(token_usage),
        "metrics": {
            "pairwise": metric_summary(rows),
            "automatic_shape": {
                model_key: {
                    key: sum(bool(row["automatic"][model_key][key]) for row in rows) / len(rows)
                    for key in ("idea_count_valid", "unique_nonempty_names", "prioritization_notes_present")
                }
                for model_key in ("base", "tuned")
            },
        },
        "results": rows,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    """Run OpenAI evaluation only after explicit authorization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--tuned-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--expected-ideas", type=int, default=10)
    parser.add_argument("--limit", type=int, default=30, help="Maximum examples; round-robin by industry.")
    parser.add_argument("--execute", action="store_true", help="Authorize OpenAI judge API calls.")
    args = parser.parse_args()
    if not args.execute:
        parser.error("No API calls made. Review code and pass --execute to authorize judging.")
    if args.expected_ideas < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--expected-ideas and --limit must be positive.")

    report = run_evaluation(
        test_file=args.test_file,
        base_file=args.base_predictions,
        tuned_file=args.tuned_predictions,
        output_file=args.output,
        model=args.model,
        expected_ideas=args.expected_ideas,
        limit=args.limit,
    )
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
