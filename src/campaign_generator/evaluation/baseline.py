"""GPU baseline/evaluation CLI. Requires an explicit --execute after code review."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..io import read_jsonl
from ..prompts.campaign import SYSTEM_PROMPT, render_user_prompt
from ..schemas import TrainingRecord, validate_output


def choose_diverse_test_rows(path: Path, limit: int, minimum_industries: int) -> list[dict[str, Any]]:
    """Choose held-out briefs round-robin across available industries."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, row in read_jsonl(path):
        TrainingRecord.model_validate(row)
        groups[row["input"]["industry"]].append(row)
    if len(groups) < minimum_industries:
        raise ValueError(f"Test split has {len(groups)} industries; need {minimum_industries}.")
    selected: list[dict[str, Any]] = []
    industries = sorted(groups)
    while len(selected) < limit and any(groups.values()):
        for industry in industries:
            if groups[industry] and len(selected) < limit:
                selected.append(groups[industry].pop(0))
    return selected


def main() -> None:
    """Run GPU inference only after the reviewer supplies --execute."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/day1.json"))
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, help="Optional LoRA adapter for Day 2 comparison")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly authorize model inference after reviewing this code",
    )
    args = parser.parse_args()
    if not args.execute:
        parser.error("Model inference is disabled until code review; pass --execute afterwards.")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = choose_diverse_test_rows(
        args.test_file, config["baseline"]["limit"], config["baseline"]["minimum_industries"]
    )
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Baseline requires a Colab GPU; no model was loaded.")
    model_id = config["model"]["repo_id"]
    revision = config["model"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, device_map="auto", torch_dtype="auto"
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()

    results: list[dict[str, Any]] = []
    for row in rows:
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_user_prompt(row["input"])},
        ]
        tokens = tokenizer.apply_chat_template(
            prompt_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        started = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                tokens,
                max_new_tokens=config["model"]["max_new_tokens"],
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - started
        raw = tokenizer.decode(generated[0][tokens.shape[-1] :], skip_special_tokens=True).strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        errors = validate_output(parsed) if parsed is not None else ["invalid_json"]
        results.append(
            {
                "source_id": row["id"],
                "industry": row["input"]["industry"],
                "latency_seconds": round(latency, 3),
                "json_valid": parsed is not None,
                "schema_valid": not errors,
                "validation_errors": errors,
                "raw_response": raw,
            }
        )

    result = {
        "model": {
            "repo_id": model_id,
            "revision": revision,
            "adapter": str(args.adapter) if args.adapter else None,
        },
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "test_file_sha256": hashlib.sha256(args.test_file.read_bytes()).hexdigest(),
        "sample_count": len(results),
        "metrics": {
            "json_valid_rate": sum(item["json_valid"] for item in results) / len(results),
            "schema_valid_rate": sum(item["schema_valid"] for item in results) / len(results),
            "mean_latency_seconds": round(sum(item["latency_seconds"] for item in results) / len(results), 3),
        },
        "predictions": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"]))


if __name__ == "__main__":
    main()
