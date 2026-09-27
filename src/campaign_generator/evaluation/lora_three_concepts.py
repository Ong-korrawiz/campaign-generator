"""Smoke-test three distinct LoRA concepts with the exact training prompt."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from ..api.service import _json_object
from ..io import read_jsonl
from ..prompts.campaign import SYSTEM_PROMPT_V2, render_user_prompt
from ..schemas import ApiCampaignBrief, CampaignDirectionV2
from ..training.train_lora import MODEL_ID, MODEL_REVISION
from .predict_concepts import localize_gcs_uri, upload_gcs_uri


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-source", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute or not args.output.startswith("gs://"):
        parser.error("Pass --execute and a gs:// output URI")
    if args.per_source < 1 or args.max_attempts < 3:
        parser.error("Need at least one brief per source and three generation attempts")

    with tempfile.TemporaryDirectory(prefix="lora-three-") as tmp:
        root = Path(tmp)
        validation = localize_gcs_uri(args.validation_file, destination=root / "validation.jsonl")
        adapter = localize_gcs_uri(args.adapter, destination=root / "adapter", directory=True)
        selected = []
        source_counts: Counter[str] = Counter()
        for _, row in read_jsonl(validation):
            source = "synthetic" if row["id"].startswith("synthetic-v3-") else "legacy"
            if source_counts[source] < args.per_source:
                selected.append(row)
                source_counts[source] += 1
        if any(source_counts[source] < args.per_source for source in ("legacy", "synthetic")):
            raise ValueError(f"Insufficient validation briefs: {source_counts}")

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, torch_dtype="auto", device_map="auto"
        )
        model = PeftModel.from_pretrained(base, str(adapter))
        model.eval()

        results = []
        for brief_index, row in enumerate(selected):
            brief = ApiCampaignBrief.model_validate(row["input"])
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_V2},
                {"role": "user", "content": render_user_prompt(row["input"])},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
            concepts = []
            names = set()
            signatures = set()
            attempts = []
            for attempt in range(args.max_attempts):
                torch.manual_seed(42 + brief_index * 100 + attempt)
                started = time.perf_counter()
                with torch.no_grad():
                    output = model.generate(
                        prompt,
                        max_new_tokens=1536,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                completion = output[0][prompt.shape[-1] :]
                raw = tokenizer.decode(completion, skip_special_tokens=True).strip()
                reason = "valid"
                try:
                    concept = CampaignDirectionV2.model_validate(_json_object(raw))
                    channels = {item.channel for item in concept.channel_plan} | {
                        item.channel for item in concept.asset_plan
                    }
                    name = concept.campaign_direction.casefold().strip()
                    signature = (
                        concept.key_message.casefold().strip(),
                        concept.campaign_description.casefold().strip(),
                    )
                    if not channels.issubset(set(brief.channels)):
                        reason = "unapproved_channel"
                    elif name in names:
                        reason = "duplicate_name"
                    elif signature in signatures:
                        reason = "duplicate_content"
                    else:
                        concepts.append(concept.model_dump(mode="json"))
                        names.add(name)
                        signatures.add(signature)
                except (ValueError, ValidationError, json.JSONDecodeError) as exc:
                    reason = "invalid_json" if isinstance(exc, json.JSONDecodeError) else "schema_or_parse"
                attempts.append(
                    {
                        "reason": reason,
                        "output_tokens": int(completion.shape[-1]),
                        "latency_seconds": round(time.perf_counter() - started, 3),
                    }
                )
                if len(concepts) == 3:
                    break
            results.append(
                {
                    "source_id": row["id"],
                    "source_type": "synthetic" if row["id"].startswith("synthetic-v3-") else "legacy",
                    "three_valid_distinct": len(concepts) == 3,
                    "valid_concept_count": len(concepts),
                    "attempts": attempts,
                    "concepts": concepts,
                }
            )
            print(
                json.dumps(
                    {
                        "completed_briefs": len(results),
                        "total_briefs": len(selected),
                        "three_valid_distinct": sum(x["three_valid_distinct"] for x in results),
                    }
                ),
                flush=True,
            )

        output_file = root / "lora-three-concepts.jsonl"
        output_file.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8"
        )
        upload_gcs_uri(output_file, args.output)
        summary = {
            "total": len(results),
            "three_valid_distinct": sum(x["three_valid_distinct"] for x in results),
            "attempt_errors": dict(Counter(a["reason"] for r in results for a in r["attempts"])),
            "output": args.output,
        }
        summary_file = root / "summary.json"
        summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        upload_gcs_uri(summary_file, args.output.removesuffix(".jsonl") + "-summary.json")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
