"""Compare training and API first-attempt prompts on the same validation briefs."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from ..api.service import CampaignGenerator, _json_object
from ..io import read_jsonl
from ..prompts.campaign import SYSTEM_PROMPT_V2, render_user_prompt
from ..schemas import ApiCampaignBrief, CampaignDirectionV2
from .predict_concepts import LocalBackend, localize_gcs_uri, upload_gcs_uri


class PromptCaptured(Exception):
    pass


class CaptureBackend:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    async def complete(self, *, messages: list[dict[str, str]], max_tokens: int) -> str:
        self.messages = messages
        raise PromptCaptured


def api_first_prompt(brief: ApiCampaignBrief) -> list[dict[str, str]]:
    backend = CaptureBackend()
    try:
        asyncio.run(CampaignGenerator(backend).generate(brief))
    except PromptCaptured:
        return backend.messages
    raise AssertionError("API prompt was not captured")


def evaluate(raw: str, brief: ApiCampaignBrief) -> dict[str, object]:
    try:
        parsed = _json_object(raw)
    except (ValueError, json.JSONDecodeError):
        return {"json_valid": False, "schema_valid": False, "channel_valid": False, "error": "invalid_json"}
    try:
        concept = CampaignDirectionV2.model_validate(parsed)
    except ValidationError as exc:
        missing = sorted({str(e["loc"][0]) for e in exc.errors() if e["type"] == "missing"})
        return {
            "json_valid": True,
            "schema_valid": False,
            "channel_valid": False,
            "error": "schema",
            "missing_fields": missing,
            "keys": sorted(parsed),
        }
    used = {item.channel for item in concept.channel_plan} | {item.channel for item in concept.asset_plan}
    channel_valid = used.issubset(set(brief.channels))
    return {
        "json_valid": True,
        "schema_valid": True,
        "channel_valid": channel_valid,
        "error": "" if channel_valid else "unapproved_channel",
        "keys": sorted(parsed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-source", type=int, default=6)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Pass --execute to run GPU inference")
    if args.per_source < 1:
        parser.error("--per-source must be positive")
    if not args.output.startswith("gs://"):
        parser.error("--output must be a gs:// URI")

    with tempfile.TemporaryDirectory(prefix="prompt-diagnostic-") as tmp:
        root = Path(tmp)
        validation = localize_gcs_uri(args.validation_file, destination=root / "validation.jsonl")
        adapter = localize_gcs_uri(args.adapter, destination=root / "adapter", directory=True)
        rows = [row for _, row in read_jsonl(validation)]
        selected: list[dict] = []
        counts: Counter[str] = Counter()
        for row in rows:
            source = "synthetic" if row["id"].startswith("synthetic-v3-") else "legacy"
            if counts[source] < args.per_source:
                selected.append(row)
                counts[source] += 1
        if any(counts[source] != args.per_source for source in ("synthetic", "legacy")):
            raise ValueError(f"Insufficient validation rows by source: {counts}")

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ..training.train_lora import MODEL_ID, MODEL_REVISION

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, torch_dtype="auto", device_map="auto"
        )
        model = PeftModel.from_pretrained(base, str(adapter))
        model.eval()
        backend = LocalBackend(model, tokenizer, torch)

        results: list[dict] = []
        for row in selected:
            brief = ApiCampaignBrief.model_validate(row["input"])
            prompts = {
                "training": [
                    {"role": "system", "content": SYSTEM_PROMPT_V2},
                    {"role": "user", "content": render_user_prompt(row["input"])},
                ],
                "api_first": api_first_prompt(brief),
            }
            for mode, messages in prompts.items():
                before = backend.output_tokens
                started = time.perf_counter()
                raw = asyncio.run(backend.complete(messages=messages, max_tokens=1536))
                results.append(
                    {
                        "source_id": row["id"],
                        "source_type": "synthetic" if row["id"].startswith("synthetic-v3-") else "legacy",
                        "prompt_mode": mode,
                        "raw_output": raw,
                        "output_tokens": backend.output_tokens - before,
                        "latency_seconds": round(time.perf_counter() - started, 3),
                        **evaluate(raw, brief),
                    }
                )
            print(
                json.dumps({"completed_briefs": len(results) // 2, "total_briefs": len(selected)}), flush=True
            )

        output = root / "prompt-comparison.jsonl"
        output.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8"
        )
        upload_gcs_uri(output, args.output)
        summary: dict[str, dict[str, int]] = {}
        for mode in ("training", "api_first"):
            subset = [r for r in results if r["prompt_mode"] == mode]
            summary[mode] = {
                key: sum(bool(r[key]) for r in subset)
                for key in ("json_valid", "schema_valid", "channel_valid")
            }
            summary[mode]["total"] = len(subset)
        summary_file = root / "summary.json"
        summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary_uri = args.output.removesuffix(".jsonl") + "-summary.json"
        upload_gcs_uri(summary_file, summary_uri)
        print(json.dumps({"summary": summary, "output": args.output, "summary_uri": summary_uri}), flush=True)


if __name__ == "__main__":
    main()
