"""Run baseline or LoRA inference over the fresh synthetic holdout."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from ..api.service import CampaignGenerator
from ..io import read_jsonl
from ..schemas import ApiCampaignBrief


class LocalBackend:
    def __init__(self, model, tokenizer, torch) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch
        self.output_tokens = 0

    async def complete(self, *, messages: list[dict[str, str]], max_tokens: int) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with self.torch.no_grad():
            output = self.model.generate(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        completion = output[0][prompt.shape[-1] :]
        self.output_tokens += int(completion.shape[-1])
        return self.tokenizer.decode(completion, skip_special_tokens=True).strip()


def localize_gcs_uri(uri: str, *, destination: Path, directory: bool = False) -> Path:
    """Download a GCS object or directory prefix for a Vertex GPU worker."""
    if not uri.startswith("gs://"):
        return Path(uri)
    from google.cloud import storage

    bucket_name, _, object_name = uri[5:].partition("/")
    if not bucket_name or not object_name:
        raise ValueError(f"Invalid GCS URI: {uri}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    if directory:
        prefix = object_name.rstrip("/") + "/"
        destination.mkdir(parents=True, exist_ok=True)
        blobs = list(bucket.list_blobs(prefix=prefix))
        if not blobs:
            raise FileNotFoundError(f"No files found at {uri}")
        for blob in blobs:
            relative = blob.name.removeprefix(prefix)
            if relative:
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                blob.download_to_filename(target)
        return destination
    blob = bucket.blob(object_name)
    if not blob.exists():
        raise FileNotFoundError(uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(destination)
    return destination


def upload_gcs_uri(source: Path, uri: str) -> None:
    """Upload a local prediction file to its requested GCS object URI."""
    if not uri.startswith("gs://"):
        raise ValueError("Output URI must use gs://")
    from google.cloud import storage

    bucket_name, _, object_name = uri[5:].partition("/")
    if not bucket_name or not object_name:
        raise ValueError(f"Invalid GCS URI: {uri}")
    storage.Client().bucket(bucket_name).blob(object_name).upload_from_filename(source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", required=True, help="Local JSONL path or gs:// object URI")
    parser.add_argument("--output", required=True, help="Local JSONL path or gs:// object URI")
    parser.add_argument("--adapter", help="Local adapter directory or gs:// directory prefix")
    parser.add_argument("--limit", type=int, default=36)
    parser.add_argument("--execute", action="store_true", help="Explicitly authorize local GPU inference")
    args = parser.parse_args()
    if not args.execute:
        parser.error("No model loaded. Review code and pass --execute to authorize inference.")
    if not args.output.startswith("gs://") and Path(args.output).exists():
        parser.error(f"Output already exists: {args.output}")
    if args.limit < 1:
        parser.error("--limit must be positive")

    with tempfile.TemporaryDirectory(prefix="campaign-eval-") as tmp:
        root = Path(tmp)
        test_file = localize_gcs_uri(args.test_file, destination=root / "test.jsonl")
        adapter_path = (
            localize_gcs_uri(args.adapter, destination=root / "adapter", directory=True)
            if args.adapter
            else None
        )
        output_path = root / "predictions.jsonl" if args.output.startswith("gs://") else Path(args.output)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ..training.train_lora import MODEL_ID, MODEL_REVISION

        if not torch.cuda.is_available():
            raise RuntimeError("Campaign prediction requires CUDA")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, torch_dtype="auto", device_map="auto"
        )
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()
        backend = LocalBackend(model, tokenizer, torch)
        generator = CampaignGenerator(backend)

        tests = [
            row for _, row in read_jsonl(test_file) if str(row.get("id", "")).startswith("synthetic-v3-")
        ]
        if len(tests) != 36:
            raise ValueError(f"Expected 36 fresh synthetic holdout rows, found {len(tests)}")
        if args.limit > len(tests):
            raise ValueError("Requested limit exceeds the 36-row primary holdout")
        predictions: list[dict[str, Any]] = []
        for row in tests[: args.limit]:
            started = time.perf_counter()
            try:
                brief = ApiCampaignBrief.model_validate(row["input"])
                response = asyncio.run(generator.generate(brief))
                content: Any = response.model_dump(mode="json")
                errors = []
            except Exception as exc:
                content = {"invalid_output": str(exc)}
                errors = [type(exc).__name__]
            predictions.append(
                {
                    "source_id": row["id"],
                    "output": content,
                    "latency_seconds": round(time.perf_counter() - started, 3),
                    "output_tokens": backend.output_tokens,
                    "validation_errors": errors,
                }
            )
            backend.output_tokens = 0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8"
        )
        if args.output.startswith("gs://"):
            upload_gcs_uri(output_path, args.output)
        print(
            json.dumps(
                {"prediction_count": len(predictions), "adapter": bool(adapter_path), "output": args.output}
            )
        )


if __name__ == "__main__":
    main()
