"""Train a completion-only Qwen LoRA adapter and checkpoint it to GCS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..schemas import TrainingRecordV2

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
MAX_SEQUENCE_LENGTH = 4096


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [TrainingRecordV2.model_validate(row).model_dump(mode="json") for row in rows]


def _tokenize(rows, tokenizer, max_length: int):
    encoded = []
    truncated = 0
    for row in rows:
        messages = row["messages"]
        prompt_ids = tokenizer.apply_chat_template(messages[:2], tokenize=True, add_generation_prompt=True)
        full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"Chat template prefix differs for {row['id']}; refusing to train on prompt tokens"
            )
        if len(full_ids) > max_length:
            truncated += 1
            full_ids = full_ids[:max_length]
        prompt_length = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_length + full_ids[prompt_length:]
        if not any(token != -100 for token in labels):
            raise ValueError(f"No assistant completion tokens remain for {row['id']}")
        encoded.append({"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels})
    return encoded, truncated


def _percentile_95(lengths: list[int]) -> int:
    ordered = sorted(lengths)
    index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999) - 1))
    return min(MAX_SEQUENCE_LENGTH, ((ordered[index] + 255) // 256) * 256)


def _download_dataset(uri: str, destination: Path) -> None:
    if uri.startswith("gs://"):
        from google.cloud import storage

        bucket_name, prefix = uri[5:].split("/", 1)
        client = storage.Client()
        for split in ("train", "validation"):
            blob = client.bucket(bucket_name).blob(f"{prefix.rstrip('/')}/{split}.jsonl")
            if not blob.exists():
                raise FileNotFoundError(f"Missing gs://{bucket_name}/{blob.name}")
            blob.download_to_filename(destination / f"{split}.jsonl")
    else:
        source = Path(uri)
        for split in ("train", "validation"):
            shutil.copyfile(source / f"{split}.jsonl", destination / f"{split}.jsonl")


def _latest_complete_checkpoint(output_uri: str, local_dir: Path) -> str | None:
    if not output_uri.startswith("gs://"):
        checkpoints = sorted(
            local_dir.glob("checkpoint-*.tar.gz"),
            key=lambda p: int(p.name.removeprefix("checkpoint-").removesuffix(".tar.gz")),
        )
        if not checkpoints:
            return None
        archive = checkpoints[-1]
        target = local_dir / "resume"
        target.mkdir(exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(target)
        return str(target / archive.name.removesuffix(".tar.gz"))
    from google.cloud import storage

    bucket_name, _, prefix = output_uri[5:].partition("/")
    if not prefix:
        prefix = ""
    blobs = list(storage.Client().bucket(bucket_name).list_blobs(prefix=f"{prefix.rstrip('/')}/checkpoints/"))
    complete = [blob for blob in blobs if blob.name.endswith(".complete")]
    if not complete:
        return None
    marker = max(complete, key=lambda blob: int(blob.name.split("checkpoint-")[-1].split(".")[0]))
    archive_name = marker.name.removesuffix(".complete") + ".tar.gz"
    bucket = storage.Client().bucket(bucket_name)
    archive_path = local_dir / Path(archive_name).name
    bucket.blob(archive_name).download_to_filename(archive_path)
    target = local_dir / "resume"
    target.mkdir(exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(target)
    return str(target / archive_path.name.removesuffix(".tar.gz"))


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("LoRA training requires a CUDA GPU")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = args.work_dir / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    _download_dataset(args.dataset_uri, dataset_dir)
    train_rows = _read_jsonl(dataset_dir / "train.jsonl")
    validation_rows = _read_jsonl(dataset_dir / "validation.jsonl")
    if not args.smoke_overfit and (len(train_rows) != 168 or len(validation_rows) != 24):
        raise ValueError("Expected approved dataset v3 splits of 168 train and 24 validation rows")
    if args.smoke_overfit:
        train_rows = train_rows[:8]
        validation_rows = train_rows

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_lengths = [len(tokenizer.apply_chat_template(row["messages"], tokenize=True)) for row in train_rows]
    max_length = (
        min(1024, _percentile_95(train_lengths)) if args.smoke_overfit else _percentile_95(train_lengths)
    )
    tokenized_train, train_truncated = _tokenize(train_rows, tokenizer, max_length)
    tokenized_validation, validation_truncated = _tokenize(validation_rows, tokenizer, max_length)

    class ChatDataset(torch.utils.data.Dataset):
        def __init__(self, values):
            self.values = values

        def __len__(self):
            return len(self.values)

        def __getitem__(self, index):
            return self.values[index]

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.bfloat16, use_cache=False
    )
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.print_trainable_parameters()
    output_dir = args.work_dir / "trainer-output"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-4,
        num_train_epochs=3,
        max_steps=20 if args.smoke_overfit else -1,
        per_device_train_batch_size=1 if args.smoke_overfit else 4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1 if args.smoke_overfit else 8,
        bf16=True,
        logging_steps=5,
        save_strategy="no" if args.smoke_overfit else "steps",
        save_steps=25,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=25,
        load_best_model_at_end=not args.smoke_overfit,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42,
        data_seed=42,
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
    )
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100, return_tensors="pt")

    from transformers import TrainerCallback

    class GCSCheckpointCallback(TrainerCallback):
        def on_save(self, training_args, state, control, **kwargs):
            checkpoint = output_dir / f"checkpoint-{state.global_step}"
            archive = args.work_dir / f"checkpoint-{state.global_step}.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(checkpoint, arcname=checkpoint.name)
            if args.output_uri.startswith("gs://"):
                from google.cloud import storage

                bucket_name, _, prefix = args.output_uri[5:].partition("/")
                bucket = storage.Client().bucket(bucket_name)
                object_prefix = f"{prefix.rstrip('/')}/checkpoints/{archive.name}"
                bucket.blob(object_prefix).upload_from_filename(archive)
                bucket.blob(object_prefix.removesuffix(".tar.gz") + ".complete").upload_from_string(
                    str(state.global_step)
                )
            else:
                shutil.copyfile(archive, args.work_dir / archive.name)
            return control

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ChatDataset(tokenized_train),
        eval_dataset=ChatDataset(tokenized_validation),
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[GCSCheckpointCallback()],
    )
    resume = _latest_complete_checkpoint(args.output_uri, args.work_dir)
    initial_eval_loss = trainer.evaluate().get("eval_loss") if args.smoke_overfit else None
    result = trainer.train(resume_from_checkpoint=None if args.smoke_overfit else resume)
    metrics = trainer.evaluate()
    if args.smoke_overfit and metrics.get("eval_loss", float("inf")) >= initial_eval_loss:
        raise RuntimeError("8-example overfit smoke did not reduce assistant completion loss")
    final_dir = args.work_dir / "adapter"
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    dataset_hashes = {
        split: hashlib.sha256((dataset_dir / f"{split}.jsonl").read_bytes()).hexdigest()
        for split in ("train", "validation")
    }
    report = {
        "run_id": args.run_id,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_uri": args.dataset_uri,
        "dataset_hashes": dataset_hashes,
        "code_commit": os.getenv("CODE_COMMIT", "unknown"),
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
        "training": {"lr": 2e-4, "epochs": 3, "microbatch": 4, "gradient_accumulation": 8, "seed": 42},
        "sequence_limit": max_length,
        "train_truncated_rows": train_truncated,
        "validation_truncated_rows": validation_truncated,
        "train_metrics": result.metrics,
        "validation_metrics": metrics,
        "resumed_from": resume,
        "smoke_overfit": args.smoke_overfit,
        "initial_eval_loss": initial_eval_loss,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.work_dir / "run_manifest.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    if args.output_uri.startswith("gs://"):
        from google.cloud import storage

        bucket_name, _, prefix = args.output_uri[5:].partition("/")
        bucket = storage.Client().bucket(bucket_name)
        base = prefix.rstrip("/")
        for file in final_dir.rglob("*"):
            if file.is_file():
                bucket.blob(f"{base}/adapter/{file.relative_to(final_dir).as_posix()}").upload_from_filename(
                    file
                )
        bucket.blob(f"{base}/run_manifest.json").upload_from_filename(args.work_dir / "run_manifest.json")
        bucket.blob(f"{base}/_COMPLETE").upload_from_string("ok")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/campaign-training"))
    parser.add_argument(
        "--smoke-overfit",
        action="store_true",
        help="Run 20 GPU steps over eight rows and require loss to decrease",
    )
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2, default=str))


if __name__ == "__main__":
    main()
