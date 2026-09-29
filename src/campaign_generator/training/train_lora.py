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

from ..config import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    CODE_COMMIT_ENV,
    TRAIN_CHECKPOINT_LIMIT,
    TRAIN_CHECKPOINT_STEPS,
    TRAIN_DEFAULT_MAX_STEPS,
    TRAIN_DEFAULT_WORK_DIR,
    TRAIN_EPOCHS,
    TRAIN_EVAL_BATCH,
    TRAIN_EVAL_METRIC,
    TRAIN_GRADIENT_ACCUMULATION,
    TRAIN_LABEL_IGNORE_INDEX,
    TRAIN_LEARNING_RATE,
    TRAIN_LOGGING_STEPS,
    TRAIN_LORA_ALPHA,
    TRAIN_LORA_BIAS,
    TRAIN_LORA_DROPOUT,
    TRAIN_LORA_R,
    TRAIN_LORA_TARGET_MODULES,
    TRAIN_LORA_TASK_TYPE,
    TRAIN_MAX_SEQUENCE_LENGTH,
    TRAIN_MICROBATCH,
    TRAIN_SEED,
    TRAIN_SEQUENCE_PERCENTILE,
    TRAIN_SEQUENCE_ROUND_TO,
    TRAIN_SMOKE_GRADIENT_ACCUMULATION,
    TRAIN_SMOKE_MAX_STEPS,
    TRAIN_SMOKE_MICROBATCH,
    TRAIN_SMOKE_ROW_COUNT,
    TRAIN_SMOKE_SEQUENCE_LIMIT,
)
from ..schemas import TrainingRecordV2


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
        labels = [TRAIN_LABEL_IGNORE_INDEX] * prompt_length + full_ids[prompt_length:]
        if not any(token != TRAIN_LABEL_IGNORE_INDEX for token in labels):
            raise ValueError(f"No assistant completion tokens remain for {row['id']}")
        encoded.append({"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels})
    return encoded, truncated


def _percentile_95(lengths: list[int]) -> int:
    ordered = sorted(lengths)
    index = max(0, min(len(ordered) - 1, int(TRAIN_SEQUENCE_PERCENTILE * len(ordered) + 0.999) - 1))
    rounded = (ordered[index] + TRAIN_SEQUENCE_ROUND_TO - 1) // TRAIN_SEQUENCE_ROUND_TO
    return min(TRAIN_MAX_SEQUENCE_LENGTH, rounded * TRAIN_SEQUENCE_ROUND_TO)


def _download_dataset(uri: str, destination: Path) -> None:
    if uri.startswith("gs://"):
        from google.cloud import storage

        bucket_name, prefix = uri[5:].split("/", 1)
        client = storage.Client()
        for name in ("manifest.json", "train.jsonl", "validation.jsonl"):
            blob = client.bucket(bucket_name).blob(f"{prefix.rstrip('/')}/{name}")
            if not blob.exists():
                raise FileNotFoundError(f"Missing gs://{bucket_name}/{blob.name}")
            blob.download_to_filename(destination / name)
    else:
        source = Path(uri)
        for name in ("manifest.json", "train.jsonl", "validation.jsonl"):
            shutil.copyfile(source / name, destination / name)


def _load_training_splits(dataset_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    train_rows = _read_jsonl(dataset_dir / "train.jsonl")
    validation_rows = _read_jsonl(dataset_dir / "validation.jsonl")
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        content = (dataset_dir / f"{split}.jsonl").read_bytes()
        expected_hash = (manifest.get("training_files_sha256") or manifest.get("files_sha256") or {}).get(
            split
        )
        if (
            len(rows) != manifest["split_counts"][split]
            or hashlib.sha256(content).hexdigest() != expected_hash
        ):
            raise ValueError(f"Dataset {split} split differs from manifest")
    if not train_rows or not validation_rows:
        raise ValueError("Training and validation splits must not be empty")
    return train_rows, validation_rows


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
    train_rows, validation_rows = _load_training_splits(dataset_dir)
    if args.smoke_overfit:
        train_rows = train_rows[:TRAIN_SMOKE_ROW_COUNT]
        validation_rows = train_rows

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, revision=BASE_MODEL_REVISION)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_lengths = [len(tokenizer.apply_chat_template(row["messages"], tokenize=True)) for row in train_rows]
    max_length = (
        min(TRAIN_SMOKE_SEQUENCE_LIMIT, _percentile_95(train_lengths))
        if args.smoke_overfit
        else _percentile_95(train_lengths)
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
        BASE_MODEL_ID, revision=BASE_MODEL_REVISION, torch_dtype=torch.bfloat16, use_cache=False
    )
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=TRAIN_LORA_R,
            lora_alpha=TRAIN_LORA_ALPHA,
            lora_dropout=TRAIN_LORA_DROPOUT,
            bias=TRAIN_LORA_BIAS,
            task_type=TRAIN_LORA_TASK_TYPE,
            target_modules=list(TRAIN_LORA_TARGET_MODULES),
        ),
    )
    model.print_trainable_parameters()
    output_dir = args.work_dir / "trainer-output"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=TRAIN_LEARNING_RATE,
        num_train_epochs=TRAIN_EPOCHS,
        max_steps=TRAIN_SMOKE_MAX_STEPS if args.smoke_overfit else TRAIN_DEFAULT_MAX_STEPS,
        per_device_train_batch_size=TRAIN_SMOKE_MICROBATCH if args.smoke_overfit else TRAIN_MICROBATCH,
        per_device_eval_batch_size=TRAIN_EVAL_BATCH,
        gradient_accumulation_steps=(
            TRAIN_SMOKE_GRADIENT_ACCUMULATION if args.smoke_overfit else TRAIN_GRADIENT_ACCUMULATION
        ),
        bf16=True,
        logging_steps=TRAIN_LOGGING_STEPS,
        save_strategy="no" if args.smoke_overfit else "steps",
        save_steps=TRAIN_CHECKPOINT_STEPS,
        save_total_limit=TRAIN_CHECKPOINT_LIMIT,
        eval_strategy="steps",
        eval_steps=TRAIN_CHECKPOINT_STEPS,
        load_best_model_at_end=not args.smoke_overfit,
        metric_for_best_model=TRAIN_EVAL_METRIC,
        greater_is_better=False,
        seed=TRAIN_SEED,
        data_seed=TRAIN_SEED,
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer, padding=True, label_pad_token_id=TRAIN_LABEL_IGNORE_INDEX, return_tensors="pt"
    )

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
        "model_id": BASE_MODEL_ID,
        "model_revision": BASE_MODEL_REVISION,
        "dataset_uri": args.dataset_uri,
        "dataset_hashes": dataset_hashes,
        "code_commit": os.getenv(CODE_COMMIT_ENV, "unknown"),
        "lora": {"r": TRAIN_LORA_R, "alpha": TRAIN_LORA_ALPHA, "dropout": TRAIN_LORA_DROPOUT},
        "training": {
            "lr": TRAIN_LEARNING_RATE,
            "epochs": TRAIN_EPOCHS,
            "microbatch": TRAIN_MICROBATCH,
            "gradient_accumulation": TRAIN_GRADIENT_ACCUMULATION,
            "seed": TRAIN_SEED,
        },
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
    parser.add_argument("--work-dir", type=Path, default=TRAIN_DEFAULT_WORK_DIR)
    parser.add_argument(
        "--smoke-overfit",
        action="store_true",
        help=f"Run {TRAIN_SMOKE_MAX_STEPS} GPU steps over {TRAIN_SMOKE_ROW_COUNT} rows and require loss to decrease",
    )
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2, default=str))


if __name__ == "__main__":
    main()
