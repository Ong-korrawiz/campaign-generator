"""Feedback snapshot tests use the committed immutable dataset as a base."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from pydantic import ValidationError

from campaign_generator.dataset_v3.feedback import build_version
from campaign_generator.schemas import TrainingRecordV2
from campaign_generator.training.train_lora import _load_training_splits

ROOT = Path(__file__).resolve().parents[1]


class FeedbackDatasetTests(TestCase):
    def setUp(self):
        self.base = {
            name: (ROOT / "dataset" / name).read_bytes()
            for name in ("manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl")
        }
        original = json.loads(self.base["train.jsonl"].splitlines()[0])
        self.brief = {**original["input"], "brand": "Feedback Test Brand", "product": "New product"}
        self.concept = original["output"]
        self.generation = {"brief": self.brief, "concepts": [self.concept] * 3}
        self.uri = "gs://example-dataset/versions/base"

    def test_up_votes_add_one_deduplicated_train_record(self):
        votes = [
            ("generation-a", 1, {"ratings": {"0": "up", "1": "down"}}, self.generation),
            ("generation-b", 1, {"ratings": {"0": "up"}}, self.generation),
        ]
        result = build_version(self.uri, self.base, votes)
        self.assertIsNotNone(result)
        manifest = json.loads(result["manifest.json"])
        self.assertEqual(manifest["split_counts"], {"train": 169, "validation": 24, "test": 48})
        self.assertEqual(manifest["feedback_added_count"], 1)
        self.assertEqual(result["validation.jsonl"], self.base["validation.jsonl"])
        row = TrainingRecordV2.model_validate_json(result["train.jsonl"].splitlines()[-1])
        self.assertEqual(row.source.annotation_status, "public_thumb_up_schema_validated_unreviewed")
        self.assertEqual(row.output.campaign_direction, self.concept["campaign_direction"])
        with TemporaryDirectory() as temp:
            for name in ("manifest.json", "train.jsonl", "validation.jsonl"):
                (Path(temp) / name).write_bytes(result[name])
            train_rows, validation_rows = _load_training_splits(Path(temp))
            self.assertEqual((len(train_rows), len(validation_rows)), (169, 24))

    def test_down_only_and_holdout_brief_do_not_add_rows(self):
        down = [("generation-a", 2, {"ratings": {"0": "down"}}, self.generation)]
        self.assertIsNone(build_version(self.uri, self.base, down))
        holdout = json.loads(self.base["validation.jsonl"].splitlines()[0])["input"]
        self.assertIsNone(
            build_version(
                self.uri,
                self.base,
                [
                    (
                        "generation-a",
                        3,
                        {"ratings": {"0": "up"}},
                        {"brief": holdout, "concepts": [self.concept] * 3},
                    )
                ],
            )
        )

    def test_corrupt_base_is_rejected(self):
        corrupt = {**self.base, "train.jsonl": self.base["train.jsonl"] + b"{}\n"}
        with self.assertRaises(ValidationError):
            build_version(self.uri, corrupt, [])
