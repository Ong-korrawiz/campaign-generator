"""Exercise GCS preconditions and immutable feedback snapshot publication."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from google.api_core.exceptions import NotFound, PreconditionFailed

from campaign_generator.api.feedback import GCSFeedbackStore
from campaign_generator.dataset_v3.feedback import prepare_feedback_version

ROOT = Path(__file__).resolve().parents[1]


class FakeBlob:
    def __init__(self, bucket, name, generation=None):
        self.bucket = bucket
        self.name = name
        self.generation = generation

    def reload(self):
        versions = self.bucket.objects.get(self.name)
        if not versions:
            raise NotFound(self.name)
        self.generation = len(versions)

    def download_as_bytes(self):
        versions = self.bucket.objects.get(self.name)
        if not versions:
            raise NotFound(self.name)
        return versions[self.generation - 1 if self.generation else -1]

    def upload_from_string(self, content, *, if_generation_match, content_type=None):
        versions = self.bucket.objects.setdefault(self.name, [])
        if if_generation_match != len(versions):
            raise PreconditionFailed(self.name)
        versions.append(content.encode() if isinstance(content, str) else content)
        self.generation = len(versions)


class FakeBucket:
    def __init__(self):
        self.objects = {}

    def blob(self, name, generation=None):
        return FakeBlob(self, name, generation)

    def list_blobs(self, *, prefix):
        return [
            SimpleNamespace(name=name, generation=len(versions))
            for name, versions in sorted(self.objects.items())
            if name.startswith(prefix)
        ]


class FeedbackGCSTests(TestCase):
    def setUp(self):
        self.feedback = FakeBucket()
        self.dataset = FakeBucket()

    def test_rating_upsert_preserves_other_indices_and_noop_generation(self):
        store = GCSFeedbackStore.__new__(GCSFeedbackStore)
        store.bucket = self.feedback
        store.save_generation("generation-a", {"brief": {}, "concepts": []})
        self.assertEqual(store.submit("generation-a", {"0": "up"}), {"0": "up"})
        self.assertEqual(store.submit("generation-a", {"0": "up"}), {"0": "up"})
        self.assertEqual(len(self.feedback.objects["feedback/votes/generation-a.json"]), 1)
        self.assertEqual(store.submit("generation-a", {"1": "down"}), {"0": "up", "1": "down"})
        self.assertIsNone(store.submit("missing", {"0": "up"}))

    def test_snapshot_publishes_immutable_files_to_dataset_bucket(self):
        for name in ("manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl"):
            self.dataset.objects[f"versions/base/{name}"] = [(ROOT / "dataset" / name).read_bytes()]
        original = json.loads((ROOT / "dataset" / "train.jsonl").read_bytes().splitlines()[0])
        generation = {
            "brief": {**original["input"], "brand": "Another Feedback Brand", "product": "New product"},
            "concepts": [original["output"]] * 3,
        }
        store = GCSFeedbackStore.__new__(GCSFeedbackStore)
        store.bucket = self.feedback
        store.save_generation("generation-a", generation)
        store.submit("generation-a", {"0": "up"})
        client = SimpleNamespace(
            bucket=lambda name: {"example-dataset": self.dataset, "example-feedback": self.feedback}[name]
        )
        with patch("google.cloud.storage.Client", return_value=client):
            first = prepare_feedback_version(
                "gs://example-dataset/versions/base", "example-feedback", "example"
            )
            second = prepare_feedback_version(
                "gs://example-dataset/versions/base", "example-feedback", "example"
            )
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("gs://example-dataset/versions/"))
        self.assertEqual(
            len(self.dataset.objects[first.removeprefix("gs://example-dataset/") + "/manifest.json"]), 1
        )
        self.assertFalse(any(name.startswith("versions/") for name in self.feedback.objects))
