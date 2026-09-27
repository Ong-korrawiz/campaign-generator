"""Check the immutable upload contract against the committed v3 dataset."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from campaign_generator.dataset_v3.gcs import _put_immutable, upload_draft, upload_final

ROOT = Path(__file__).resolve().parents[1]


class DatasetUploadTests(TestCase):
    def test_committed_dataset_upload_is_idempotent_and_hash_addressed(self):
        objects: dict[str, bytes] = {}

        def fake_gcloud(action: str, *args: str, allow_missing: bool = False):
            if action == "cat":
                return objects.get(args[0])
            if action == "cp":
                self.assertEqual(args[0], "--if-generation-match=0")
                objects[args[2]] = Path(args[1]).read_bytes()
                return b""
            if action == "ls":
                return None
            raise AssertionError(action)

        with patch("campaign_generator.dataset_v3.gcs._gcloud", side_effect=fake_gcloud):
            first = upload_final(ROOT / "dataset", "campaign-generator-509812")
            first_objects = dict(objects)
            second = upload_final(ROOT / "dataset", "campaign-generator-509812")
        expected_hash = hashlib.sha256((ROOT / "dataset" / "manifest.json").read_bytes()).hexdigest()
        self.assertEqual(first, second)
        self.assertEqual(objects, first_objects)
        self.assertTrue(first.endswith(f"/versions/{expected_hash}"))

    def test_hash_mismatch_stops_upload_before_any_remote_write(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for name in (
                "manifest.json",
                "train.jsonl",
                "validation.jsonl",
                "test.jsonl",
                "quality_report.md",
            ):
                shutil.copyfile(ROOT / "dataset" / name, root / name)
            with (root / "train.jsonl").open("a", encoding="utf-8") as output:
                output.write("{}\n")
            with patch("campaign_generator.dataset_v3.gcs._gcloud") as gcloud:
                with self.assertRaisesRegex(ValueError, "hash differs"):
                    upload_final(root, "campaign-generator-509812")
                gcloud.assert_not_called()

    def test_remote_conflict_is_never_overwritten(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            path.write_text("local", encoding="utf-8")
            with patch("campaign_generator.dataset_v3.gcs._gcloud", return_value=b"remote") as gcloud:
                with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                    _put_immutable(path, "gs://example/manifest.json")
                gcloud.assert_called_once()

    def test_draft_upload_failure_keeps_local_files_and_retry_succeeds(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            draft = root / "drafts.jsonl"
            draft.write_text('{"id":"example"}\n', encoding="utf-8")
            digest = hashlib.sha256(draft.read_bytes()).hexdigest()
            (root / "manifest.json").write_text(json.dumps({"drafts_sha256": digest}), encoding="utf-8")
            objects: dict[str, bytes] = {}
            fail_once = True

            def fake_gcloud(action: str, *args: str, allow_missing: bool = False):
                nonlocal fail_once
                if action == "cat":
                    return objects.get(args[0])
                if action == "cp":
                    if fail_once:
                        fail_once = False
                        raise RuntimeError("temporary upload failure")
                    objects[args[2]] = Path(args[1]).read_bytes()
                    return b""
                raise AssertionError(action)

            with patch("campaign_generator.dataset_v3.gcs._gcloud", side_effect=fake_gcloud):
                with self.assertRaisesRegex(RuntimeError, "temporary upload failure"):
                    upload_draft(root, "campaign-generator-509812")
                self.assertTrue(draft.exists())
                uri = upload_draft(root, "campaign-generator-509812")
                self.assertIn(digest, uri)
                self.assertEqual(objects[f"{uri}/drafts.jsonl"], draft.read_bytes())
