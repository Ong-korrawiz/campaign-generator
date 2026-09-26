"""Offline integration checks for the first-party dataset generator."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from campaign_generator.dataset_v2.contracts import TeacherResult
from campaign_generator.dataset_v2.generate import DatasetGenerator, SqliteTeacherCache
from campaign_generator.dataset_v2.io import JsonlBriefReader, JsonlDraftWriter, sha256_file
from campaign_generator.schemas import CampaignBrief, CampaignDirection


class FakeTeacher:
    model_id = "gpt-6-luna"
    prompt_version = "test-v1"
    prompt_sha256 = "a" * 64

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, brief: CampaignBrief) -> TeacherResult:
        self.calls += 1
        return TeacherResult(
            output=CampaignDirection(
                campaign_direction=f"Make {brief.industry} approachable",
                audience_insight="People want a simple first step",
                key_message="See how it works",
                channel_plan=[{"channel": "Email", "role": "Explain the offer"}],
                asset_plan=[{"asset": "Intro email", "channel": "Email"}],
            ),
            response_id=f"fake-{self.calls}",
            generated_at=datetime.now(timezone.utc),
            input_tokens=10,
            output_tokens=20,
        )


class DatasetV2Tests(unittest.TestCase):
    def test_draft_generation_uses_cache_and_keeps_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "briefs.jsonl"
            input_path.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "id": f"brief-{number}",
                            "brief": {
                                "industry": industry,
                                "target_audience": "local shoppers",
                                "objective": "increase trial visits",
                            },
                        }
                    )
                    for number, industry in enumerate(("food", "beauty"), 1)
                )
                + "\n",
                encoding="utf-8",
            )
            teacher = FakeTeacher()
            cache = SqliteTeacherCache(root / "cache.sqlite3")
            try:
                generator = DatasetGenerator(JsonlBriefReader(), teacher, cache, JsonlDraftWriter())
                first = generator.run(input_path, root / "run-1")
                second = generator.run(input_path, root / "run-2")
            finally:
                cache.close()
            self.assertEqual(teacher.calls, 2)
            self.assertEqual(first["draft_count"], 2)
            self.assertEqual(second["cache_hits"], 2)
            self.assertEqual(second["api_input_tokens"], 0)
            drafts_path = root / "run-1" / "drafts.jsonl"
            self.assertEqual(first["drafts_sha256"], sha256_file(drafts_path))
            records = [json.loads(line) for line in drafts_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["review_status"] for row in records], ["pending", "pending"])
            self.assertEqual(records[0]["provenance"]["source_line"], 1)
            self.assertEqual(records[0]["provenance"]["model"], "gpt-6-luna")

    def test_invalid_input_fails_before_teacher_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "briefs.jsonl"
            row = {
                "id": "brief-1",
                "brief": {
                    "industry": "food",
                    "target_audience": "local shoppers",
                    "objective": "increase trial visits",
                },
            }
            input_path.write_text(
                json.dumps(row) + "\n" + json.dumps({**row, "id": "brief-2"}) + "\n",
                encoding="utf-8",
            )
            teacher = FakeTeacher()
            cache = SqliteTeacherCache(root / "cache.sqlite3")
            try:
                generator = DatasetGenerator(JsonlBriefReader(), teacher, cache, JsonlDraftWriter())
                with self.assertRaisesRegex(ValueError, "duplicate brief"):
                    generator.run(input_path, root / "run")
            finally:
                cache.close()
            self.assertEqual(teacher.calls, 0)
            self.assertFalse((root / "run").exists())


if __name__ == "__main__":
    unittest.main()
