"""Offline checks for the deterministic v3 brief design and v1 compatibility."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from campaign_generator.dataset_v3.pipeline import assign_new_splits, synthetic_briefs
from campaign_generator.evaluation.llm_judge import automatic_checks
from campaign_generator.schemas import CampaignDirectionV2


class DatasetV3Tests(unittest.TestCase):
    def test_committed_source_briefs_preserve_original_splits(self):
        source = Path(__file__).resolve().parents[1] / "dataset" / "source_briefs.jsonl"
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 120)
        self.assertEqual(len({row["id"] for row in rows}), 120)
        self.assertEqual(
            {split: sum(row["split"] == split for row in rows) for split in ("train", "validation", "test")},
            {"train": 96, "validation": 12, "test": 12},
        )

    def test_new_briefs_are_120_unique_balanced_rows(self):
        rows = synthetic_briefs()
        self.assertEqual(len(rows), 120)
        self.assertEqual(len({row["id"] for row in rows}), 120)
        self.assertEqual(len({row["brief"]["brand"] for row in rows}), 120)
        self.assertEqual(
            {row["brief"]["industry"] for row in rows}, {row["brief"]["industry"] for row in rows[::12]}
        )
        self.assertTrue(all(row["brief"]["channels"] and row["brief"]["proof_point"] for row in rows))
        splits = assign_new_splits(rows)
        self.assertEqual(
            {name: list(splits.values()).count(name) for name in ("train", "validation", "test")},
            {"train": 72, "validation": 12, "test": 36},
        )
        self.assertEqual(splits, assign_new_splits(rows))

    def test_description_is_required(self):
        legacy = {
            "campaign_direction": "Clear first pass",
            "audience_insight": "Teams may value clear steps.",
            "key_message": "Start with the next action.",
            "channel_plan": [{"channel": "Email", "role": "Explain"}],
            "asset_plan": [{"asset": "Intro email", "channel": "Email"}],
        }
        with self.assertRaises(ValidationError):
            CampaignDirectionV2.model_validate(legacy)
        CampaignDirectionV2.model_validate(
            {**legacy, "campaign_description": "Show the idea. Explain the action."}
        )

    def test_concept_shape_checks_descriptions_and_count(self):
        concept = {
            "campaign_direction": "A clear first pass",
            "campaign_description": "Show the supplied proof in a simple example. Use the same idea across channels.",
            "proposed_kpis": [
                {"metric": "Sign-ups", "target": "Proposed +10%", "rationale": "Planning estimate."}
            ],
        }
        result = automatic_checks({"concepts": [concept, concept, concept]}, expected_ideas=3)
        self.assertTrue(result["idea_count_valid"])
        self.assertFalse(result["unique_nonempty_names"])
        self.assertTrue(result["campaign_descriptions_2_to_3_sentences"])


if __name__ == "__main__":
    unittest.main()
