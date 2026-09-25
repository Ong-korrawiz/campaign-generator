"""Offline checks for mapping, duplicate handling, and the shared contract."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from campaign_generator.dataset.pipeline import audit_brainstorming, convert_zarn, deduplicate_splits
from campaign_generator.prompts.campaign import render_user_prompt
from campaign_generator.schemas import (
    ApiCampaignBrief,
    CampaignDirection,
    validate_brief,
    validate_output,
    validate_training_record,
)

# Fixed fixture revision keeps source metadata checks deterministic.
REVISION = "a" * 40


def source_row() -> dict:
    """Return a compact Zarn-shaped fixture with one direction and asset."""
    return {
        "example_id": "sample-1",
        "annotation_status": "starter",
        "creative_brief": {
            "brand": "Acme",
            "project_background": "Launch a useful service.",
            "business_goal": "Increase trial signups",
            "audience": "small business owners",
            "channels": ["Email"],
            "non_goals": ["No hype"],
            "must_keep_proof": "Product demo",
        },
        "brand_context": {"vertical": "software", "must_avoid": ["False claims"]},
        "audience_profile": {"primary": "small business owners", "trust_builder": "Show the workflow"},
        "constraints": ["No unsupported metrics"],
        "reference_brief_package": {"hero_angle": "Make the first action easy"},
        "message_hierarchy": [{"message": "See the workflow in one minute"}],
        "channel_strategy": [{"channel": "Email", "job": "Invite a demo", "format_note": "Short note"}],
        "reference_output": {
            "deliverables": [
                {"priority": 1, "asset": "Demo email", "channel": "Email", "reason": "Show proof"}
            ]
        },
        "reference_variants": [{"label": "asset-order variant"}],
    }


class DataTests(unittest.TestCase):
    """Exercise mapping and shared interface behavior without model inference."""

    def test_zarn_maps_one_brief_to_one_grounded_record(self):
        """Keep the existing one-brief-to-one-record mapping and JSON layout."""
        record, errors = convert_zarn(
            source_row(), revision=REVISION, license_name="apache-2.0", split="train"
        )
        self.assertEqual(errors, [])
        self.assertEqual(validate_training_record(record), [])
        self.assertEqual(record["input"]["industry"], "software")
        self.assertEqual(record["output"]["campaign_direction"], "Make the first action easy")
        self.assertEqual(len(record["output"]["asset_plan"]), 1)
        self.assertNotIn("budget_min", record["input"])
        self.assertNotIn("kpis", record["output"])
        self.assertEqual(json.loads(record["messages"][2]["content"]), record["output"])

    def test_missing_hero_angle_is_rejected(self):
        """Retain the legacy missing-direction rejection code."""
        row = source_row()
        row["reference_brief_package"] = {}
        record, errors = convert_zarn(row, revision=REVISION, license_name="apache-2.0", split="train")
        self.assertIsNone(record)
        self.assertIn("missing_campaign_direction", errors)

    def test_exact_leakage_removal_prefers_test(self):
        """Reserve an identical brief for the held-out split."""
        row, _ = convert_zarn(source_row(), revision=REVISION, license_name="apache-2.0", split="train")
        kept, removed = deduplicate_splits({"train": [row], "validation": [row], "test": [row]})
        self.assertEqual([len(kept[s]) for s in ("train", "validation", "test")], [0, 0, 1])
        self.assertEqual(removed, {"train": 1, "validation": 1, "test": 0})

    def test_budget_contract_and_deterministic_prompt(self):
        """Validate API budgets and keep prompts stable across key order."""
        brief = {
            "industry": "retail",
            "target_audience": "students",
            "objective": "awareness",
            "budget_min": 100,
            "budget_max": 500,
            "currency": "THB",
        }
        self.assertEqual(validate_brief(brief, api_request=True), [])
        self.assertEqual(ApiCampaignBrief.model_validate(brief).currency, "THB")
        self.assertEqual(render_user_prompt(brief), render_user_prompt(dict(reversed(list(brief.items())))))
        brief["budget_max"] = 10
        self.assertIn("invalid_budget_range", validate_brief(brief, api_request=True))

    def test_brainstorming_duplicate_audit(self):
        """Exclude repeated marketing prompts below the configured diversity gate."""
        row = {
            "metadata": {"domain": "marketing_campaigns"},
            "conversations": [
                {"from": "human", "value": "Situation: launch a bakery"},
                {"from": "gpt", "value": "1. Local tasting event"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text("\n".join(json.dumps(row) for _ in range(3)) + "\n", encoding="utf-8")
            audit = audit_brainstorming(path, minimum_unique=2)
        self.assertEqual(audit["marketing_rows"], 3)
        self.assertEqual(audit["unique_marketing_prompts_after_simple_filter"], 1)
        self.assertEqual(audit["decision"], "exclude_from_training_insufficient_diversity")

    def test_invalid_nested_plans_keep_error_categories(self):
        """Reject missing fields inside channel and asset entries."""
        output = {
            "campaign_direction": "One clear action",
            "audience_insight": "Proof builds trust",
            "key_message": "Try a demo",
            "channel_plan": [{"channel": "Email", "role": ""}],
            "asset_plan": [{"asset": "Demo email", "channel": ""}],
        }
        self.assertIn("invalid_channel_plan_item", validate_output(output))
        self.assertIn("invalid_asset_plan_item", validate_output(output))
        with self.assertRaises(ValueError):
            CampaignDirection.model_validate(output)

    def test_invalid_brief_fields(self):
        """Reject blank core fields and malformed list entries."""
        errors = validate_brief(
            {"industry": " ", "target_audience": "buyers", "objective": "sales", "channels": [" "]}
        )
        self.assertIn("missing_industry", errors)
        self.assertIn("invalid_channels", errors)


if __name__ == "__main__":
    unittest.main()
