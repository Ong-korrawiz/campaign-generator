"""Offline checks for the shared campaign contracts."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from campaign_generator.prompts.campaign import render_user_prompt
from campaign_generator.schemas import (
    ApiCampaignBrief,
    CampaignDirection,
    validate_brief,
    validate_output,
)


class SchemaTests(unittest.TestCase):
    """Exercise API and campaign output validation without model inference."""

    def test_budget_contract_and_deterministic_prompt(self):
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

    def test_invalid_nested_plans_keep_error_categories(self):
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
        errors = validate_brief(
            {"industry": " ", "target_audience": "buyers", "objective": "sales", "channels": [" "]}
        )
        self.assertIn("missing_industry", errors)
        self.assertIn("invalid_channels", errors)


if __name__ == "__main__":
    unittest.main()
