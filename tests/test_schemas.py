"""Offline checks for the shared campaign contracts."""

import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from campaign_generator.prompts.campaign import render_user_prompt
from campaign_generator.schemas import (
    ApiCampaignBrief,
    CampaignDirectionV2,
    validate_brief,
    validate_output_v2,
)


class SchemaTests(unittest.TestCase):
    """Exercise API and campaign output validation without model inference."""

    def test_budget_contract_and_deterministic_prompt(self):
        brief = {
            "industry": "retail",
            "target_audience": "students",
            "objective": "awareness",
            "channels": ["Email"],
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
            "campaign_description": "Explain the supplied proof. Use Email to show it.",
            "audience_insight": "Proof builds trust",
            "key_message": "Try a demo",
            "channel_plan": [{"channel": "Email", "role": ""}],
            "asset_plan": [{"asset": "Demo email", "channel": ""}],
        }
        self.assertIn("invalid_channel_plan_item", validate_output_v2(output))
        self.assertIn("invalid_asset_plan_item", validate_output_v2(output))
        with self.assertRaises(ValueError):
            CampaignDirectionV2.model_validate(output)

    def test_api_budget_is_optional_but_all_budget_fields_are_atomic(self):
        base = {
            "industry": "retail",
            "target_audience": "students",
            "objective": "awareness",
            "channels": ["Email"],
        }
        self.assertEqual(validate_brief(base, api_request=True), [])
        self.assertIn("incomplete_budget", validate_brief({**base, "budget_min": 10}, api_request=True))
        self.assertEqual(
            validate_brief({**base, "budget_min": 10, "budget_max": 20, "currency": "THB"}, api_request=True),
            [],
        )

    def test_v3_requires_description(self):
        output = {
            "campaign_direction": "A clear first step",
            "audience_insight": "Teams may need a simple way to begin.",
            "key_message": "Start with one action.",
            "channel_plan": [{"channel": "Email", "role": "Explain the offer"}],
            "asset_plan": [{"asset": "An introductory email", "channel": "Email"}],
        }
        with self.assertRaises(ValidationError):
            CampaignDirectionV2.model_validate(output)
        self.assertEqual(
            CampaignDirectionV2.model_validate(
                {**output, "campaign_description": "A two sentence example. It stays grounded."}
            ).campaign_direction,
            "A clear first step",
        )

    def test_invalid_brief_fields(self):
        errors = validate_brief(
            {"industry": " ", "target_audience": "buyers", "objective": "sales", "channels": [" "]}
        )
        self.assertIn("missing_industry", errors)
        self.assertIn("invalid_channels", errors)


if __name__ == "__main__":
    unittest.main()
