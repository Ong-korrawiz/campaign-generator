"""The shared campaign prompt used by training examples and evaluation."""

from __future__ import annotations

import json
from typing import Any

# A single-concept instruction keeps the SFT target aligned with each source row.
SYSTEM_PROMPT = (
    "You are a strategic campaign planner. Produce exactly one campaign direction "
    "and an actionable asset plan as a JSON object. Lead with the business goal and "
    "audience, keep the key message consistent across channels, and use only proof "
    "provided in the brief. Avoid unsupported outcome claims. Do not invent budget "
    "figures or numeric KPI targets. Required keys: campaign_direction, "
    "audience_insight, key_message, channel_plan, asset_plan."
)


def render_user_prompt(brief: dict[str, Any]) -> str:
    """Render a brief with stable JSON ordering for training and evaluation."""
    return "Create one campaign direction for this brief. Return JSON only.\n" + json.dumps(
        brief, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def make_messages(brief: dict[str, Any], output: dict[str, Any]) -> list[dict[str, str]]:
    """Build the three chat messages without changing the existing JSONL format."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(brief)},
        {"role": "assistant", "content": json.dumps(output, ensure_ascii=False, sort_keys=True)},
    ]
