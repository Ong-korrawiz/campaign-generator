"""Pydantic contracts shared by data preparation, evaluation, and the future API."""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, model_validator

# Shared text field type used by all interface models for required strings.
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class StrictModel(BaseModel):
    """Reject undeclared fields so changes to the wire format are deliberate."""

    # Unknown keys signal an unintended interface change or a mapping error.
    model_config = ConfigDict(extra="forbid")


class CampaignBrief(StrictModel):
    """Brief available to SFT examples, which may not contain budget data."""

    industry: NonEmptyText
    target_audience: NonEmptyText
    objective: NonEmptyText
    brand: str = ""
    brand_context: str = ""
    channels: list[NonEmptyText] = Field(default_factory=list)
    constraints: list[NonEmptyText] = Field(default_factory=list)
    proof_point: str = ""
    product: str | None = None
    market: str | None = None
    duration_days: int | None = Field(default=None, ge=1)


class ApiCampaignBrief(CampaignBrief):
    """Campaign brief accepted by the public API with an optional complete budget range."""

    budget_min: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    budget_max: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    currency: NonEmptyText | None = None

    @model_validator(mode="after")
    def check_budget_range(self) -> ApiCampaignBrief:
        """Require budget fields together and reject a reversed range."""
        if not self.channels:
            raise ValueError("channels_required")
        supplied = (self.budget_min is not None, self.budget_max is not None, self.currency is not None)
        if any(supplied) and not all(supplied):
            raise ValueError("budget_fields_must_be_supplied_together")
        if self.budget_min is not None and self.budget_max is not None and self.budget_max < self.budget_min:
            raise ValueError("invalid_budget_range")
        return self


class ChannelPlanItem(StrictModel):
    """Channel-specific job in one campaign direction."""

    channel: NonEmptyText
    role: NonEmptyText
    format_note: str = ""


class AssetPlanItem(StrictModel):
    """Prioritized asset derived from the source reference output."""

    priority: int | None = None
    asset: NonEmptyText
    channel: NonEmptyText
    reason: str = ""


class CampaignDirectionV2(StrictModel):
    """Description-aware SFT target; KPI and budget estimates are absent."""

    campaign_direction: NonEmptyText
    campaign_description: NonEmptyText
    audience_insight: NonEmptyText
    key_message: NonEmptyText
    channel_plan: list[ChannelPlanItem] = Field(min_length=1)
    asset_plan: list[AssetPlanItem] = Field(min_length=1)


class ProposedKPI(StrictModel):
    """Proposed, unverified KPI target for one concept."""

    metric: NonEmptyText
    target: NonEmptyText
    rationale: NonEmptyText


class BudgetAllocationItem(StrictModel):
    """Proposed channel allocation within one concept's optional budget."""

    channel: NonEmptyText
    amount: Decimal = Field(ge=0, allow_inf_nan=False)


class CampaignConcept(CampaignDirectionV2):
    """One public API concept with proposals separate from the training label."""

    proposed_kpis: list[ProposedKPI] = Field(min_length=1)
    budget_allocation: list[BudgetAllocationItem] = Field(default_factory=list)


class CampaignGenerationResponse(StrictModel):
    """Public API response containing exactly three distinct concepts."""

    generation_id: UUID | None = None
    concepts: list[CampaignConcept] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def check_concepts(self) -> CampaignGenerationResponse:
        names = [concept.campaign_direction.casefold().strip() for concept in self.concepts]
        if len(set(names)) != 3:
            raise ValueError("duplicate_campaign_directions")
        return self


class SourceInfo(StrictModel):
    """Trace one transformed example to a pinned dataset row."""

    dataset: NonEmptyText
    revision: NonEmptyText
    split: Literal["train", "validation", "test"]
    source_id: NonEmptyText
    license: str = ""
    annotation_status: str = ""


class ChatMessage(StrictModel):
    """One system, user, or assistant message in an SFT record."""

    role: Literal["system", "user", "assistant"]
    content: NonEmptyText


class TransformationInfo(StrictModel):
    """Record how a model-generated training target was derived."""

    method: Literal["openai_responses_structured_output"]
    model: NonEmptyText
    prompt_version: NonEmptyText
    prompt_sha256: NonEmptyText


class TrainingRecordV2(StrictModel):
    """Description-aware SFT record used by dataset v3."""

    id: NonEmptyText
    source: SourceInfo
    input: CampaignBrief
    output: CampaignDirectionV2
    messages: list[ChatMessage] = Field(min_length=3, max_length=3)
    transformation: TransformationInfo | None = None

    @model_validator(mode="after")
    def check_message_roles(self) -> TrainingRecordV2:
        if [message.role for message in self.messages] != ["system", "user", "assistant"]:
            raise ValueError("invalid_messages")
        return self


def clean_text(value: Any) -> str:
    """Collapse whitespace in source text without adding information."""
    return re.sub(r"\s+", " ", value).strip() if isinstance(value, str) else ""


def validate_brief(brief: Any, *, api_request: bool = False) -> list[str]:
    """Return stable error codes while validating with Pydantic models."""
    if not isinstance(brief, dict):
        return ["brief_not_object"]
    model = ApiCampaignBrief if api_request else CampaignBrief
    try:
        model.model_validate(brief)
        return []
    except ValidationError as exc:
        errors: list[str] = []
        for issue in exc.errors():
            field = issue["loc"][0] if issue["loc"] else ""
            if "invalid_budget_range" in str(issue["msg"]):
                errors.append("invalid_budget_range")
            elif "budget_fields_must_be_supplied_together" in str(issue["msg"]):
                errors.append("incomplete_budget")
            elif "channels_required" in str(issue["msg"]):
                errors.append("missing_channels")
            elif field in ("industry", "target_audience", "objective", "currency") and issue["type"] in (
                "missing",
                "string_too_short",
            ):
                errors.append(f"missing_{field}")
            else:
                errors.append(f"invalid_{field or 'brief'}")
        return list(dict.fromkeys(errors))


def validate_output_v2(output: Any) -> list[str]:
    """Validate the version 2 campaign direction, including its description."""
    if not isinstance(output, dict):
        return ["output_not_object"]
    try:
        CampaignDirectionV2.model_validate(output)
        return []
    except ValidationError as exc:
        errors: list[str] = []
        for issue in exc.errors():
            field = issue["loc"][0] if issue["loc"] else ""
            if field in ("channel_plan", "asset_plan"):
                errors.append(f"invalid_{field}_item" if len(issue["loc"]) > 1 else f"missing_{field}")
            elif field in ("campaign_direction", "campaign_description", "audience_insight", "key_message"):
                errors.append(f"missing_{field}")
            else:
                errors.append(f"invalid_{field or 'output'}")
        return list(dict.fromkeys(errors))
