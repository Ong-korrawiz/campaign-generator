"""Pydantic contracts shared by data preparation, evaluation, and the future API."""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

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
    """Campaign brief accepted by the planned API when budget is supplied."""

    budget_min: float = Field(ge=0, allow_inf_nan=False)
    budget_max: float = Field(ge=0, allow_inf_nan=False)
    currency: NonEmptyText

    @model_validator(mode="after")
    def check_budget_range(self) -> ApiCampaignBrief:
        """Reject a maximum budget below the minimum budget."""
        if self.budget_max < self.budget_min:
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


class CampaignDirection(StrictModel):
    """One SFT target; KPI and budget estimates are intentionally absent."""

    campaign_direction: NonEmptyText
    audience_insight: NonEmptyText
    key_message: NonEmptyText
    channel_plan: list[ChannelPlanItem] = Field(min_length=1)
    asset_plan: list[AssetPlanItem] = Field(min_length=1)


class BrainstormIdea(StrictModel):
    """One named campaign idea with a concise creative and execution description."""

    name: NonEmptyText
    description: NonEmptyText


class BrainstormingSet(StrictModel):
    """Brainstorming-style target matching the candidate dataset's ten-idea format."""

    ideas: list[BrainstormIdea] = Field(min_length=10, max_length=10)
    prioritization_notes: NonEmptyText

    @model_validator(mode="after")
    def check_unique_idea_names(self) -> BrainstormingSet:
        """Reject repeated campaign names within one brainstorm."""
        names = [idea.name.casefold() for idea in self.ideas]
        if len(set(names)) != len(names):
            raise ValueError("duplicate_idea_names")
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


class TrainingRecord(StrictModel):
    """Complete normalized example written to the split JSONL files."""

    id: NonEmptyText
    source: SourceInfo
    input: CampaignBrief
    output: CampaignDirection
    messages: list[ChatMessage] = Field(min_length=3, max_length=3)
    transformation: TransformationInfo | None = None

    @model_validator(mode="after")
    def check_message_roles(self) -> TrainingRecord:
        """Preserve the system, user, assistant order required by the trainer."""
        if [message.role for message in self.messages] != ["system", "user", "assistant"]:
            raise ValueError("invalid_messages")
        return self


class BrainstormingTrainingRecord(StrictModel):
    """SFT record using the candidate brainstorming dataset's user/assistant shape."""

    id: NonEmptyText
    source: SourceInfo
    input: CampaignBrief
    output: BrainstormingSet
    messages: list[ChatMessage] = Field(min_length=2, max_length=2)
    transformation: TransformationInfo

    @model_validator(mode="after")
    def check_message_roles(self) -> BrainstormingTrainingRecord:
        """Require a user prompt followed by the assistant's ten-idea response."""
        if [message.role for message in self.messages] != ["user", "assistant"]:
            raise ValueError("invalid_brainstorming_messages")
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
            elif field in ("industry", "target_audience", "objective", "currency") and issue["type"] in (
                "missing",
                "string_too_short",
            ):
                errors.append(f"missing_{field}")
            else:
                errors.append(f"invalid_{field or 'brief'}")
        return list(dict.fromkeys(errors))


def validate_output(output: Any) -> list[str]:
    """Return stable output error codes from the nested Pydantic contract."""
    if not isinstance(output, dict):
        return ["output_not_object"]
    try:
        CampaignDirection.model_validate(output)
        return []
    except ValidationError as exc:
        errors: list[str] = []
        for issue in exc.errors():
            field = issue["loc"][0] if issue["loc"] else ""
            if field in ("channel_plan", "asset_plan"):
                errors.append(f"invalid_{field}_item" if len(issue["loc"]) > 1 else f"missing_{field}")
            elif field in ("campaign_direction", "audience_insight", "key_message") and issue["type"] in (
                "missing",
                "string_too_short",
            ):
                errors.append(f"missing_{field}")
            else:
                errors.append(f"invalid_{field or 'output'}")
        return list(dict.fromkeys(errors))


def validate_training_record(record: Any) -> list[str]:
    """Check a complete JSONL record and preserve legacy error categories."""
    if not isinstance(record, dict):
        return ["record_not_object"]
    errors = validate_brief(record.get("input")) + validate_output(record.get("output"))
    try:
        TrainingRecord.model_validate(record)
    except ValidationError as exc:
        for issue in exc.errors():
            field = issue["loc"][0] if issue["loc"] else ""
            if field == "source":
                errors.append("invalid_source")
            elif field == "messages" or "invalid_messages" in str(issue["msg"]):
                errors.append("invalid_messages")
            elif field == "id":
                errors.append("invalid_id")
            elif field not in ("input", "output"):
                errors.append("invalid_record")
    return list(dict.fromkeys(errors))
