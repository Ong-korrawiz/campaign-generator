"""Contracts for reviewable, first-party campaign dataset drafts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field

from ..schemas import CampaignBrief, CampaignDirection, NonEmptyText, StrictModel


class BriefSeed(StrictModel):
    """A stable identifier and the exact brief available to the student model."""

    id: NonEmptyText
    brief: CampaignBrief


class TeacherResult(StrictModel):
    """A validated model response, including metadata retained in the cache."""

    output: CampaignDirection
    response_id: str | None = None
    generated_at: datetime
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class DraftProvenance(StrictModel):
    """Trace a draft to its source line, prompt, model, and API response."""

    model: NonEmptyText
    prompt_version: NonEmptyText
    prompt_sha256: NonEmptyText
    source_line: int = Field(ge=1)
    response_id: str | None = None
    generated_at: datetime
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_hit: bool


class DraftRecord(StrictModel):
    """A candidate label that must be reviewed before training."""

    id: NonEmptyText
    input: CampaignBrief
    output: CampaignDirection
    review_status: Literal["pending"] = "pending"
    provenance: DraftProvenance


class BriefReader(Protocol):
    """Read and validate all input briefs."""

    def read(self, path: Path) -> list[tuple[int, BriefSeed]]: ...


class CampaignTeacher(Protocol):
    """Generate one concept from one brief."""

    model_id: str
    prompt_version: str
    prompt_sha256: str

    def generate(self, brief: CampaignBrief) -> TeacherResult: ...


class TeacherCache(Protocol):
    """Reuse successful generations across interrupted runs."""

    def get(self, key: str) -> TeacherResult | None: ...

    def put(self, key: str, result: TeacherResult) -> None: ...


class DraftWriter(Protocol):
    """Write one immutable draft run."""

    def write(self, path: Path, rows: list[DraftRecord], manifest: dict[str, Any]) -> dict[str, Any]: ...
