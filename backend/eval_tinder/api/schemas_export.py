"""Request/response schemas for bulk grading and exports. No framework types leak through here."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class GradingJobCreate(BaseModel):
    grader_id: str = Field(min_length=1)
    partition: Literal["TRAIN", "DEV", "AUDIT_RESERVE"] | None = None  # AUDIT_RESERVE is refused (400)
    trace_ids: list[str] | None = None
    idempotency_key: str = Field(min_length=1)


class PredictionPage(BaseModel):
    items: list[dict[str, Any]]
    total: int
    limit: int
    offset: int
    grader_id: str
    note: str = (
        "Every item is a MACHINE prediction of one grader version. It is not a human label and carries "
        "no per-case confidence figure."
    )


class ExportCreate(BaseModel):
    kind: Literal["FULL", "GRADER"] = "FULL"
    grader_id: str | None = None
    idempotency_key: str = Field(min_length=1)


class ExportOut(BaseModel):
    id: str
    project_id: str
    kind: str
    state: str
    job_id: str | None
    manifest: dict[str, Any]
    download_url: str | None
    error: str | None = None
    created_at: datetime
