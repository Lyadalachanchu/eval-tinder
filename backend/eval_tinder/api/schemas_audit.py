"""Pydantic schemas for audits and automation policies. No framework types leak through here."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class AuditCreate(BaseModel):
    grader_id: str
    planned_n: int = Field(ge=1, le=100_000, description="Planned sample size, fixed before review")
    seed: int | None = Field(default=None, description="Sampling seed; a system random value when omitted")
    population: dict[str, Any] = Field(
        default_factory=dict,
        description="source_type (PRODUCTION), partition (AUDIT_RESERVE), optional time_window {start,end}, task_types",
    )
    sampling_plan: dict[str, Any] = Field(
        description="unit='group', method='uniform_random', independence_assumption_documented, independence_note",
    )
    risk_targets: dict[str, Any] = Field(
        description=(
            "permitted_verdicts, max_error_rate, min_coverage, confidence (all required; no default is invented); "
            "optional gate_false_pass_rate + joint_allocation='bonferroni', unresolved_automatic_rule"
        ),
    )
    idempotency_key: str = Field(min_length=1, max_length=200)


class AuditOut(BaseModel):
    id: str
    project_id: str
    grader_id: str
    pipeline_hash: str
    policy_epoch: int
    state: str
    population_definition: dict[str, Any]
    sampling_plan: dict[str, Any]
    risk_targets: dict[str, Any]
    planned_n: int
    locked_count: int
    judged_count: int
    unresolved_count: int
    report: dict[str, Any] | None
    report_version: int
    report_history_versions: list[int]
    correction_history: list[dict[str, Any]]
    grading_job_id: str | None
    created_at: datetime
    completed_at: datetime | None
    kind: Literal["AUDIT_EVIDENCE"] = "AUDIT_EVIDENCE"


class AuditSpend(BaseModel):
    reason: str = Field(min_length=1)


class PolicySet(BaseModel):
    audit_id: str
    enable: bool
    reason: str = Field(min_length=1)
    permitted_verdicts: list[Literal["PASS", "FAIL"]] | None = None
    supported_scope: dict[str, Any] | None = None


class PolicyOut(BaseModel):
    id: str | None
    project_id: str
    state: str
    reason: str
    pipeline_hash: str | None
    grader_id: str | None
    audit_id: str | None
    risk_targets: dict[str, Any]
    permitted_verdicts: list[str]
    supported_scope: dict[str, Any]
    gate_result: dict[str, Any]
    enabled_by: str | None
    history: list[dict[str, Any]]
    updated_at: datetime | None
    note: str = ""
