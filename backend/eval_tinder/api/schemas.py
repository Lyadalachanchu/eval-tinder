"""Pydantic request/response schemas. No framework (dspy/sqlalchemy) types leak through here."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    partition_seed: int | None = None
    configuration: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = None


class ProjectOut(BaseModel):
    id: str
    name: str
    description: str
    policy_epoch: int
    policy_notes: str
    configuration: dict[str, Any]
    active_shadow_grader_id: str | None
    automation_policy_id: str | None
    created_at: datetime


class ProjectDashboard(BaseModel):
    project: ProjectOut
    partitions: dict[str, int]
    labels: dict[str, dict[str, int]]
    review_states: dict[str, int]
    readiness: dict[str, Any]
    graders: int
    runs: int
    shadow_grader: dict[str, Any] | None
    automation: dict[str, Any] | None


class ImportOut(BaseModel):
    id: str
    project_id: str
    filename: str
    state: str
    counts: dict[str, Any]
    line_errors: list[dict[str, Any]]
    job_id: str | None
    created_at: datetime


class ReviewBatchCreate(BaseModel):
    purpose: Literal["TRAIN", "DEV"]
    kind: Literal["SEED", "DEV_RANDOM", "RANDOM", "ACTIVE"] = "SEED"
    size: int | None = Field(default=None, ge=1, le=200)
    seed: int | None = None
    idempotency_key: str | None = None


class ReviewRequestOut(BaseModel):
    id: str
    project_id: str
    trace_id: str
    purpose: str
    state: str
    selection_category: str
    selection_reason: dict[str, Any] | None = None  # hidden until judged (TRAIN only)
    expected_reading_length: int
    lease_owner: str | None
    lease_expiry: datetime | None
    judgment_id: str | None
    batch_id: str | None
    created_at: datetime


class TraceView(BaseModel):
    id: str
    external_id: str
    group_id: str
    revision: int
    timestamp: datetime | None
    input: str
    context: Any
    tool_calls: Any
    output: str
    metadata: dict[str, Any]
    source_type: str
    content_hash: str
    partition: str | None = None


class ReviewCase(BaseModel):
    request: ReviewRequestOut
    trace: TraceView
    shown_context_hash: str
    predictions: list[dict[str, Any]] | None = None  # revealed only after a TRAIN judgment


class ClaimRequest(BaseModel):
    lease_seconds: int | None = None


class JudgmentCreate(BaseModel):
    verdict: Literal["PASS", "FAIL", "CANNOT_JUDGE"]
    explanation: str = ""
    cannot_judge_reason: Literal["MISSING_CONTEXT", "AMBIGUOUS_POLICY", "OUT_OF_SCOPE", "OTHER"] | None = None
    shown_context_hash: str
    active_review_ms: int = 0
    idempotency_key: str


class JudgmentOut(BaseModel):
    id: str
    trace_id: str
    review_request_id: str | None
    purpose: str
    policy_epoch: int
    verdict: str
    explanation: str
    cannot_judge_reason: str | None
    reviewer_id: str
    active_review_ms: int
    supersedes_id: str | None
    superseded_by_id: str | None
    created_at: datetime


class CorrectionCreate(BaseModel):
    verdict: Literal["PASS", "FAIL", "CANNOT_JUDGE"]
    explanation: str = ""
    cannot_judge_reason: Literal["MISSING_CONTEXT", "AMBIGUOUS_POLICY", "OUT_OF_SCOPE", "OTHER"] | None = None
    idempotency_key: str


class OptimizationRunCreate(BaseModel):
    max_metric_calls: int | None = Field(default=None, ge=1)
    reflection_minibatch_size: int | None = Field(default=None, ge=1, le=50)
    num_threads: int | None = Field(default=None, ge=1, le=32)
    seed: int = 0
    seed_grader_id: str | None = None
    label: str = ""
    max_provider_calls: int | None = None
    max_total_tokens: int | None = None
    evaluate_all_candidates: bool = False
    idempotency_key: str


class CandidateOut(BaseModel):
    grader_id: str
    candidate_index: int | None
    label: str
    parent_ids: list[str]
    instruction_text: str
    manifest_hash: str
    evaluation: dict[str, Any] | None
    is_seed: bool
    is_member: bool
    diff_from_seed: str


class OptimizationRunOut(BaseModel):
    id: str
    project_id: str
    state: str
    seed_grader_id: str
    seed_choice: str
    train_snapshot_id: str
    dev_snapshot_id: str
    train_size: int
    dev_size: int
    policy_epoch: int
    metric_version: str
    config: dict[str, Any]
    budgets: dict[str, Any]
    usage: dict[str, Any]
    result_summary: dict[str, Any]
    job_id: str | None
    error: str | None
    created_at: datetime
    finished_at: datetime | None
    candidates: list[CandidateOut] = Field(default_factory=list)


class GraderOut(BaseModel):
    id: str
    project_id: str
    label: str
    origin: str
    parent_ids: list[str]
    optimization_run_id: str | None
    candidate_index: int | None
    instruction_text: str
    immutable_policy_context: str
    model_config_: dict[str, Any] = Field(alias="model_config")
    renderer_version: str
    parser_version: str
    policy_epoch: int
    manifest: dict[str, Any]
    manifest_hash: str
    pipeline_hash: str
    created_at: datetime
    diff_from_parent: str | None = None
    evaluations: list[dict[str, Any]] = Field(default_factory=list)
    is_active_shadow: bool = False

    model_config = {"populate_by_name": True, "protected_namespaces": ()}


class ShadowSelect(BaseModel):
    grader_id: str | None  # null clears the shadow grader
    reason: str = Field(min_length=1)


class JobOut(BaseModel):
    id: str
    project_id: str | None
    kind: str
    state: str
    progress: dict[str, Any]
    result: dict[str, Any]
    attempts: int
    error: str | None
    cancel_requested: bool
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class TracePage(BaseModel):
    items: list[dict[str, Any]]
    total: int
    limit: int
    offset: int
