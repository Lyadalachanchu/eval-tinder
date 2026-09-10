"""SQLAlchemy models for every persisted entity.

Design notes
- Every entity is project-scoped. Immutable snapshots/versions never change after creation.
- Human judgments are append-only; corrections supersede, they never overwrite.
- Machine predictions (GradingRun) live in a separate table from HumanJudgment.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from eval_tinder.db.base import Base
from eval_tinder.ids import new_id, utcnow

JSONType = JSON().with_variant(JSON(none_as_null=True), "postgresql")


def _pk() -> Mapped[str]:
    return mapped_column(String(36), primary_key=True, default=new_id)


def _ts() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = _pk()
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    policy_epoch: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    policy_notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    configuration: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    partition_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    active_shadow_grader_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    automation_policy_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = _ts()


class TraceSnapshot(Base):
    __tablename__ = "trace_snapshots"
    __table_args__ = (
        UniqueConstraint("project_id", "external_id", "revision", name="uq_trace_external_revision"),
        Index("ix_trace_project_group", "project_id", "group_id"),
        Index("ix_trace_project_hash", "project_id", "content_hash"),
    )

    id: Mapped[str] = _pk()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    external_id: Mapped[str] = mapped_column(String(300), nullable=False)
    group_id: Mapped[str] = mapped_column(String(300), nullable=False)
    revision: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    is_latest: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    input: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[Any] = mapped_column(JSONType, nullable=True)
    tool_calls: Mapped[Any] = mapped_column(JSONType, nullable=True)
    output: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONType, default=dict, nullable=False)
    source_type: Mapped[str] = mapped_column(String(20), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    import_batch_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = _ts()


class PartitionAssignment(Base):
    __tablename__ = "partition_assignments"
    __table_args__ = (UniqueConstraint("project_id", "group_id", name="uq_partition_group"),)

    id: Mapped[str] = _pk()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    group_id: Mapped[str] = mapped_column(String(300), nullable=False)
    partition: Mapped[str] = mapped_column(String(20), nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    assignment_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    exposure_status: Mapped[str] = mapped_column(String(20), default="UNTOUCHED", nullable=False)
    created_at: Mapped[datetime] = _ts()


class ExposureEvent(Base):
    """Append-only history of how a group was exposed (review, probe, audit seal, ...)."""

    __tablename__ = "exposure_events"
    __table_args__ = (Index("ix_exposure_project_group", "project_id", "group_id"),)

    id: Mapped[str] = _pk()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    group_id: Mapped[str] = mapped_column(String(300), nullable=False)
    kind: Mapped[str] = mapped_column(String(30), nullable=False)
    reference_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = _ts()


class ImportBatch(Base):
    __tablename__ = "import_batches"

    id: Mapped[str] = _pk()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    filename: Mapped[str] = mapped_column(String(300), default="upload.jsonl", nullable=False)
    stored_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    state: Mapped[str] = mapped_column(String(20), default="QUEUED", nullable=False)
    counts: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    line_errors: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    job_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = _ts()


class ReviewRequest(Base):
    __tablename__ = "review_requests"
    __table_args__ = (Index("ix_review_project_state", "project_id", "state"),)

    id: Mapped[str] = _pk()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    trace_id: Mapped[str] = mapped_column(ForeignKey("trace_snapshots.id"), nullable=False)
    purpose: Mapped[str] = mapped_column(String(10), nullable=False)
    selection_round_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    audit_run_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    batch_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    selection_category: Mapped[str] = mapped_column(String(20), nullable=False)
    selection_reason: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    expected_reading_length: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    state: Mapped[str] = mapped_column(String(20), default="OPEN", nullable=False)
    lease_owner: Mapped[str | None] = mapped_column(String(100), nullable=True)
    lease_expiry: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    judgment_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = _ts()

    trace: Mapped[TraceSnapshot] = relationship(lazy="joined")


class HumanJudgment(Base):
    __tablename__ = "human_judgments"
    __table_args__ = (
        UniqueConstraint("project_id", "idempotency_key", name="uq_judgment_idempotency"),
        Index("ix_judgment_trace_epoch", "trace_id", "policy_epoch"),
    )

    id: Mapped[str] = _pk()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    trace_id: Mapped[str] = mapped_column(ForeignKey("trace_snapshots.id"), nullable=False)
    review_request_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    purpose: Mapped[str] = mapped_column(String(10), nullable=False)
    policy_epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    verdict: Mapped[str] = mapped_column(String(20), nullable=False)
    explanation: Mapped[str] = mapped_column(Text, default="", nullable=False)
    cannot_judge_reason: Mapped[str | None] = mapped_column(String(30), nullable=True)
    reviewer_id: Mapped[str] = mapped_column(String(100), nullable=False)
    shown_context_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    active_review_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    supersedes_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    superseded_by_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    idempotency_key: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = _ts()


class DatasetSnapshot(Base):
    __tablename__ = "dataset_snapshots"

    id: Mapped[str] = _pk()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    partition: Mapped[str] = mapped_column(String(20), nullable=False)
    policy_epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    ordered_trace_ids: Mapped[list[str]] = mapped_column(JSONType, nullable=False)
    ordered_judgment_ids: Mapped[list[str]] = mapped_column(JSONType, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = _ts()


class GraderVersion(Base):
    __tablename__ = "grader_versions"
    __table_args__ = (Index("ix_grader_project_manifest", "project_id", "manifest_hash"),)

    id: Mapped[str] = _pk()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    label: Mapped[str] = mapped_column(String(200), default="", nullable=False)
    origin: Mapped[str] = mapped_column(String(20), nullable=False)
    parent_ids: Mapped[list[str]] = mapped_column(JSONType, default=list, nullable=False)
    optimization_run_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    candidate_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    instruction_text: Mapped[str] = mapped_column(Text, nullable=False)
    immutable_policy_context: Mapped[str] = mapped_column(Text, default="", nullable=False)
    model_config_: Mapped[dict[str, Any]] = mapped_column("model_config", JSONType, nullable=False)
    renderer_version: Mapped[str] = mapped_column(String(20), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(20), nullable=False)
    policy_epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = _ts()


class OptimizationRun(Base):
    __tablename__ = "optimization_runs"

    id: Mapped[str] = _pk()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    seed_grader_id: Mapped[str] = mapped_column(ForeignKey("grader_versions.id"), nullable=False)
    seed_choice: Mapped[str] = mapped_column(String(200), default="", nullable=False)
    train_snapshot_id: Mapped[str] = mapped_column(ForeignKey("dataset_snapshots.id"), nullable=False)
    dev_snapshot_id: Mapped[str] = mapped_column(ForeignKey("dataset_snapshots.id"), nullable=False)
    policy_epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    metric_version: Mapped[str] = mapped_column(String(40), nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    artifact_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    state: Mapped[str] = mapped_column(String(20), default="QUEUED", nullable=False)
    budgets: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    usage: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    result_summary: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    job_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = _ts()
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class CandidateEvaluation(Base):
    """A grader's per-case results on one frozen DEV snapshot. Missing scores are unknown, not zero."""

    __tablename__ = "candidate_evaluations"
    __table_args__ = (UniqueConstraint("grader_id", "dev_snapshot_id", name="uq_candidate_eval"),)

    id: Mapped[str] = _pk()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    run_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    grader_id: Mapped[str] = mapped_column(ForeignKey("grader_versions.id"), nullable=False)
    dev_snapshot_id: Mapped[str] = mapped_column(ForeignKey("dataset_snapshots.id"), nullable=False)
    per_case_scores: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    verdicts: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    aggregate_metrics: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    complete: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    source: Mapped[str] = mapped_column(String(30), default="GEPA", nullable=False)
    created_at: Mapped[datetime] = _ts()


class GradingRun(Base):
    __tablename__ = "grading_runs"
    __table_args__ = (
        Index("ix_grading_grader_trace", "grader_id", "trace_id"),
        Index("ix_grading_cache_key", "cache_key"),
        Index("ix_grading_job", "job_id"),
    )

    id: Mapped[str] = _pk()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    grader_id: Mapped[str] = mapped_column(ForeignKey("grader_versions.id"), nullable=False)
    trace_id: Mapped[str] = mapped_column(ForeignKey("trace_snapshots.id"), nullable=False)
    purpose: Mapped[str] = mapped_column(String(20), nullable=False)
    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    cache_key: Mapped[str] = mapped_column(String(64), nullable=False)
    cache_hit_of: Mapped[str | None] = mapped_column(String(36), nullable=True)
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    verdict: Mapped[str] = mapped_column(String(10), nullable=False)  # effective verdict; errors => REVIEW
    evidence: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, default="", nullable=False)
    usage: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempt: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    job_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    audit_run_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = _ts()


class SelectionRound(Base):
    __tablename__ = "selection_rounds"

    id: Mapped[str] = _pk()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    state: Mapped[str] = mapped_column(String(20), default="QUEUED", nullable=False)
    strategy_version: Mapped[str] = mapped_column(String(20), nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    committee_ids: Mapped[list[str]] = mapped_column(JSONType, default=list, nullable=False)
    committee_report: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    probe_ids: Mapped[list[str]] = mapped_column(JSONType, default=list, nullable=False)
    pool_ids: Mapped[list[str]] = mapped_column(JSONType, default=list, nullable=False)
    scores: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    selected_requests: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    batch_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    job_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = _ts()


class AuditRun(Base):
    __tablename__ = "audit_runs"

    id: Mapped[str] = _pk()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    grader_id: Mapped[str] = mapped_column(ForeignKey("grader_versions.id"), nullable=False)
    pipeline_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    policy_epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    population_definition: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    sampling_plan: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    risk_targets: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    locked_sample_ids: Mapped[list[str]] = mapped_column(JSONType, default=list, nullable=False)
    state: Mapped[str] = mapped_column(String(20), default="LOCKED", nullable=False)
    report: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    report_version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    report_history: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    correction_history: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    grading_job_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = _ts()
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AutomationPolicy(Base):
    __tablename__ = "automation_policies"

    id: Mapped[str] = _pk()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    pipeline_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    grader_id: Mapped[str] = mapped_column(String(36), nullable=False)
    supported_scope: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    permitted_verdicts: Mapped[list[str]] = mapped_column(JSONType, default=list, nullable=False)
    risk_targets: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    audit_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    state: Mapped[str] = mapped_column(String(20), default="DISABLED", nullable=False)
    gate_result: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    enabled_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    reason: Mapped[str] = mapped_column(Text, default="", nullable=False)
    history: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    created_at: Mapped[datetime] = _ts()
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_job_idempotency"),
        Index("ix_job_state_kind", "state", "kind"),
    )

    id: Mapped[str] = _pk()
    project_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    kind: Mapped[str] = mapped_column(String(30), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    payload_ref: Mapped[str | None] = mapped_column(String(36), nullable=True)
    idempotency_key: Mapped[str] = mapped_column(String(200), nullable=False)
    state: Mapped[str] = mapped_column(String(20), default="QUEUED", nullable=False)
    progress: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    result: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    lease_owner: Mapped[str | None] = mapped_column(String(100), nullable=True)
    lease_expiry: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = _ts()
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ExportBundle(Base):
    __tablename__ = "export_bundles"

    id: Mapped[str] = _pk()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    kind: Mapped[str] = mapped_column(String(30), nullable=False)
    path: Mapped[str] = mapped_column(String(500), nullable=False)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    job_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = _ts()


class UsageRecord(Base):
    """Provider usage ledger: grading and reflection calls counted separately per job/run."""

    __tablename__ = "usage_records"
    __table_args__ = (Index("ix_usage_job", "job_id"),)

    id: Mapped[str] = _pk()
    project_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    job_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # grading | reflection
    model: Mapped[str] = mapped_column(String(200), nullable=False)
    calls: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = _ts()


ALL_MODELS = [
    Project,
    TraceSnapshot,
    PartitionAssignment,
    ExposureEvent,
    ImportBatch,
    ReviewRequest,
    HumanJudgment,
    DatasetSnapshot,
    GraderVersion,
    OptimizationRun,
    CandidateEvaluation,
    GradingRun,
    SelectionRound,
    AuditRun,
    AutomationPolicy,
    Job,
    ExportBundle,
    UsageRecord,
]
