"""String enums shared by the database models, services, and API schemas."""
from __future__ import annotations

from enum import StrEnum


class Partition(StrEnum):
    TRAIN = "TRAIN"
    DEV = "DEV"
    AUDIT_RESERVE = "AUDIT_RESERVE"


class SourceType(StrEnum):
    PRODUCTION = "PRODUCTION"
    SYNTHETIC = "SYNTHETIC"


class ReviewPurpose(StrEnum):
    TRAIN = "TRAIN"
    DEV = "DEV"
    AUDIT = "AUDIT"


class HumanVerdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    CANNOT_JUDGE = "CANNOT_JUDGE"


class CannotJudgeReason(StrEnum):
    MISSING_CONTEXT = "MISSING_CONTEXT"
    AMBIGUOUS_POLICY = "AMBIGUOUS_POLICY"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    OTHER = "OTHER"


class MachineVerdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"


class GradingStatus(StrEnum):
    """Operational status of one grading call. Only OK results are votes."""

    OK = "OK"
    MALFORMED_OUTPUT = "MALFORMED_OUTPUT"
    INVALID_EVIDENCE = "INVALID_EVIDENCE"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    CONTEXT_TOO_LARGE = "CONTEXT_TOO_LARGE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class GradingPurpose(StrEnum):
    OPTIMIZATION = "OPTIMIZATION"
    DEV_EVALUATION = "DEV_EVALUATION"
    PROBE = "PROBE"
    POOL = "POOL"
    BULK = "BULK"
    AUDIT = "AUDIT"
    EXPERIMENT = "EXPERIMENT"


class ReviewRequestState(StrEnum):
    OPEN = "OPEN"
    LEASED = "LEASED"
    JUDGED = "JUDGED"
    SKIPPED = "SKIPPED"
    CANCELLED = "CANCELLED"


class SelectionCategory(StrEnum):
    SEED = "SEED"
    DEV_RANDOM = "DEV_RANDOM"
    DISAGREEMENT = "DISAGREEMENT"
    COVERAGE = "COVERAGE"
    RANDOM = "RANDOM"
    AUDIT = "AUDIT"
    CONTEXT_REPAIR = "CONTEXT_REPAIR"


class ExposureKind(StrEnum):
    """Why a group was exposed. History is append-only."""

    TRAIN_REVIEW = "TRAIN_REVIEW"
    DEV_REVIEW = "DEV_REVIEW"
    PROBE = "PROBE"
    POOL = "POOL"
    BULK_GRADING = "BULK_GRADING"
    OPTIMIZATION = "OPTIMIZATION"
    AUDIT_SEALED = "AUDIT_SEALED"
    AUDIT_RELEASED = "AUDIT_RELEASED"
    AUDIT_REVIEW = "AUDIT_REVIEW"
    QUARANTINED = "QUARANTINED"
    EXPORT = "EXPORT"


class ExposureStatus(StrEnum):
    UNTOUCHED = "UNTOUCHED"
    INSPECTED = "INSPECTED"
    SEALED = "SEALED"
    QUARANTINED = "QUARANTINED"


class GraderOrigin(StrEnum):
    SEED = "SEED"
    GEPA = "GEPA"
    IMPORTED = "IMPORTED"


class OptimizationState(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    NO_IMPROVEMENT = "NO_IMPROVEMENT"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class JobKind(StrEnum):
    IMPORT = "IMPORT"
    OPTIMIZATION = "OPTIMIZATION"
    SELECTION = "SELECTION"
    BULK_GRADING = "BULK_GRADING"
    DEV_EVALUATION = "DEV_EVALUATION"
    AUDIT_GRADING = "AUDIT_GRADING"
    EXPORT = "EXPORT"
    EXPERIMENT = "EXPERIMENT"


class JobState(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class AuditState(StrEnum):
    LOCKED = "LOCKED"
    IN_REVIEW = "IN_REVIEW"
    COMPLETE = "COMPLETE"
    INVALIDATED = "INVALIDATED"
    SPENT = "SPENT"


class AutomationState(StrEnum):
    DISABLED = "DISABLED"
    ENABLED = "ENABLED"
    INVALIDATED = "INVALIDATED"


class SelectionRoundState(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
