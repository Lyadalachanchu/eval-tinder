"""JSONL import: validation, idempotency, revisions, duplicate detection, partition assignment.

Rules
- input, output, external_id are required; group_id defaults to external_id.
- An unchanged record (same external_id, same content hash) is skipped (idempotent).
- A changed record for an existing external_id becomes a new revision and keeps
  the original group (group changes are refused).
- An exact content duplicate under a new external_id is merged into the existing
  group so duplicates never straddle partitions.
- A revision cannot be merged away (it keeps its group), so a revision whose new
  content exactly duplicates a trace in another group is flagged; when the two
  groups sit in different partitions both are quarantined.
- Partition assignment is seeded per project and never rearranged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from eval_tinder.config import Settings, get_settings
from eval_tinder.db.enums import ExposureStatus, JobKind, SourceType
from eval_tinder.db.models import ImportBatch, Job, PartitionAssignment, Project, TraceSnapshot
from eval_tinder.domain.partitions import assign_partition, small_import_warning, validate_split
from eval_tinder.ids import hash_value
from eval_tinder.services import jobs as job_service
from eval_tinder.services.projects import project_config
from eval_tinder.services.review import quarantine_group

MAX_TEXT_CHARS = 200_000


@dataclass
class ParsedRecord:
    line: int
    external_id: str
    group_id: str
    timestamp: datetime | None
    input: str
    context: Any
    tool_calls: Any
    output: str
    metadata: dict[str, Any]
    source_type: str

    def content_hash(self) -> str:
        return hash_value(
            {
                "input": self.input,
                "context": self.context,
                "tool_calls": self.tool_calls,
                "output": self.output,
                "metadata": self.metadata,
                "source_type": self.source_type,
            }
        )


@dataclass
class ParseResult:
    records: list[ParsedRecord] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)


def _parse_timestamp(value: Any, line: int) -> datetime | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO 8601 string")
    text = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as e:
        raise ValueError(f"timestamp is not ISO 8601: {value!r}") from e
    if dt.tzinfo is None:
        from datetime import timezone

        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_jsonl(data: bytes | str, *, max_bytes: int | None = None) -> ParseResult:
    settings = get_settings()
    max_bytes = max_bytes or settings.max_upload_bytes
    raw = data.encode("utf-8") if isinstance(data, str) else data
    result = ParseResult()
    if len(raw) > max_bytes:
        result.errors.append({"line": 0, "error": f"upload exceeds {max_bytes} bytes"})
        return result
    text = raw.decode("utf-8", errors="replace")
    seen_ids: set[str] = set()
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            result.errors.append({"line": line_no, "error": f"invalid JSON: {e.msg}"})
            continue
        if not isinstance(obj, dict):
            result.errors.append({"line": line_no, "error": "record must be a JSON object"})
            continue
        try:
            rec = _validate_record(obj, line_no)
        except ValueError as e:
            result.errors.append({"line": line_no, "error": str(e)})
            continue
        if rec.external_id in seen_ids:
            result.errors.append(
                {"line": line_no, "error": f"duplicate external_id {rec.external_id!r} within upload"}
            )
            continue
        seen_ids.add(rec.external_id)
        result.records.append(rec)
    return result


def _validate_record(obj: dict[str, Any], line: int) -> ParsedRecord:
    for key in ("external_id", "input", "output"):
        if key not in obj or obj[key] is None:
            raise ValueError(f"missing required field {key!r}")
    external_id = str(obj["external_id"]).strip()
    if not external_id or len(external_id) > 300:
        raise ValueError("external_id must be 1-300 characters")
    for key in ("input", "output"):
        if not isinstance(obj[key], str):
            raise ValueError(f"{key} must be a string")
        if len(obj[key]) > MAX_TEXT_CHARS:
            raise ValueError(f"{key} exceeds {MAX_TEXT_CHARS} characters")
    group_id = str(obj.get("group_id") or external_id).strip()
    if len(group_id) > 300:
        raise ValueError("group_id must be at most 300 characters")
    context = obj.get("context")
    tool_calls = obj.get("tool_calls")
    if tool_calls is not None and not isinstance(tool_calls, list):
        raise ValueError("tool_calls must be a list")
    metadata = obj.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    source_type = str(obj.get("source_type") or SourceType.PRODUCTION).upper()
    if source_type not in {s.value for s in SourceType}:
        raise ValueError(f"source_type must be one of {[s.value for s in SourceType]}")
    return ParsedRecord(
        line=line,
        external_id=external_id,
        group_id=group_id,
        timestamp=_parse_timestamp(obj.get("timestamp"), line),
        input=obj["input"],
        context=context,
        tool_calls=tool_calls,
        output=obj["output"],
        metadata=metadata,
        source_type=source_type,
    )


@dataclass
class ImportCounts:
    inserted: int = 0
    unchanged: int = 0
    revised: int = 0
    merged_duplicates: int = 0
    new_groups: int = 0
    warnings: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "inserted": self.inserted,
            "unchanged": self.unchanged,
            "revised": self.revised,
            "merged_duplicates": self.merged_duplicates,
            "new_groups": self.new_groups,
            "warnings": self.warnings,
        }


def ensure_partition(session: Session, project: Project, group_id: str) -> PartitionAssignment:
    existing = session.scalar(
        select(PartitionAssignment).where(
            PartitionAssignment.project_id == project.id, PartitionAssignment.group_id == group_id
        )
    )
    if existing is not None:
        return existing
    split = validate_split(project_config(project).partition_split)
    assignment = PartitionAssignment(
        project_id=project.id,
        group_id=group_id,
        partition=assign_partition(group_id, project.partition_seed, split),
        seed=project.partition_seed,
        exposure_status=ExposureStatus.UNTOUCHED,
    )
    session.add(assignment)
    session.flush()
    return assignment


def _assignment_for(session: Session, project: Project, group_id: str) -> PartitionAssignment | None:
    return session.scalar(
        select(PartitionAssignment).where(
            PartitionAssignment.project_id == project.id, PartitionAssignment.group_id == group_id
        )
    )


def _flag_revision_duplicate(
    session: Session,
    project: Project,
    rec: ParsedRecord,
    group_id: str,
    content_hash: str,
    counts: ImportCounts,
) -> None:
    """Flag a revision whose new content exactly duplicates a trace that lives in *another* group.

    A fresh record with duplicate content is merged into the existing group (see ``import_records``),
    but a revision keeps its original group, so the duplicate would otherwise appear in two groups
    unnoticed. Like the merge path this looks at every stored revision, not only the latest ones.
    A revert to the trace's own earlier content is not a duplicate (same group). When the two groups
    sit in different partitions the shared evidence straddles them, so both groups are quarantined;
    partitions themselves are never rearranged.
    """
    duplicate = session.scalar(
        select(TraceSnapshot)
        .where(
            TraceSnapshot.project_id == project.id,
            TraceSnapshot.content_hash == content_hash,
            TraceSnapshot.group_id != group_id,
        )
        .order_by(TraceSnapshot.is_latest.desc(), TraceSnapshot.external_id)
    )
    if duplicate is None:
        return
    warning: dict[str, Any] = {
        "line": rec.line,
        "external_id": rec.external_id,
        "warning": (
            f"revision is an exact duplicate of {duplicate.external_id!r} in group {duplicate.group_id!r};"
            f" the revision keeps group {group_id!r}"
        ),
    }
    mine = _assignment_for(session, project, group_id)
    theirs = _assignment_for(session, project, duplicate.group_id)
    if mine is not None and theirs is not None and mine.partition != theirs.partition:
        reason = (
            f"cross-partition duplicate: {rec.external_id!r} ({group_id!r}, {mine.partition}) == "
            f"{duplicate.external_id!r} ({duplicate.group_id!r}, {theirs.partition})"
        )
        quarantine_group(session, project.id, group_id, reason=reason)
        quarantine_group(session, project.id, duplicate.group_id, reason=reason)
        warning["warning"] += f"; both groups quarantined ({mine.partition} vs {theirs.partition})"
        warning["quarantined_groups"] = sorted([group_id, duplicate.group_id])
    counts.warnings.append(warning)


def import_records(
    session: Session, project: Project, records: list[ParsedRecord], *, batch_id: str | None = None
) -> ImportCounts:
    counts = ImportCounts()
    for rec in records:
        content_hash = rec.content_hash()
        latest = session.scalar(
            select(TraceSnapshot).where(
                TraceSnapshot.project_id == project.id,
                TraceSnapshot.external_id == rec.external_id,
                TraceSnapshot.is_latest.is_(True),
            )
        )
        group_id = rec.group_id
        if latest is not None:
            if latest.content_hash == content_hash:
                counts.unchanged += 1
                continue
            if latest.group_id != rec.group_id:
                counts.warnings.append(
                    {"line": rec.line, "external_id": rec.external_id,
                     "warning": f"group change ignored; revision keeps group {latest.group_id!r}"}
                )
            group_id = latest.group_id
            # The revision cannot be merged into another group, so an exact duplicate elsewhere must be
            # flagged here (and quarantined when it crosses partitions) instead of slipping through.
            _flag_revision_duplicate(session, project, rec, group_id, content_hash, counts)
            latest.is_latest = False
            revision = latest.revision + 1
            counts.revised += 1
        else:
            duplicate = session.scalar(
                select(TraceSnapshot).where(
                    TraceSnapshot.project_id == project.id, TraceSnapshot.content_hash == content_hash
                )
            )
            if duplicate is not None and duplicate.group_id != rec.group_id:
                counts.warnings.append(
                    {
                        "line": rec.line,
                        "external_id": rec.external_id,
                        "warning": (
                            f"exact duplicate of {duplicate.external_id!r}; "
                            f"merged into group {duplicate.group_id!r}"
                        ),
                    }
                )
                group_id = duplicate.group_id
                counts.merged_duplicates += 1
            elif duplicate is not None:
                counts.merged_duplicates += 1
            revision = 1
            counts.inserted += 1
        before = session.scalar(
            select(PartitionAssignment.id).where(
                PartitionAssignment.project_id == project.id, PartitionAssignment.group_id == group_id
            )
        )
        ensure_partition(session, project, group_id)
        if before is None:
            counts.new_groups += 1
        session.add(
            TraceSnapshot(
                project_id=project.id,
                external_id=rec.external_id,
                group_id=group_id,
                revision=revision,
                is_latest=True,
                timestamp=rec.timestamp,
                input=rec.input,
                context=rec.context,
                tool_calls=rec.tool_calls,
                output=rec.output,
                metadata_=rec.metadata,
                source_type=rec.source_type,
                content_hash=content_hash,
                import_batch_id=batch_id,
            )
        )
        session.flush()
    total_groups = session.scalar(
        select(PartitionAssignment.id).where(PartitionAssignment.project_id == project.id).limit(1)
    )
    if total_groups is not None:
        n = len(
            list(
                session.scalars(
                    select(PartitionAssignment.group_id).where(PartitionAssignment.project_id == project.id)
                )
            )
        )
        warning = small_import_warning(n, validate_split(project_config(project).partition_split))
        if warning:
            counts.warnings.append({"line": 0, "warning": warning})
    return counts


def store_upload(settings: Settings, batch_id: str, data: bytes) -> Path:
    directory = Path(settings.artifact_path) / "imports"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{batch_id}.jsonl"
    path.write_bytes(data)
    return path


def enqueue_import(
    session: Session, project: Project, data: bytes, *, filename: str, idempotency_key: str,
    settings: Settings | None = None,
) -> tuple[ImportBatch, Job]:
    settings = settings or get_settings()
    existing_job = job_service.find_existing(session, idempotency_key, project_id=project.id, kind=JobKind.IMPORT)
    if existing_job is not None:
        batch = session.get(ImportBatch, existing_job.payload_ref)
        assert batch is not None
        return batch, existing_job
    batch = ImportBatch(project_id=project.id, filename=filename, state="QUEUED")
    session.add(batch)
    session.flush()
    path = store_upload(settings, batch.id, data)
    batch.stored_path = str(path)
    job = job_service.enqueue(
        session, kind=JobKind.IMPORT, payload={"batch_id": batch.id, "project_id": project.id},
        idempotency_key=idempotency_key, project_id=project.id, payload_ref=batch.id,
    )
    batch.job_id = job.id
    session.flush()
    return batch, job


def run_import(session: Session, batch: ImportBatch) -> ImportBatch:
    project = session.get(Project, batch.project_id)
    assert project is not None
    data = Path(batch.stored_path).read_bytes() if batch.stored_path else b""
    parsed = parse_jsonl(data)
    counts = import_records(session, project, parsed.records, batch_id=batch.id)
    batch.line_errors = parsed.errors
    batch.counts = {**counts.as_dict(), "lines_rejected": len(parsed.errors)}
    batch.state = "SUCCEEDED"
    session.flush()
    return batch


def import_jsonl_sync(
    session: Session, project: Project, data: bytes | str, *, filename: str = "inline.jsonl"
) -> ImportBatch:
    """Synchronous import (CLI, tests, demo seeding)."""
    batch = ImportBatch(project_id=project.id, filename=filename, state="RUNNING")
    session.add(batch)
    session.flush()
    parsed = parse_jsonl(data)
    counts = import_records(session, project, parsed.records, batch_id=batch.id)
    batch.line_errors = parsed.errors
    batch.counts = {**counts.as_dict(), "lines_rejected": len(parsed.errors)}
    batch.state = "SUCCEEDED"
    session.flush()
    return batch
