"""Portable exports: grader bundles and versioned project bundles (zip).

What leaves the system
- Grader bundles are JSON manifests (prompt text + configuration + provenance).
  Credentials cannot be part of them by construction (``GraderManifest``
  sanitizes model configuration) and no pickle or other executable artifact is
  ever written.
- Full bundles carry latest TRAIN/DEV traces, every human judgment (including
  superseded ones), every machine prediction tagged MACHINE with its grader
  version and audit/automation status, partition and exposure provenance,
  dataset snapshot hashes, optimizer configuration and dependency versions,
  every grader version, the policy epoch, and eligible audit reports.
- Audit material stays sealed: AUDIT_RESERVE traces, audit judgments, and
  AUDIT-purpose predictions appear only for audits that are COMPLETE or SPENT,
  and then only the locked sample of that audit, inside its own folder.

Before a zip is written its contents are scanned: any ``*.pkl``/``*.bin``
member or any credential-looking text aborts the export.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from eval_tinder.config import Settings, get_settings
from eval_tinder.db.enums import AutomationState, ExposureKind, GradingPurpose, JobKind, Partition, ReviewPurpose
from eval_tinder.db.models import (
    AuditRun,
    CandidateEvaluation,
    DatasetSnapshot,
    ExportBundle,
    GraderVersion,
    GradingRun,
    HumanJudgment,
    Job,
    OptimizationRun,
    PartitionAssignment,
    Project,
    ReviewRequest,
    TraceSnapshot,
)
from eval_tinder.domain.manifest import GraderManifest, pipeline_hash
from eval_tinder.ids import utcnow
from eval_tinder.services import jobs as job_service
from eval_tinder.services.bulk_grading import (
    RELEASED_AUDIT_STATES,
    audit_status_for,
    audit_summaries,
    automation_status_for,
    automation_summary,
    bulk_eligible_traces,
)
from eval_tinder.services.projects import get_grader, list_graders
from eval_tinder.services.review import record_exposure

log = logging.getLogger(__name__)

EXPORT_SCHEMA_VERSION = 1
BUNDLE_KIND = "GRADER_BUNDLE"
EXPORT_KINDS = ("FULL", "GRADER")

FORBIDDEN_MEMBER_SUFFIXES = (".pkl", ".pickle", ".bin", ".pt", ".pth", ".joblib")
# Credential-looking text. The lookbehind keeps ordinary words such as "task-1" or "desk-lamp" from matching.
SECRET_PATTERNS = (
    re.compile(r"api[_-]?key", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bauthorization\b", re.IGNORECASE),
    re.compile(r"\bbearer\s+[A-Za-z0-9._-]{6,}", re.IGNORECASE),
    re.compile(r"(access|auth|secret)[_-]?token", re.IGNORECASE),
)
TEXT_SUFFIXES = (".json", ".jsonl", ".md", ".txt")

MACHINE_NOTE = (
    "MACHINE rows are predictions of a versioned grader. They are not human labels, never become "
    "human labels, and carry no per-case confidence figure. Their audit_status and automation_status "
    "describe the grader version's evidence, not the correctness of any single prediction."
)


class ExportError(ValueError):
    pass


# ---------------------------------------------------------------- helpers


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(_dumps(r) + "\n" for r in rows)


def dependency_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in ("dspy", "gepa", "eval_tinder"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def scan_for_secrets(text: str, *, where: str) -> None:
    """Raise when exported text looks like it carries a credential."""
    for pattern in SECRET_PATTERNS:
        match = pattern.search(text)
        if match:
            raise ExportError(f"refusing to export {where}: credential-like text {match.group(0)[:12]!r}")


def assert_safe_members(entries: dict[str, bytes]) -> None:
    for name, data in entries.items():
        lowered = name.lower()
        if lowered.endswith(FORBIDDEN_MEMBER_SUFFIXES):
            raise ExportError(f"refusing to export executable/serialized artifact {name!r}")
        if lowered.endswith(TEXT_SUFFIXES):
            scan_for_secrets(data.decode("utf-8", errors="replace"), where=name)


# ---------------------------------------------------------------- grader bundle


def grader_bundle(session: Session, grader: GraderVersion) -> dict[str, Any]:
    """A portable, credential-free description of one grader version and its evidence status."""
    project = session.get(Project, grader.project_id)
    if project is None:
        raise ExportError(f"project {grader.project_id} not found for grader {grader.id}")
    manifest = GraderManifest.from_dict(grader.manifest)
    bundle = {
        "kind": BUNDLE_KIND,
        "schema_version": EXPORT_SCHEMA_VERSION,
        "grader_id": grader.id,
        "project_id": grader.project_id,
        "project_description": project.description or "",
        "label": grader.label,
        "origin": grader.origin,
        "parent_ids": list(grader.parent_ids or []),
        "optimization_run_id": grader.optimization_run_id,
        "candidate_index": grader.candidate_index,
        "manifest": manifest.to_dict(),
        "instruction_text": grader.instruction_text,
        "immutable_policy_context": grader.immutable_policy_context,
        "renderer_version": grader.renderer_version,
        "parser_version": grader.parser_version,
        "policy_epoch": grader.policy_epoch,
        "manifest_hash": manifest.manifest_hash,
        "pipeline_hash": pipeline_hash(manifest),
        "dependency_versions": dependency_versions(),
        "audits": audit_summaries(session, grader),
        "audit_status": audit_status_for(session, grader),
        "automation": automation_summary(session, project, grader),
        "automation_status": automation_status_for(session, project, grader),
        "created_at": _iso(grader.created_at),
        "exported_at": _iso(utcnow()),
        "notes": [
            "Reconstruct the grader from instruction_text and manifest only; no serialized program is included.",
            "Provider credentials are never part of a bundle: configure them in the grading environment.",
            MACHINE_NOTE,
            "Audit evidence applies only to the audited pipeline hash, population, and window. Grading a new "
            "file with this bundle produces provisional predictions.",
        ],
    }
    scan_for_secrets(_dumps(bundle), where=f"grader bundle {grader.id}")
    return bundle


# ---------------------------------------------------------------- full bundle rows


def _trace_row(t: TraceSnapshot, partition: str | None) -> dict[str, Any]:
    return {
        "id": t.id,
        "external_id": t.external_id,
        "group_id": t.group_id,
        "revision": t.revision,
        "is_latest": t.is_latest,
        "timestamp": _iso(t.timestamp),
        "input": t.input,
        "context": t.context,
        "tool_calls": t.tool_calls,
        "output": t.output,
        "metadata": t.metadata_ or {},
        "source_type": t.source_type,
        "content_hash": t.content_hash,
        "partition": partition,
        "import_batch_id": t.import_batch_id,
        "created_at": _iso(t.created_at),
    }


def _judgment_row(j: HumanJudgment, t: TraceSnapshot | None) -> dict[str, Any]:
    return {
        "id": j.id,
        "kind": "HUMAN",
        "trace_id": j.trace_id,
        "external_id": t.external_id if t else None,
        "group_id": t.group_id if t else None,
        "trace_revision": t.revision if t else None,
        "review_request_id": j.review_request_id,
        "purpose": j.purpose,
        "policy_epoch": j.policy_epoch,
        "verdict": j.verdict,
        "explanation": j.explanation,
        "cannot_judge_reason": j.cannot_judge_reason,
        "reviewer_id": j.reviewer_id,
        "shown_context_hash": j.shown_context_hash,
        "active_review_ms": j.active_review_ms,
        "supersedes": j.supersedes_id,
        "superseded_by": j.superseded_by_id,
        "is_active": j.superseded_by_id is None,
        "created_at": _iso(j.created_at),
    }


def _grader_meta(session: Session, project: Project, graders: list[GraderVersion]) -> dict[str, dict[str, Any]]:
    """Per-grader hashes and evidence status, computed once per export rather than once per prediction row."""
    meta: dict[str, dict[str, Any]] = {}
    for g in graders:
        meta[g.id] = {
            "manifest_hash": g.manifest_hash,
            "pipeline_hash": pipeline_hash(GraderManifest.from_dict(g.manifest)),
            "audit_status": audit_status_for(session, g),
            "automation_status": automation_status_for(session, project, g),
        }
    return meta


def _prediction_row(r: GradingRun, t: TraceSnapshot | None, meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    info = meta.get(r.grader_id, {})
    audit_status = info.get("audit_status", "UNAUDITED")
    automation_status = info.get("automation_status", AutomationState.DISABLED.value)
    return {
        "id": r.id,
        "kind": "MACHINE",
        "grader_id": r.grader_id,
        "manifest_hash": info.get("manifest_hash"),
        "pipeline_hash": info.get("pipeline_hash"),
        "trace_id": r.trace_id,
        "external_id": t.external_id if t else None,
        "group_id": t.group_id if t else None,
        "purpose": r.purpose,
        "status": r.status,
        "verdict": r.verdict,
        "evidence": r.evidence or [],
        "explanation": r.explanation,
        "error": r.error,
        "prompt_hash": r.prompt_hash,
        "cache_hit_of": r.cache_hit_of,
        "attempt": r.attempt,
        "latency_ms": r.latency_ms,
        "usage": r.usage or {},
        "job_id": r.job_id,
        "audit_run_id": r.audit_run_id,
        "audit_status": audit_status,
        "automation_status": automation_status,
        "provisional": automation_status != AutomationState.ENABLED,
        "created_at": _iso(r.created_at),
    }


def _partition_row(a: PartitionAssignment) -> dict[str, Any]:
    return {
        "group_id": a.group_id,
        "partition": a.partition,
        "seed": a.seed,
        "assignment_version": a.assignment_version,
        "exposure_status": a.exposure_status,
        "created_at": _iso(a.created_at),
    }


def _snapshot_row(s: DatasetSnapshot) -> dict[str, Any]:
    return {
        "id": s.id,
        "partition": s.partition,
        "policy_epoch": s.policy_epoch,
        "content_hash": s.content_hash,
        "sizes": {"traces": len(s.ordered_trace_ids or []), "judgments": len(s.ordered_judgment_ids or [])},
        "ordered_trace_ids": list(s.ordered_trace_ids or []),
        "ordered_judgment_ids": list(s.ordered_judgment_ids or []),
        "created_at": _iso(s.created_at),
    }


def _run_row(run: OptimizationRun, evaluations: list[CandidateEvaluation]) -> dict[str, Any]:
    return {
        "id": run.id,
        "state": run.state,
        "seed_grader_id": run.seed_grader_id,
        "seed_choice": run.seed_choice,
        "train_snapshot_id": run.train_snapshot_id,
        "dev_snapshot_id": run.dev_snapshot_id,
        "policy_epoch": run.policy_epoch,
        "metric_version": run.metric_version,
        "config": run.config or {},
        "budgets": run.budgets or {},
        "usage": run.usage or {},
        "result_summary": run.result_summary or {},
        "job_id": run.job_id,
        "error": run.error,
        "created_at": _iso(run.created_at),
        "finished_at": _iso(run.finished_at),
        "candidate_evaluations": [
            {
                "grader_id": ev.grader_id,
                "dev_snapshot_id": ev.dev_snapshot_id,
                "complete": ev.complete,
                "source": ev.source,
                "aggregate_metrics": ev.aggregate_metrics or {},
                "per_case_scores": ev.per_case_scores or {},
                "verdicts": ev.verdicts or {},
                "kind": "DEVELOPMENT_AGREEMENT",
            }
            for ev in evaluations
        ],
    }


def _audit_report_row(a: AuditRun, *, sample_size: int, unresolved_ids: int) -> dict[str, Any]:
    return {
        "audit_id": a.id,
        "grader_id": a.grader_id,
        "pipeline_hash": a.pipeline_hash,
        "policy_epoch": a.policy_epoch,
        "state": a.state,
        "population_definition": a.population_definition or {},
        "sampling_plan": a.sampling_plan or {},
        "risk_targets": a.risk_targets or {},
        "locked_sample_ids": list(a.locked_sample_ids or []),
        "planned_sample_size": len(a.locked_sample_ids or []),
        "exported_sample_size": sample_size,
        "unresolved_sample_ids": unresolved_ids,
        "report": a.report,
        "report_version": a.report_version,
        "report_history": a.report_history or [],
        "correction_history": a.correction_history or [],
        "grading_job_id": a.grading_job_id,
        "created_at": _iso(a.created_at),
        "completed_at": _iso(a.completed_at),
        "note": "This report covers only the audit's declared population and window for the exact pipeline hash.",
    }


README_TEMPLATE = """# eval-tinder export bundle

Kind: {kind}
Project: {project_id}
Policy epoch: {policy_epoch}
Schema version: {schema_version}
Exported at: {exported_at}

## Files

{files}

## Reading this bundle

- `kind: "HUMAN"` rows are expert judgments. `kind: "MACHINE"` rows are grader predictions.
  {machine_note}
- Every machine prediction names its `grader_id`, `manifest_hash`, `status`, `verdict`, `purpose`,
  `cache_hit_of` (provenance of cached results), `audit_status`, and `automation_status`.
- Human judgments are append-only. Superseded rows stay in the file with `supersedes` /
  `superseded_by` links; `is_active` marks the judgment currently in force for its policy epoch.
- Audit material is sealed until an audit is COMPLETE or SPENT. Only the locked sample of such an
  audit is included, under `audit_samples/<audit_id>/`. The remainder of the AUDIT_RESERVE partition
  is withheld (withheld reserve groups: {withheld_groups}; withheld in-progress/invalidated audits: {withheld_audits}).
- Grader bundles contain prompt text and configuration only. No serialized program, pickle, or
  provider credential is included; configure credentials in the environment that grades.
- Development agreement (`optimization_runs.json`) is a result on a frozen DEV snapshot, not evidence
  of production accuracy. Audit reports are the only independent evidence in this bundle.
"""


def _file_descriptions(kind: str, audit_ids: list[str], grader_ids: list[str]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    if kind == "FULL":
        items += [
            ("traces.jsonl", "Latest TRAIN/DEV trace snapshots (one JSON object per line) with their partition."),
            ("human_judgments.jsonl", "All non-audit human judgments including superseded ones."),
            ("machine_predictions.jsonl", "All exportable grader predictions, each tagged MACHINE."),
            ("partitions.jsonl", "Group-level partition assignments with seed and exposure status."),
            ("dataset_snapshots.json", "Frozen TRAIN/DEV snapshots: ids, policy epoch, content hash, sizes."),
            ("optimization_runs.json", "Optimizer configuration, budgets, usage, results, and dependency versions."),
            ("policy_epoch.json", "Current policy epoch, policy notes, and epoch history."),
            ("summary.json", "Counts of exported and withheld material."),
        ]
        for aid in audit_ids:
            items += [
                (f"audit_samples/{aid}/report.json", "Locked plan, targets, and the evidence report of this audit."),
                (f"audit_samples/{aid}/traces.jsonl", "Only the locked sample traces of this audit."),
                (f"audit_samples/{aid}/human_judgments.jsonl", "Blind audit judgments including corrections."),
            ]
    for gid in grader_ids:
        items.append((f"graders/{gid}.json", "Portable grader bundle (manifest, prompt text, provenance, status)."))
    items.append(("README.md", "This file."))
    return items


def _released_audits(session: Session, project: Project) -> list[AuditRun]:
    return list(
        session.scalars(
            select(AuditRun)
            .where(AuditRun.project_id == project.id, AuditRun.state.in_(list(RELEASED_AUDIT_STATES)))
            .order_by(AuditRun.created_at)
        )
    )


def _withheld_audit_count(session: Session, project: Project) -> int:
    rows = session.scalars(
        select(AuditRun.id).where(
            AuditRun.project_id == project.id, AuditRun.state.notin_(list(RELEASED_AUDIT_STATES))
        )
    )
    return len(list(rows))


def build_full_entries(session: Session, project: Project) -> dict[str, bytes]:
    """All members of a FULL bundle as ``{name: bytes}``."""
    exported_at = _iso(utcnow())
    assignments = {
        a.group_id: a
        for a in session.scalars(
            select(PartitionAssignment).where(PartitionAssignment.project_id == project.id)
        )
    }
    browsable = bulk_eligible_traces(session, project, None)
    browsable_groups = {t.group_id for t in browsable}
    exportable_group_ids = {
        gid for gid, a in assignments.items()
        if a.partition in (Partition.TRAIN, Partition.DEV) and a.exposure_status not in ("SEALED", "QUARANTINED")
    }

    # Released audits: only their locked samples leave the reserve.
    audits = _released_audits(session, project)
    audit_traces: dict[str, list[TraceSnapshot]] = {}
    audit_unresolved: dict[str, int] = {}
    for a in audits:
        ids = list(a.locked_sample_ids or [])
        rows = {
            t.id: t
            for t in session.scalars(
                select(TraceSnapshot).where(TraceSnapshot.project_id == project.id, TraceSnapshot.id.in_(ids))
            )
        } if ids else {}
        audit_traces[a.id] = [rows[i] for i in ids if i in rows]
        audit_unresolved[a.id] = len([i for i in ids if i not in rows])
    released_trace_ids = {t.id for ts in audit_traces.values() for t in ts}
    released_group_ids = {t.group_id for ts in audit_traces.values() for t in ts}
    released_audit_ids = {a.id for a in audits}

    graders = list_graders(session, project.id)
    grader_meta = _grader_meta(session, project, graders)

    # Traces: latest TRAIN/DEV, browsable only.
    trace_rows = [_trace_row(t, assignments[t.group_id].partition if t.group_id in assignments else None)
                  for t in browsable]

    # Human judgments: everything except audit judgments, for exportable groups (any revision).
    all_traces = {
        t.id: t
        for t in session.scalars(select(TraceSnapshot).where(TraceSnapshot.project_id == project.id))
    }
    judgments = list(
        session.scalars(
            select(HumanJudgment).where(HumanJudgment.project_id == project.id).order_by(HumanJudgment.created_at)
        )
    )
    judgment_rows = []
    for j in judgments:
        t = all_traces.get(j.trace_id)
        if j.purpose == ReviewPurpose.AUDIT or t is None or t.group_id not in exportable_group_ids:
            continue
        judgment_rows.append(_judgment_row(j, t))

    # Audit judgments: by review-request link or by locked sample membership, per released audit only.
    request_audit = {
        r.id: r.audit_run_id
        for r in session.scalars(
            select(ReviewRequest).where(ReviewRequest.project_id == project.id, ReviewRequest.audit_run_id.isnot(None))
        )
    }
    audit_judgment_rows: dict[str, list[dict[str, Any]]] = {a.id: [] for a in audits}
    sample_membership = {a.id: {t.id for t in audit_traces[a.id]} for a in audits}
    for j in judgments:
        if j.purpose != ReviewPurpose.AUDIT:
            continue
        aid = request_audit.get(j.review_request_id or "")
        if aid not in audit_judgment_rows:
            aid = next((a.id for a in audits if j.trace_id in sample_membership[a.id]), None)
        if aid is None:
            continue
        audit_judgment_rows[aid].append(_judgment_row(j, all_traces.get(j.trace_id)))

    # Machine predictions: browsable traces plus released audit samples; AUDIT-purpose runs only when released.
    exportable_trace_ids = {t.id for t in browsable} | released_trace_ids
    prediction_rows = []
    for r in session.scalars(
        select(GradingRun).where(GradingRun.project_id == project.id).order_by(GradingRun.created_at, GradingRun.id)
    ):
        if r.trace_id not in exportable_trace_ids:
            continue
        if r.purpose == GradingPurpose.AUDIT and r.audit_run_id not in released_audit_ids:
            continue
        prediction_rows.append(_prediction_row(r, all_traces.get(r.trace_id), grader_meta))

    partition_rows = [
        _partition_row(a)
        for gid, a in sorted(assignments.items())
        if a.partition in (Partition.TRAIN, Partition.DEV) or gid in released_group_ids
    ]
    withheld_groups = sum(
        1 for gid, a in assignments.items() if a.partition == Partition.AUDIT_RESERVE and gid not in released_group_ids
    )

    snapshots = list(
        session.scalars(
            select(DatasetSnapshot).where(DatasetSnapshot.project_id == project.id).order_by(DatasetSnapshot.created_at)
        )
    )
    runs = list(
        session.scalars(
            select(OptimizationRun).where(OptimizationRun.project_id == project.id).order_by(OptimizationRun.created_at)
        )
    )
    evals_by_run: dict[str | None, list[CandidateEvaluation]] = {}
    for ev in session.scalars(
        select(CandidateEvaluation).where(CandidateEvaluation.project_id == project.id).order_by(CandidateEvaluation.created_at)
    ):
        evals_by_run.setdefault(ev.run_id, []).append(ev)

    entries: dict[str, bytes] = {}
    entries["traces.jsonl"] = _jsonl(trace_rows).encode("utf-8")
    entries["human_judgments.jsonl"] = _jsonl(judgment_rows).encode("utf-8")
    entries["machine_predictions.jsonl"] = _jsonl(prediction_rows).encode("utf-8")
    entries["partitions.jsonl"] = _jsonl(partition_rows).encode("utf-8")
    entries["dataset_snapshots.json"] = _dumps([_snapshot_row(s) for s in snapshots]).encode("utf-8")
    entries["optimization_runs.json"] = _dumps(
        {
            "dependency_versions": dependency_versions(),
            "runs": [_run_row(r, evals_by_run.get(r.id, [])) for r in runs],
            "note": "DEV agreement is a development result on a frozen snapshot, not production evidence.",
        }
    ).encode("utf-8")
    entries["policy_epoch.json"] = _dumps(
        {
            "project_id": project.id,
            "policy_epoch": project.policy_epoch,
            "policy_notes": project.policy_notes,
            "history": (project.configuration or {}).get("policy_epoch_history", []),
            "exported_at": exported_at,
        }
    ).encode("utf-8")
    for a in audits:
        folder = f"audit_samples/{a.id}"
        rows = audit_traces[a.id]
        entries[f"{folder}/report.json"] = _dumps(
            _audit_report_row(a, sample_size=len(rows), unresolved_ids=audit_unresolved[a.id])
        ).encode("utf-8")
        entries[f"{folder}/traces.jsonl"] = _jsonl(
            [_trace_row(t, assignments[t.group_id].partition if t.group_id in assignments else None) for t in rows]
        ).encode("utf-8")
        entries[f"{folder}/human_judgments.jsonl"] = _jsonl(audit_judgment_rows[a.id]).encode("utf-8")
    for g in graders:
        entries[f"graders/{g.id}.json"] = _dumps(grader_bundle(session, g)).encode("utf-8")
    withheld_audits = _withheld_audit_count(session, project)
    entries["summary.json"] = _dumps(
        {
            "kind": "FULL",
            "schema_version": EXPORT_SCHEMA_VERSION,
            "project_id": project.id,
            "policy_epoch": project.policy_epoch,
            "exported_at": exported_at,
            "counts": {
                "traces": len(trace_rows),
                "trace_groups": len(browsable_groups),
                "human_judgments": len(judgment_rows),
                "machine_predictions": len(prediction_rows),
                "partition_rows": len(partition_rows),
                "graders": len(graders),
                "dataset_snapshots": len(snapshots),
                "optimization_runs": len(runs),
                "audits_released": len(audits),
                "audits_withheld": withheld_audits,
                "audit_reserve_groups_withheld": withheld_groups,
            },
            "dependency_versions": dependency_versions(),
        }
    ).encode("utf-8")
    files = _file_descriptions("FULL", [a.id for a in audits], [g.id for g in graders])
    entries["README.md"] = README_TEMPLATE.format(
        kind="FULL",
        project_id=project.id,
        policy_epoch=project.policy_epoch,
        schema_version=EXPORT_SCHEMA_VERSION,
        exported_at=exported_at,
        files="\n".join(f"- `{name}`: {desc}" for name, desc in files),
        machine_note=MACHINE_NOTE,
        withheld_groups=withheld_groups,
        withheld_audits=withheld_audits,
    ).encode("utf-8")
    # Exposure history: the released audit samples have now left the system.
    for gid in sorted(released_group_ids):
        record_exposure(session, project.id, gid, ExposureKind.EXPORT, None)
    return entries


def build_grader_entries(session: Session, project: Project, grader: GraderVersion) -> dict[str, bytes]:
    exported_at = _iso(utcnow())
    entries = {f"graders/{grader.id}.json": _dumps(grader_bundle(session, grader)).encode("utf-8")}
    files = _file_descriptions("GRADER", [], [grader.id])
    entries["README.md"] = README_TEMPLATE.format(
        kind="GRADER",
        project_id=project.id,
        policy_epoch=project.policy_epoch,
        schema_version=EXPORT_SCHEMA_VERSION,
        exported_at=exported_at,
        files="\n".join(f"- `{name}`: {desc}" for name, desc in files),
        machine_note=MACHINE_NOTE,
        withheld_groups="all",
        withheld_audits="all",
    ).encode("utf-8")
    return entries


def write_bundle_zip(
    session: Session,
    project: Project,
    *,
    kind: str,
    grader_id: str | None,
    path: str | Path,
) -> dict[str, Any]:
    """Write a FULL or GRADER bundle zip and return its manifest (files, sizes, sha256)."""
    if kind not in EXPORT_KINDS:
        raise ExportError(f"unknown export kind {kind!r}; expected one of {list(EXPORT_KINDS)}")
    grader = None
    if grader_id is not None:
        grader = get_grader(session, grader_id)
        if grader.project_id != project.id:
            raise ExportError("grader belongs to another project")
    if kind == "GRADER":
        if grader is None:
            raise ExportError("a GRADER export requires grader_id")
        entries = build_grader_entries(session, project, grader)
    else:
        entries = build_full_entries(session, project)
    assert_safe_members(entries)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(entries):
            zf.writestr(name, entries[name])
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        for name in names:
            if name.lower().endswith(FORBIDDEN_MEMBER_SUFFIXES):
                raise ExportError(f"zip contains forbidden member {name!r}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "kind": kind,
        "schema_version": EXPORT_SCHEMA_VERSION,
        "grader_id": grader.id if grader else None,
        "files": sorted(entries),
        "sizes": {name: len(entries[name]) for name in sorted(entries)},
        "file_sha256": {name: hashlib.sha256(entries[name]).hexdigest() for name in sorted(entries)},
        "sha256": digest,
        "bytes": path.stat().st_size,
        "path": str(path),
        "exported_at": _iso(utcnow()),
    }


# ---------------------------------------------------------------- jobs


def create_export(
    session: Session,
    project: Project,
    *,
    kind: str,
    grader_id: str | None,
    idempotency_key: str,
    settings: Settings | None = None,
) -> tuple[ExportBundle, Job]:
    settings = settings or get_settings()
    existing = job_service.find_existing(session, idempotency_key, project_id=project.id, kind=JobKind.EXPORT)
    if existing is not None:
        bundle = session.get(ExportBundle, existing.payload_ref)
        if bundle is None:
            raise ExportError(f"job {existing.id} does not reference an export bundle")
        return bundle, existing
    if kind not in EXPORT_KINDS:
        raise ExportError(f"unknown export kind {kind!r}; expected one of {list(EXPORT_KINDS)}")
    if grader_id is not None:
        grader = get_grader(session, grader_id)
        if grader.project_id != project.id:
            raise ExportError("grader belongs to another project")
    elif kind == "GRADER":
        raise ExportError("a GRADER export requires grader_id")
    bundle = ExportBundle(project_id=project.id, kind=kind, path="", manifest={})
    session.add(bundle)
    session.flush()
    bundle.path = str(Path(settings.artifact_path) / "exports" / f"{bundle.id}.zip")
    job = job_service.enqueue(
        session,
        kind=JobKind.EXPORT,
        payload={"bundle_id": bundle.id, "project_id": project.id, "kind": kind, "grader_id": grader_id},
        idempotency_key=idempotency_key,
        project_id=project.id,
        payload_ref=bundle.id,
        max_attempts=2,
    )
    bundle.job_id = job.id
    session.flush()
    return bundle, job


def export_job_handler(job: Job, ctx) -> dict[str, Any]:
    payload = job.payload or {}
    with ctx.session() as s:
        bundle = s.get(ExportBundle, payload["bundle_id"])
        if bundle is None:
            raise ExportError(f"export bundle {payload['bundle_id']} not found")
        project = s.get(Project, bundle.project_id)
        if project is None:
            raise ExportError(f"project {bundle.project_id} not found")
        ctx.heartbeat(force=True)
        manifest = write_bundle_zip(
            s, project, kind=bundle.kind, grader_id=payload.get("grader_id"), path=bundle.path
        )
        bundle.manifest = manifest
        s.commit()
    ctx.progress(files=len(manifest["files"]), bytes=manifest["bytes"])
    return {"bundle_id": payload["bundle_id"], "files": manifest["files"], "sha256": manifest["sha256"],
            "bytes": manifest["bytes"]}


def export_ready(bundle: ExportBundle, job: Job | None) -> bool:
    return bool(job and job.state == "SUCCEEDED" and bundle.path and Path(bundle.path).exists())
