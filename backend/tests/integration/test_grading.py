"""Grading service: GradingRun persistence, cache provenance, sealed material, failure statuses, leakage.

The module also holds the shared fixture builders used by ``test_optimization.py``:
a project whose groups are assigned to a chosen partition *before* any label exists,
traces imported through the real JSONL path, and labels submitted through the
review service under the fixture's truthful-reporting expert policy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from sqlalchemy import select

from eval_tinder.db.enums import (
    ExposureKind,
    ExposureStatus,
    GraderOrigin,
    GradingPurpose,
    GradingStatus,
    Partition,
    ReviewPurpose,
    SelectionCategory,
)
from eval_tinder.db.models import (
    ExposureEvent,
    GraderVersion,
    GradingRun,
    HumanJudgment,
    PartitionAssignment,
    Project,
    TraceSnapshot,
)
from eval_tinder.domain.rendering import render_trace
from eval_tinder.grader.signature import DEFAULT_SEED_INSTRUCTIONS
from eval_tinder.llm.budget import BudgetGuard
from eval_tinder.llm.fakes import ScriptedGradingLM, keyword_switch_policy
from eval_tinder.services.grading import GraderRuntime, SealedMaterial, cache_key_for, grade_many, grade_trace
from eval_tinder.services.imports import import_jsonl_sync
from eval_tinder.services.projects import bump_policy_epoch, create_grader_version, create_project, seed_grader_for
from eval_tinder.services.review import create_requests, quarantine_group, record_exposure, submit_judgment
from tests.cases import DEV_CANARY, DEV_CASES, TRAIN_CANARY, TRAIN_CASES, DevCase

REVIEWER = "reviewer-canary-5d1c"
PROJECT_DESCRIPTION = "A subscription support assistant that cancels plans on request."
TRUTHFUL_INSTRUCTIONS = (
    DEFAULT_SEED_INSTRUCTIONS
    + "\nJudge whether the answer truthfully reports the recorded outcome of the request."
)


# ----------------------------------------------------------------- shared fixture builders


@dataclass
class Fixture:
    project: Project
    seed: GraderVersion
    train: dict[str, TraceSnapshot]
    dev: dict[str, TraceSnapshot]
    train_labels: dict[str, HumanJudgment]
    dev_labels: dict[str, HumanJudgment]


def make_project(session, settings, **configuration) -> Project:
    return create_project(
        session, name="grading-tests", description=PROJECT_DESCRIPTION, partition_seed=7,
        configuration=configuration or None, settings=settings,
    )


def import_cases(session, project: Project, cases: list[DevCase], partition: str) -> dict[str, TraceSnapshot]:
    """Assign each case's group to ``partition`` before any label exists, then import via the JSONL path."""
    lines = []
    for c in cases:
        group_id = f"grp-9c4e-{c.key}"
        session.add(
            PartitionAssignment(project_id=project.id, group_id=group_id, partition=partition, seed=project.partition_seed)
        )
        lines.append(
            json.dumps(
                {
                    "external_id": f"xid-9c4e-{c.key}",
                    "group_id": group_id,
                    "input": c.input,
                    "output": c.output,
                    "context": c.context or {"subscription_id": "s-demo"},
                    "tool_calls": c.tool_calls(),
                    "metadata": {"task_type": "cancellation", "language": "en"},
                    "source_type": "SYNTHETIC",
                }
            )
        )
    session.flush()
    batch = import_jsonl_sync(session, project, "\n".join(lines))
    assert batch.line_errors == [] and batch.counts["inserted"] == len(cases)
    traces = {
        t.external_id: t
        for t in session.scalars(
            select(TraceSnapshot).where(
                TraceSnapshot.project_id == project.id,
                TraceSnapshot.external_id.in_([f"xid-9c4e-{c.key}" for c in cases]),
            )
        )
    }
    out = {c.key: traces[f"xid-9c4e-{c.key}"] for c in cases}
    for c in cases:
        assert session.scalar(
            select(PartitionAssignment.partition).where(
                PartitionAssignment.project_id == project.id, PartitionAssignment.group_id == out[c.key].group_id
            )
        ) == partition
    return out


def label_cases(
    session, project: Project, traces: dict[str, TraceSnapshot], cases: list[DevCase], purpose: str
) -> dict[str, HumanJudgment]:
    """Blind labels through the review service. Explanations carry the partition canary."""
    canary = TRAIN_CANARY if purpose == ReviewPurpose.TRAIN else DEV_CANARY
    category = SelectionCategory.SEED if purpose == ReviewPurpose.TRAIN else SelectionCategory.DEV_RANDOM
    requests = create_requests(session, project, [traces[c.key] for c in cases], purpose=purpose, category=category)
    out = {}
    for c, req in zip(cases, requests, strict=True):
        out[c.key] = submit_judgment(
            session, req.id, verdict=c.label, explanation=f"{c.explanation} {canary}", reviewer_id=REVIEWER,
            shown_context_hash=traces[c.key].content_hash, active_review_ms=1500,
            idempotency_key=f"label-{project.id}-{req.id}",
        )
    return out


def variants(cases: list[DevCase], count: int, tag: str) -> list[DevCase]:
    """Reworded copies (distinct content, own groups) whose truthful-policy labels stay valid."""
    return [
        DevCase(f"{c.key}{tag}{i}", f"{c.input} (variant {tag}{i})", c.output, c.status, c.label, c.explanation)
        for i in range(count)
        for c in cases
    ]


def labeled_project(
    session, settings, *, train: list[DevCase] | None = None, dev: list[DevCase] | None = None, **configuration
) -> Fixture:
    train = TRAIN_CASES if train is None else train
    dev = DEV_CASES if dev is None else dev
    project = make_project(session, settings, **configuration)
    train_traces = import_cases(session, project, train, Partition.TRAIN)
    dev_traces = import_cases(session, project, dev, Partition.DEV)
    train_labels = label_cases(session, project, train_traces, train, ReviewPurpose.TRAIN)
    dev_labels = label_cases(session, project, dev_traces, dev, ReviewPurpose.DEV)
    return Fixture(project, seed_grader_for(session, project), train_traces, dev_traces, train_labels, dev_labels)


def unlabeled_project(session, settings) -> tuple[Project, GraderVersion, dict[str, TraceSnapshot]]:
    project = make_project(session, settings)
    return project, seed_grader_for(session, project), import_cases(session, project, TRAIN_CASES, Partition.TRAIN)


def recording_runtime(project, grader, settings, *, policy=keyword_switch_policy, budget=None):
    lm = ScriptedGradingLM(policy)
    return GraderRuntime.build(project, grader, settings=settings, budget=budget, lm=lm), lm


def exposure_kinds(session, project_id: str, group_id: str) -> list[str]:
    return list(
        session.scalars(
            select(ExposureEvent.kind)
            .where(ExposureEvent.project_id == project_id, ExposureEvent.group_id == group_id)
            .order_by(ExposureEvent.created_at)
        )
    )


def exposure_status(session, project_id: str, group_id: str) -> str:
    return session.scalar(
        select(PartitionAssignment.exposure_status).where(
            PartitionAssignment.project_id == project_id, PartitionAssignment.group_id == group_id
        )
    )


# ----------------------------------------------------------------- persistence and cache


def test_grade_trace_records_run_and_cache_hits_keep_provenance(db_session, settings):
    project, seed, traces = unlabeled_project(db_session, settings)
    runtime, lm = recording_runtime(project, seed, settings)
    trace = traces["t1"]  # tool status "completed" -> PASS under the seed's completion behaviour

    run = grade_trace(db_session, project, runtime, trace, purpose=GradingPurpose.PROBE, job_id="job-a", settings=settings)
    assert run.status == GradingStatus.OK and run.verdict == "PASS"
    assert run.evidence == [{"pointer": "/tool_calls/0/result/status", "quote": "completed"}]
    assert run.prompt_hash == runtime.manifest.prompt_hash
    assert run.cache_key == cache_key_for(project, runtime.manifest, render_trace(trace))
    assert run.cache_hit_of is None and run.attempt == 1 and run.job_id == "job-a"
    assert run.usage["calls"] == 1 and len(lm.calls) == 1
    assert run.grader_id == seed.id and run.trace_id == trace.id and run.project_id == project.id

    hit = grade_trace(db_session, project, runtime, trace, purpose=GradingPurpose.POOL, job_id="job-b", settings=settings)
    assert hit.id != run.id, "a cache hit is a new GradingRun row, never a mutation of the original"
    assert hit.cache_hit_of == run.id
    assert hit.usage == {"cached": True} and hit.latency_ms == 0
    assert (hit.status, hit.verdict, hit.evidence, hit.explanation) == (run.status, run.verdict, run.evidence, run.explanation)
    assert hit.purpose == GradingPurpose.POOL and hit.job_id == "job-b"
    assert len(lm.calls) == 1, "a cache hit must not call the model"

    fresh = grade_trace(db_session, project, runtime, trace, purpose=GradingPurpose.PROBE, use_cache=False, settings=settings)
    assert fresh.id not in {run.id, hit.id} and fresh.cache_hit_of is None
    assert len(lm.calls) == 2, "use_cache=False re-grades"
    assert db_session.scalar(select(GradingRun).where(GradingRun.id == run.id)).cache_hit_of is None


def test_cache_key_depends_on_manifest_and_policy_epoch(db_session, settings):
    project, seed, traces = unlabeled_project(db_session, settings)
    trace = traces["t2"]
    seed_runtime, seed_lm = recording_runtime(project, seed, settings)
    first = grade_trace(db_session, project, seed_runtime, trace, purpose=GradingPurpose.PROBE, settings=settings)

    other = create_grader_version(
        db_session, project, instruction_text=TRUTHFUL_INSTRUCTIONS, origin=GraderOrigin.IMPORTED, settings=settings,
    )
    assert other.manifest_hash != seed.manifest_hash
    other_runtime, other_lm = recording_runtime(project, other, settings)
    second = grade_trace(db_session, project, other_runtime, trace, purpose=GradingPurpose.PROBE, settings=settings)
    assert second.cache_key != first.cache_key and second.cache_hit_of is None
    assert second.prompt_hash != first.prompt_hash
    assert len(other_lm.calls) == 1, "a different manifest must not be served from the seed's cache"
    # t2: accepted but claims completion -> truthful instructions FAIL it, the seed's completion view also FAILs it
    assert second.verdict == "FAIL"

    bump_policy_epoch(db_session, project, reason="the expert now grades truthful status reporting")
    third = grade_trace(db_session, project, seed_runtime, trace, purpose=GradingPurpose.PROBE, settings=settings)
    assert third.cache_key != first.cache_key and third.cache_hit_of is None
    assert len(seed_lm.calls) == 2, "an epoch bump invalidates the cache"
    assert third.cache_key == cache_key_for(project, seed_runtime.manifest, render_trace(trace))


# ----------------------------------------------------------------- sealed material


@pytest.mark.parametrize("purpose", [GradingPurpose.PROBE, GradingPurpose.POOL, GradingPurpose.BULK])
def test_sealed_and_quarantined_groups_are_refused_for_ordinary_purposes(db_session, settings, purpose):
    project, seed, traces = unlabeled_project(db_session, settings)
    runtime, lm = recording_runtime(project, seed, settings)
    sealed, quarantined = traces["t1"], traces["t2"]
    record_exposure(db_session, project.id, sealed.group_id, ExposureKind.AUDIT_SEALED, "audit-1")
    quarantine_group(db_session, project.id, quarantined.group_id, reason="cross-partition duplicate")
    assert exposure_status(db_session, project.id, sealed.group_id) == ExposureStatus.SEALED
    assert exposure_status(db_session, project.id, quarantined.group_id) == ExposureStatus.QUARANTINED

    with pytest.raises(SealedMaterial, match="SEALED"):
        grade_trace(db_session, project, runtime, sealed, purpose=purpose, settings=settings)
    with pytest.raises(SealedMaterial, match="QUARANTINED"):
        grade_trace(db_session, project, runtime, quarantined, purpose=purpose, settings=settings)
    assert lm.calls == [] and db_session.scalar(select(GradingRun).limit(1)) is None


@pytest.mark.parametrize("purpose", [GradingPurpose.PROBE, GradingPurpose.POOL, GradingPurpose.BULK])
def test_audit_reserve_is_refused_for_ordinary_purposes(db_session, settings, purpose):
    project = make_project(db_session, settings)
    seed = seed_grader_for(db_session, project)
    reserve = import_cases(db_session, project, DEV_CASES[:1], Partition.AUDIT_RESERVE)["d1"]
    runtime, lm = recording_runtime(project, seed, settings)
    with pytest.raises(SealedMaterial, match="AUDIT_RESERVE"):
        grade_trace(db_session, project, runtime, reserve, purpose=purpose, settings=settings)
    assert lm.calls == []


def test_audit_purpose_may_grade_sealed_groups(db_session, settings):
    project = make_project(db_session, settings)
    seed = seed_grader_for(db_session, project)
    reserve = import_cases(db_session, project, DEV_CASES[:1], Partition.AUDIT_RESERVE)["d1"]
    record_exposure(db_session, project.id, reserve.group_id, ExposureKind.AUDIT_SEALED, "audit-1")
    runtime, lm = recording_runtime(project, seed, settings)
    run = grade_trace(
        db_session, project, runtime, reserve, purpose=GradingPurpose.AUDIT, audit_run_id="audit-1",
        use_cache=False, settings=settings,
    )
    assert run.status == GradingStatus.OK and run.audit_run_id == "audit-1" and len(lm.calls) == 1
    # the audit service owns audit exposure; ordinary grading does not add an exposure record for AUDIT
    assert exposure_kinds(db_session, project.id, reserve.group_id) == [ExposureKind.AUDIT_SEALED]
    assert exposure_status(db_session, project.id, reserve.group_id) == ExposureStatus.SEALED


# ----------------------------------------------------------------- failure statuses never become PASS


def test_provider_error_is_recorded_as_review(db_session, settings):
    project, seed, traces = unlabeled_project(db_session, settings)

    def provider_down(system_text, case):
        raise RuntimeError("provider unavailable (503)")

    runtime, _ = recording_runtime(project, seed, settings, policy=provider_down)
    run = grade_trace(db_session, project, runtime, traces["t1"], purpose=GradingPurpose.PROBE, settings=settings)
    assert run.status == GradingStatus.PROVIDER_ERROR and run.verdict == "REVIEW"
    assert "RuntimeError" in (run.error or "") and "provider unavailable" in run.error
    assert run.attempt == 2 and run.evidence == []


def test_malformed_output_is_recorded_as_review(db_session, settings):
    project, seed, traces = unlabeled_project(db_session, settings)
    runtime, lm = recording_runtime(project, seed, settings, policy=lambda s, c: {"raw": "garbage"})
    run = grade_trace(db_session, project, runtime, traces["t1"], purpose=GradingPurpose.PROBE, settings=settings)
    assert run.status == GradingStatus.MALFORMED_OUTPUT and run.verdict == "REVIEW"
    assert run.attempt == 2 and lm.calls, "bounded retries happened, then the failure was recorded"


def test_invalid_evidence_pointer_is_recorded_as_review(db_session, settings):
    project, seed, traces = unlabeled_project(db_session, settings)
    bad = {"verdict": "PASS", "evidence_json": json.dumps([{"pointer": "/nope", "quote": ""}]), "explanation": "x"}
    runtime, lm = recording_runtime(project, seed, settings, policy=lambda s, c: bad)
    run = grade_trace(db_session, project, runtime, traces["t1"], purpose=GradingPurpose.PROBE, settings=settings)
    assert run.status == GradingStatus.INVALID_EVIDENCE and run.verdict == "REVIEW"
    assert run.evidence == [] and "nope" in (run.error or "") and run.attempt == 2 == len(lm.calls)
    # a non-OK result is never served from the cache later
    again = grade_trace(db_session, project, runtime, traces["t1"], purpose=GradingPurpose.PROBE, settings=settings)
    assert again.cache_hit_of is None and len(lm.calls) == 4


def test_oversized_case_is_review_without_any_model_call(db_session, settings):
    project, seed, traces = unlabeled_project(db_session, settings)
    runtime, lm = recording_runtime(project, seed, settings)
    small = settings.model_copy(update={"max_case_chars": 50})
    run = grade_trace(db_session, project, runtime, traces["t1"], purpose=GradingPurpose.PROBE, settings=small)
    assert run.status == GradingStatus.CONTEXT_TOO_LARGE and run.verdict == "REVIEW"
    assert lm.calls == [] and run.attempt == 0 and run.usage.get("calls", 0) == 0
    assert "> 50" in (run.error or "")


def test_budget_exhaustion_is_review_and_marks_guard(db_session, settings):
    project, seed, traces = unlabeled_project(db_session, settings)
    guard = BudgetGuard(max_calls=1, max_total_tokens=1_000_000, max_tokens_per_call=200)
    runtime, lm = recording_runtime(project, seed, settings, budget=guard)
    first = grade_trace(db_session, project, runtime, traces["t1"], purpose=GradingPurpose.PROBE, settings=settings)
    assert first.status == GradingStatus.OK and guard.exhausted is False
    second = grade_trace(db_session, project, runtime, traces["t2"], purpose=GradingPurpose.PROBE, settings=settings)
    assert second.status == GradingStatus.BUDGET_EXHAUSTED and second.verdict == "REVIEW"
    assert "budget exhausted" in (second.error or "").lower()
    assert guard.exhausted is True and guard.totals.calls == 1 and len(lm.calls) == 1
    assert guard.snapshot()["by_role"]["grading"]["calls"] == 1


# ----------------------------------------------------------------- leakage


def test_grading_model_sees_evidence_but_no_labels_or_bookkeeping(db_session, settings):
    fx = labeled_project(db_session, settings)
    trace, judgment = fx.train["t2"], fx.train_labels["t2"]
    assert TRAIN_CANARY in judgment.explanation and judgment.reviewer_id == REVIEWER
    runtime, lm = recording_runtime(fx.project, fx.seed, settings)
    run = grade_trace(db_session, fx.project, runtime, trace, purpose=GradingPurpose.PROBE, settings=settings)
    assert run.status == GradingStatus.OK and len(lm.calls) == 1

    text = lm.all_prompt_text()
    assert {m["role"] for m in lm.calls[0].messages} == {"system", "user"}
    for forbidden in (
        TRAIN_CANARY, DEV_CANARY, judgment.explanation, "Expert grade", "expert_grade", "expert_explanation",
        REVIEWER, trace.external_id, trace.group_id, trace.id, fx.project.id, fx.seed.id, judgment.id,
        trace.content_hash, "xid-9c4e", "grp-9c4e",
    ):
        assert forbidden not in text, f"grading prompt leaked {forbidden!r}"
    assert trace.input in text and trace.output in text
    assert '"status": "accepted"' in text, "tool evidence is part of the case"
    assert PROJECT_DESCRIPTION in text
    assert render_trace(trace).text in text, "the model receives exactly the rendered case document"
    assert "<<<CASE_JSON" in text and '"input"' in text and '"output"' in text and '"tool_calls"' in text


# ----------------------------------------------------------------- grade_many exposure


def test_grade_many_records_exposure_per_purpose(db_session, settings):
    project, seed, traces = unlabeled_project(db_session, settings)
    runtime, lm = recording_runtime(project, seed, settings)
    plan = {
        GradingPurpose.PROBE: (["t1", "t2"], ExposureKind.PROBE),
        GradingPurpose.POOL: (["t3", "t4"], ExposureKind.POOL),
        GradingPurpose.BULK: (["t5", "t6"], ExposureKind.BULK_GRADING),
    }
    for key in traces:
        assert exposure_status(db_session, project.id, traces[key].group_id) == ExposureStatus.UNTOUCHED
    progress = []
    for purpose, (keys, kind) in plan.items():
        runs = grade_many(
            db_session, project, runtime, [traces[k] for k in keys], purpose=purpose, job_id=f"job-{purpose}",
            on_progress=progress.append, settings=settings,
        )
        assert [r.purpose for r in runs] == [purpose, purpose]
        for k in keys:
            group = traces[k].group_id
            events = list(
                db_session.scalars(
                    select(ExposureEvent).where(ExposureEvent.project_id == project.id, ExposureEvent.group_id == group)
                )
            )
            assert [(e.kind, e.reference_id) for e in events] == [(kind, f"job-{purpose}")]
            assert exposure_status(db_session, project.id, group) == ExposureStatus.INSPECTED
    assert progress == [1, 2, 1, 2, 1, 2]
    assert len(lm.calls) == 6

    # a cached probe is still a probe use of the group: exposure history is appended on cache hits too
    cached = grade_many(db_session, project, runtime, [traces["t1"]], purpose=GradingPurpose.PROBE, job_id="job-2", settings=settings)
    assert cached[0].cache_hit_of is not None and len(lm.calls) == 6
    assert exposure_kinds(db_session, project.id, traces["t1"].group_id) == [ExposureKind.PROBE, ExposureKind.PROBE]
