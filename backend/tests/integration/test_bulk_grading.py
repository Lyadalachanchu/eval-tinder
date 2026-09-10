"""Bulk grading: frozen manifest per job, TRAIN/DEV only, cache provenance, cancellation, budgets.

Groups are mapped to partitions with the project's own seed through
``domain.partitions.assign_partition`` and imported through the real JSONL path,
so the tests exercise the same partition rules the application enforces.
The helpers here are shared with ``test_exports.py`` and the API tests.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from sqlalchemy import select

from eval_tinder.db.enums import ExposureKind, GradingPurpose, GradingStatus, JobKind, JobState, Partition
from eval_tinder.db.models import ExposureEvent, GraderVersion, GradingRun, Job, Project, TraceSnapshot
from eval_tinder.domain.partitions import assign_partition, validate_split
from eval_tinder.grader.signature import DEFAULT_SEED_INSTRUCTIONS
from eval_tinder.llm.fakes import ScriptedGradingLM, keyword_switch_policy
from eval_tinder.services import bulk_grading, grading
from eval_tinder.services import jobs as job_service
from eval_tinder.services.bulk_grading import (
    BulkGradingError,
    bulk_grading_job_handler,
    enqueue_bulk_grading,
    predictions_for,
)
from eval_tinder.services.imports import import_jsonl_sync
from eval_tinder.services.projects import create_grader_version, create_project, project_config, seed_grader_for
from eval_tinder.services.review import quarantine_group, record_exposure
from eval_tinder.worker.main import Worker, drain
from tests.cases import TRAIN_CASES, DevCase

PARTITION_SEED = 4242
RESERVE_CANARY = "CANARY-RESERVE-4e9a"
TRUTHFUL_INSTRUCTIONS = (
    DEFAULT_SEED_INSTRUCTIONS + "\nJudge whether the answer truthfully reports the recorded outcome of the request."
)


# ----------------------------------------------------------------- shared fixture helpers


def make_project(session, settings, *, description: str = "A subscription assistant that cancels plans.") -> Project:
    return create_project(session, name="bulk-export-tests", description=description, partition_seed=PARTITION_SEED,
                          settings=settings)


def group_ids_for(project: Project, partition: str, count: int, *, prefix: str = "grp") -> list[str]:
    """Group ids that the project's own seeded split assigns to ``partition``."""
    split = validate_split(project_config(project).partition_split)
    picks: list[str] = []
    i = 0
    while len(picks) < count:
        gid = f"{prefix}-{partition.lower()}-{i:04d}"
        if assign_partition(gid, project.partition_seed, split) == partition:
            picks.append(gid)
        i += 1
        assert i < 100_000, "could not find enough group ids for the partition"
    return picks


def record_for(gid: str, case: DevCase, *, source_type: str = "SYNTHETIC", output_suffix: str = "") -> dict:
    return {
        "external_id": f"{gid}-r1",
        "group_id": gid,
        "timestamp": "2026-08-12T10:30:00Z",
        "input": f"{case.input} [{gid}]",
        "output": case.output + output_suffix,
        "context": case.context or {"subscription_id": "s-demo"},
        "tool_calls": case.tool_calls(),
        "metadata": {"task_type": "cancellation", "language": "en"},
        "source_type": source_type,
    }


@dataclass
class Imported:
    project: Project
    grader: GraderVersion
    train: list[TraceSnapshot]
    dev: list[TraceSnapshot]
    reserve: list[TraceSnapshot]

    @property
    def browsable(self) -> list[TraceSnapshot]:
        return self.train + self.dev


def import_partitioned(
    session, settings, *, train: int = 4, dev: int = 3, reserve: int = 3, reserve_source: str = "SYNTHETIC",
    truthful_grader: bool = False,
) -> Imported:
    project = make_project(session, settings)
    lines = []
    plan = {
        Partition.TRAIN.value: group_ids_for(project, Partition.TRAIN.value, train),
        Partition.DEV.value: group_ids_for(project, Partition.DEV.value, dev),
        Partition.AUDIT_RESERVE.value: group_ids_for(project, Partition.AUDIT_RESERVE.value, reserve),
    }
    for partition, gids in plan.items():
        for i, gid in enumerate(gids):
            case = TRAIN_CASES[i % len(TRAIN_CASES)]
            suffix = f" {RESERVE_CANARY}" if partition == Partition.AUDIT_RESERVE else ""
            source = reserve_source if partition == Partition.AUDIT_RESERVE else "SYNTHETIC"
            lines.append(json.dumps(record_for(gid, case, source_type=source, output_suffix=suffix)))
    batch = import_jsonl_sync(session, project, "\n".join(lines))
    assert batch.line_errors == [] and batch.counts["inserted"] == train + dev + reserve
    traces = {t.group_id: t for t in session.scalars(select(TraceSnapshot).where(TraceSnapshot.project_id == project.id))}
    grader = seed_grader_for(session, project)
    if truthful_grader:
        grader = create_grader_version(
            session, project, instruction_text=TRUTHFUL_INSTRUCTIONS, origin="IMPORTED", label="truthful",
            settings=settings,
        )
    return Imported(
        project=project,
        grader=grader,
        train=[traces[g] for g in plan[Partition.TRAIN.value]],
        dev=[traces[g] for g in plan[Partition.DEV.value]],
        reserve=[traces[g] for g in plan[Partition.AUDIT_RESERVE.value]],
    )


def bulk_worker(settings, session_factory, *, worker_id: str = "bulk-w") -> Worker:
    return Worker({JobKind.BULK_GRADING: bulk_grading_job_handler}, settings=settings, worker_id=worker_id,
                  session_factory=session_factory)


def runs_for_job(session, job_id: str) -> list[GradingRun]:
    return list(session.scalars(select(GradingRun).where(GradingRun.job_id == job_id).order_by(GradingRun.created_at)))


# ----------------------------------------------------------------- enqueue


def test_enqueue_freezes_manifest_and_targets_train_dev_only(db_session, settings):
    fx = import_partitioned(db_session, settings)
    job = enqueue_bulk_grading(db_session, fx.project, grader_id=fx.grader.id, partition=None, trace_ids=None,
                               idempotency_key="bulk-1", settings=settings)
    assert job.kind == JobKind.BULK_GRADING and job.state == JobState.QUEUED
    assert job.payload["manifest_hash"] == fx.grader.manifest_hash
    assert job.payload["grader_id"] == fx.grader.id
    assert set(job.payload["trace_ids"]) == {t.id for t in fx.browsable}
    assert not set(job.payload["trace_ids"]) & {t.id for t in fx.reserve}
    # idempotent replay returns the same job
    again = enqueue_bulk_grading(db_session, fx.project, grader_id=fx.grader.id, partition="DEV", trace_ids=None,
                                 idempotency_key="bulk-1", settings=settings)
    assert again.id == job.id and again.payload["partition"] is None
    only_train = enqueue_bulk_grading(db_session, fx.project, grader_id=fx.grader.id, partition="TRAIN",
                                      trace_ids=None, idempotency_key="bulk-train", settings=settings)
    assert set(only_train.payload["trace_ids"]) == {t.id for t in fx.train}


def test_enqueue_refuses_audit_reserve_sealed_and_quarantined_material(db_session, settings):
    fx = import_partitioned(db_session, settings)
    with pytest.raises(ValueError, match="AUDIT_RESERVE"):
        enqueue_bulk_grading(db_session, fx.project, grader_id=fx.grader.id, partition="AUDIT_RESERVE",
                             trace_ids=None, idempotency_key="bulk-bad-partition", settings=settings)
    with pytest.raises(BulkGradingError, match="AUDIT_RESERVE"):
        enqueue_bulk_grading(db_session, fx.project, grader_id=fx.grader.id, partition=None,
                             trace_ids=[fx.train[0].id, fx.reserve[0].id], idempotency_key="bulk-bad-ids",
                             settings=settings)
    # a sealed DEV group (audit material) and a quarantined TRAIN group are refused too
    record_exposure(db_session, fx.project.id, fx.dev[0].group_id, ExposureKind.AUDIT_SEALED, "audit-x")
    quarantine_group(db_session, fx.project.id, fx.train[0].group_id, reason="cross-partition duplicate")
    with pytest.raises(BulkGradingError, match="SEALED"):
        enqueue_bulk_grading(db_session, fx.project, grader_id=fx.grader.id, partition=None, trace_ids=[fx.dev[0].id],
                             idempotency_key="bulk-sealed", settings=settings)
    with pytest.raises(BulkGradingError, match="QUARANTINED"):
        enqueue_bulk_grading(db_session, fx.project, grader_id=fx.grader.id, partition=None,
                             trace_ids=[fx.train[0].id], idempotency_key="bulk-quarantined", settings=settings)
    # and the default target list silently leaves them out
    job = enqueue_bulk_grading(db_session, fx.project, grader_id=fx.grader.id, partition=None, trace_ids=None,
                               idempotency_key="bulk-rest", settings=settings)
    assert set(job.payload["trace_ids"]) == {t.id for t in fx.browsable} - {fx.dev[0].id, fx.train[0].id}
    with pytest.raises(BulkGradingError, match="unknown trace"):
        enqueue_bulk_grading(db_session, fx.project, grader_id=fx.grader.id, partition=None,
                             trace_ids=["no-such-trace"], idempotency_key="bulk-unknown", settings=settings)


# ----------------------------------------------------------------- worker run


def test_worker_stores_bulk_runs_for_train_and_dev_never_audit_reserve(db_session, session_factory, settings):
    fx = import_partitioned(db_session, settings)
    job = enqueue_bulk_grading(db_session, fx.project, grader_id=fx.grader.id, partition=None, trace_ids=None,
                               idempotency_key="bulk-run", settings=settings)
    db_session.commit()
    assert drain(bulk_worker(settings, session_factory)) == 1
    db_session.expire_all()
    job = db_session.get(Job, job.id)
    assert job.state == JobState.SUCCEEDED
    assert job.result["complete"] is True and job.result["partial"] is False
    assert job.result["graded"] == len(fx.browsable) and job.result["ok"] == len(fx.browsable)
    assert job.result["cache_hits"] == 0 and job.result["errors"] == 0
    assert job.progress["done"] == job.progress["total"] == len(fx.browsable)
    runs = runs_for_job(db_session, job.id)
    assert len(runs) == len(fx.browsable)
    assert {r.trace_id for r in runs} == {t.id for t in fx.browsable}
    assert all(r.purpose == GradingPurpose.BULK and r.grader_id == fx.grader.id for r in runs)
    assert all(r.prompt_hash == grading.GraderRuntime.build(fx.project, fx.grader, settings=settings).manifest.prompt_hash
               for r in runs)
    reserve_ids = {t.id for t in fx.reserve}
    assert not db_session.scalars(select(GradingRun).where(GradingRun.trace_id.in_(reserve_ids))).all()
    exposures = set(db_session.scalars(select(ExposureEvent.group_id).where(
        ExposureEvent.project_id == fx.project.id, ExposureEvent.kind == ExposureKind.BULK_GRADING)))
    assert exposures == {t.group_id for t in fx.browsable}
    assert not exposures & {t.group_id for t in fx.reserve}


def test_second_run_hits_cache_with_provenance_and_no_new_calls(db_session, session_factory, settings, monkeypatch):
    fx = import_partitioned(db_session, settings, train=3, dev=2, reserve=1)
    lms: list[ScriptedGradingLM] = []

    def recording_lm(model_config, *, settings=None):
        lm = ScriptedGradingLM(keyword_switch_policy, model=model_config.model)
        lms.append(lm)
        return lm

    monkeypatch.setattr(grading, "build_grading_lm", recording_lm)
    first = enqueue_bulk_grading(db_session, fx.project, grader_id=fx.grader.id, partition=None, trace_ids=None,
                                 idempotency_key="bulk-c1", settings=settings)
    db_session.commit()
    drain(bulk_worker(settings, session_factory))
    second = enqueue_bulk_grading(db_session, fx.project, grader_id=fx.grader.id, partition=None, trace_ids=None,
                                  idempotency_key="bulk-c2", settings=settings)
    db_session.commit()
    drain(bulk_worker(settings, session_factory))
    db_session.expire_all()
    first_runs = {r.trace_id: r for r in runs_for_job(db_session, first.id)}
    second_runs = runs_for_job(db_session, second.id)
    assert len(second_runs) == len(first_runs) == 5
    for r in second_runs:
        assert r.cache_hit_of == first_runs[r.trace_id].id
        assert r.verdict == first_runs[r.trace_id].verdict and r.usage == {"cached": True}
    assert db_session.get(Job, second.id).result["cache_hits"] == 5
    assert len(lms) == 2 and len(lms[0].calls) == 5 and len(lms[1].calls) == 0


def test_cancellation_mid_job_ends_cancelled_without_double_publishing(db_session, session_factory, settings,
                                                                       monkeypatch):
    fx = import_partitioned(db_session, settings, train=4, dev=2, reserve=1)
    monkeypatch.setattr(bulk_grading, "PROGRESS_EVERY", 1)
    calls = {"n": 0}

    def cancelling_policy(system_text, case):
        calls["n"] += 1
        if calls["n"] == 2:  # the expert cancels while the second trace is being graded
            with session_factory() as s:
                running = s.scalar(select(Job).where(Job.kind == JobKind.BULK_GRADING, Job.state == JobState.RUNNING))
                job_service.request_cancel(s, running.id)
                s.commit()
        return keyword_switch_policy(system_text, case)

    monkeypatch.setattr(grading, "build_grading_lm", lambda mc, *, settings=None: ScriptedGradingLM(cancelling_policy))
    job = enqueue_bulk_grading(db_session, fx.project, grader_id=fx.grader.id, partition=None, trace_ids=None,
                               idempotency_key="bulk-cancel", settings=settings)
    db_session.commit()
    worker = bulk_worker(settings, session_factory)
    assert drain(worker) == 1
    db_session.expire_all()
    job = db_session.get(Job, job.id)
    assert job.state == JobState.CANCELLED and job.cancel_requested
    assert "complete" not in job.result and job.progress.get("cancelled") is True
    runs = runs_for_job(db_session, job.id)
    assert 1 <= len(runs) < len(fx.browsable)  # the graded prefix stays, the rest was never graded
    assert calls["n"] == len(runs)
    finished_at = job.finished_at
    # A second worker (restart) finds nothing runnable and cannot re-publish or re-finalize the job.
    assert drain(bulk_worker(settings, session_factory, worker_id="bulk-w2")) == 0
    db_session.expire_all()
    job = db_session.get(Job, job.id)
    assert job.state == JobState.CANCELLED and job.finished_at == finished_at
    assert len(runs_for_job(db_session, job.id)) == len(runs)
    with pytest.raises(job_service.LeaseLost):
        job_service.finalize(db_session, job.id, "bulk-w", JobState.SUCCEEDED, result={"complete": True})


def test_budget_exhaustion_keeps_graded_prefix_and_marks_partial(db_session, session_factory, settings):
    fx = import_partitioned(db_session, settings, train=4, dev=2, reserve=1)
    tight = settings.model_copy(update={"max_provider_calls_per_job": 3})
    job = enqueue_bulk_grading(db_session, fx.project, grader_id=fx.grader.id, partition=None, trace_ids=None,
                               idempotency_key="bulk-budget", settings=settings)
    db_session.commit()
    assert drain(bulk_worker(tight, session_factory)) == 1
    db_session.expire_all()
    job = db_session.get(Job, job.id)
    assert job.state == JobState.BUDGET_EXHAUSTED
    assert job.result == {"partial": True}  # never claims completion
    assert "budget exhausted" in (job.error or "").lower()
    runs = runs_for_job(db_session, job.id)
    assert [r.trace_id for r in runs] == job.payload["trace_ids"][:3]  # the graded prefix, in order
    assert all(r.status == GradingStatus.OK for r in runs)
    assert not db_session.scalars(select(GradingRun).where(GradingRun.status == GradingStatus.BUDGET_EXHAUSTED)).all()
    assert job.progress["done"] == 3 and job.progress["total"] == 6 and job.progress["partial"] is True
    # the untouched remainder can be graded by a new explicit job; nothing was invented for it
    remaining = job.payload["trace_ids"][3:]
    follow_up = enqueue_bulk_grading(db_session, fx.project, grader_id=fx.grader.id, partition=None,
                                     trace_ids=remaining, idempotency_key="bulk-budget-2", settings=settings)
    db_session.commit()
    drain(bulk_worker(settings, session_factory))
    db_session.expire_all()
    assert db_session.get(Job, follow_up.id).state == JobState.SUCCEEDED
    assert len(runs_for_job(db_session, follow_up.id)) == 3


def test_handler_refuses_a_changed_manifest(db_session, session_factory, settings):
    fx = import_partitioned(db_session, settings, train=2, dev=1, reserve=1)
    job = enqueue_bulk_grading(db_session, fx.project, grader_id=fx.grader.id, partition=None, trace_ids=None,
                               idempotency_key="bulk-tamper", settings=settings)
    job.payload = {**job.payload, "manifest_hash": "0" * 64}
    db_session.commit()
    drain(bulk_worker(settings, session_factory))
    db_session.expire_all()
    job = db_session.get(Job, job.id)
    assert job.state == JobState.FAILED and "manifest changed" in (job.error or "")
    assert runs_for_job(db_session, job.id) == []


# ----------------------------------------------------------------- predictions


def test_predictions_are_machine_tagged_provisional_and_confidence_free(db_session, session_factory, settings):
    fx = import_partitioned(db_session, settings, truthful_grader=True)
    job = enqueue_bulk_grading(db_session, fx.project, grader_id=fx.grader.id, partition=None, trace_ids=None,
                               idempotency_key="bulk-pred", settings=settings)
    db_session.commit()
    drain(bulk_worker(settings, session_factory))
    db_session.expire_all()
    fx.project = db_session.get(Project, fx.project.id)
    items = predictions_for(db_session, fx.project, grader_id=fx.grader.id)
    assert {p["trace_id"] for p in items} == {t.id for t in fx.browsable}
    assert not {p["trace_id"] for p in items} & {t.id for t in fx.reserve}
    for p in items:
        assert p["kind"] == "MACHINE" and p["provisional"] is True
        assert p["automation_status"] == "DISABLED" and p["audit_status"] == "UNAUDITED"
        assert p["grader_id"] == fx.grader.id and p["manifest_hash"] == fx.grader.manifest_hash
        assert p["status"] == GradingStatus.OK and p["verdict"] in {"PASS", "FAIL", "REVIEW"}
        assert not any("confidence" in key.lower() or "probability" in key.lower() for key in p)
        assert p["grading_run_id"] in {r.id for r in runs_for_job(db_session, job.id)}
    # explicit trace filter; audit material is refused rather than silently dropped
    subset = predictions_for(db_session, fx.project, grader_id=fx.grader.id, trace_ids=[fx.train[0].id])
    assert [p["trace_id"] for p in subset] == [fx.train[0].id]
    with pytest.raises(BulkGradingError, match="not browsable"):
        predictions_for(db_session, fx.project, grader_id=fx.grader.id, trace_ids=[fx.reserve[0].id])
    # a grader that never graded anything has no predictions (nothing is invented)
    assert predictions_for(db_session, fx.project, grader_id=seed_grader_for(db_session, fx.project).id) == []
