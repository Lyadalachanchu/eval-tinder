"""Integration tests for the locked, blind, independent production audit.

Ground truth (the *truthful reporting* expert policy) lives in this test module, never in the
application grader. The fake provider grades truthfully only when the grader's instruction text
mentions truthful reporting; the generic seed grader grades task completion instead, which
produces disagreements on queued/processing cases.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from sqlalchemy import func, select

from eval_tinder.config import Settings
from eval_tinder.db.enums import (
    AuditState,
    ExposureKind,
    ExposureStatus,
    GradingStatus,
    JobState,
    ReviewRequestState,
)
from eval_tinder.db.models import (
    AuditRun,
    ExposureEvent,
    GradingRun,
    GraderVersion,
    HumanJudgment,
    Job,
    PartitionAssignment,
    Project,
    ReviewRequest,
    TraceSnapshot,
    UsageRecord,
)
from eval_tinder.domain.manifest import GraderManifest, pipeline_hash
from eval_tinder.domain.metrics import NOT_ESTIMABLE, bonferroni_confidence, error_upper_bound
from eval_tinder.domain.partitions import assign_partition, validate_split
from eval_tinder.services import audits, review
from eval_tinder.services.audits import AuditError, audit_grading_job_handler
from eval_tinder.services.grading import GraderRuntime, SealedMaterial, grade_trace
from eval_tinder.services.imports import import_jsonl_sync
from eval_tinder.services.projects import create_grader_version, create_project, project_config, seed_grader_for
from eval_tinder.worker.main import Worker, drain

PARTITION_SEED = 20260910

TRUTHFUL_INSTRUCTIONS = (
    "Predict the domain expert's judgment of the target output.\n"
    "Judge whether the answer truthfully reports the recorded outcome of the request: a completion claim is "
    "acceptable only when the recorded tool status shows completion; a processing/queued answer is acceptable "
    "when the recorded status is accepted or queued.\n"
    "Use REVIEW when essential evidence or the applicable standard is unclear."
)

# (recorded tool status, assistant output, expert label under the truthful-reporting policy)
CASES: list[tuple[str, str, str]] = [
    ("completed", "Your subscription has been cancelled.", "PASS"),
    ("accepted", "Your cancellation request is processing and will be confirmed shortly.", "PASS"),
    ("accepted", "Your subscription has been cancelled.", "FAIL"),
    ("failed", "I could not cancel the subscription because the system returned an error.", "PASS"),
    ("failed", "Done, your subscription has been cancelled.", "FAIL"),
    ("completed", "Your request is processing.", "FAIL"),
]

SAMPLING_PLAN = {
    "unit": "group",
    "method": "uniform_random",
    "independence_assumption_documented": True,
    "independence_note": "Each group is one customer conversation; groups were drawn uniformly from the untouched reserve.",
}
RISK_TARGETS = {
    "permitted_verdicts": ["PASS", "FAIL"],
    "max_error_rate": 0.2,
    "min_coverage": 0.8,
    "confidence": 0.95,
}


def group_ids_for(project: Project, partition: str, count: int, prefix: str) -> list[str]:
    """Group ids the project's seeded split assigns to ``partition`` (the split is never rearranged)."""
    split = validate_split(project_config(project).partition_split)
    out: list[str] = []
    i = 0
    while len(out) < count:
        i += 1
        gid = f"{prefix}-{i}"
        if assign_partition(gid, project.partition_seed, split) == partition:
            out.append(gid)
    return out


def make_record(
    external_id: str,
    group_id: str,
    case: tuple[str, str, str],
    *,
    day: int,
    task_type: str = "cancellation",
    source: str = "PRODUCTION",
    timestamp: str | None = "auto",
) -> dict:
    status, output, _label = case
    ts = f"2026-03-{day:02d}T10:00:00Z" if timestamp == "auto" else timestamp
    # Context carries the group's own subscription id so records never collapse into one duplicate group.
    subscription = f"s-{group_id}"
    return {
        "external_id": external_id,
        "group_id": group_id,
        "timestamp": ts,
        "input": f"Cancel my subscription {subscription}.",
        "context": {"subscription_id": subscription},
        "tool_calls": [{"name": "cancel_subscription", "arguments": {"subscription_id": subscription},
                        "result": {"status": status}}],
        "output": output,
        "metadata": {"task_type": task_type, "language": "en"},
        "source_type": source,
    }


@dataclass
class AuditFixture:
    project: Project
    truthful_grader: GraderVersion
    seed_grader: GraderVersion
    settings: Settings
    truth: dict[str, str] = field(default_factory=dict)  # external_id -> expert label
    reserve_groups: list[str] = field(default_factory=list)


def build_project(session, settings: Settings, *, reserve: int = 40, train: int = 6, dev: int = 4) -> AuditFixture:
    project = create_project(
        session, name="audit-project", description="Subscription cancellation assistant",
        partition_seed=PARTITION_SEED, settings=settings,
    )
    fx = AuditFixture(project=project, truthful_grader=None, seed_grader=None, settings=settings)  # type: ignore[arg-type]
    records = []
    fx.reserve_groups = group_ids_for(project, "AUDIT_RESERVE", reserve, "prod")
    for i, gid in enumerate(fx.reserve_groups):
        case = CASES[i % len(CASES)]
        rec = make_record(f"{gid}-final", gid, case, day=1 + i % 28)
        records.append(rec)
        fx.truth[rec["external_id"]] = case[2]
    for partition, count, prefix in (("TRAIN", train, "train"), ("DEV", dev, "dev")):
        for i, gid in enumerate(group_ids_for(project, partition, count, prefix)):
            case = CASES[i % len(CASES)]
            rec = make_record(f"{gid}-final", gid, case, day=1 + i % 28)
            records.append(rec)
            fx.truth[rec["external_id"]] = case[2]
    import_jsonl_sync(session, project, "\n".join(json.dumps(r) for r in records))
    fx.truthful_grader = create_grader_version(
        session, project, instruction_text=TRUTHFUL_INSTRUCTIONS, origin="IMPORTED", label="truthful", settings=settings
    )
    fx.seed_grader = seed_grader_for(session, project)
    session.flush()
    return fx


def lock(session, fx: AuditFixture, *, grader=None, planned_n=20, seed=7, risk_targets=None, sampling_plan=None,
         population=None, key="audit-1"):
    return audits.lock_audit(
        session, fx.project, grader_id=(grader or fx.truthful_grader).id, planned_n=planned_n, seed=seed,
        population=population if population is not None else {},
        sampling_plan=sampling_plan if sampling_plan is not None else SAMPLING_PLAN,
        risk_targets=risk_targets if risk_targets is not None else RISK_TARGETS,
        idempotency_key=key, settings=fx.settings,
    )


def run_grading(db_session, session_factory, settings) -> int:
    db_session.commit()
    worker = Worker({"AUDIT_GRADING": audit_grading_job_handler}, settings=settings, worker_id="w-audit",
                    session_factory=session_factory)
    n = drain(worker)
    db_session.expire_all()
    return n


def truth_labeler(fx: AuditFixture):
    return lambda trace: (fx.truth[trace.external_id], None)


def judge_all(session, audit: AuditRun, label_fn, *, reviewer: str = "expert", limit: int | None = None):
    judged = []
    while limit is None or len(judged) < limit:
        req = audits.next_audit_review(session, audit, owner=reviewer, lease_seconds=600)
        if req is None:
            break
        trace = session.get(TraceSnapshot, req.trace_id)
        verdict, reason = label_fn(trace)
        judged.append(
            review.submit_judgment(
                session, req.id, verdict=verdict, cannot_judge_reason=reason, reviewer_id=reviewer,
                shown_context_hash=trace.content_hash, idempotency_key=f"j-{req.id}", owner=reviewer,
            )
        )
    return judged


def audit_runs(session, audit: AuditRun) -> dict[str, GradingRun]:
    out: dict[str, GradingRun] = {}
    for r in session.scalars(select(GradingRun).where(GradingRun.audit_run_id == audit.id).order_by(GradingRun.created_at)):
        out[r.trace_id] = r
    return out


def assignment_for(session, project: Project, group_id: str) -> PartitionAssignment:
    return session.scalar(
        select(PartitionAssignment).where(
            PartitionAssignment.project_id == project.id, PartitionAssignment.group_id == group_id
        )
    )


# ----------------------------------------------------------------- locking


def test_lock_freezes_seals_queues_blind_review_and_grading(db_session, session_factory, settings):
    fx = build_project(db_session, settings)
    audit, job = lock(db_session, fx)
    manifest = GraderManifest.from_dict(fx.truthful_grader.manifest)
    assert audit.state == AuditState.IN_REVIEW
    assert audit.grader_id == fx.truthful_grader.id
    assert audit.pipeline_hash == pipeline_hash(manifest)
    assert audit.policy_epoch == fx.project.policy_epoch == 1
    assert len(audit.locked_sample_ids) == 20 and audit.locked_sample_ids == sorted(audit.locked_sample_ids)
    assert audit.sampling_plan["fresh_groups"] is True
    assert audit.sampling_plan["use_cache"] is False
    assert audit.sampling_plan["seed"] == 7 and audit.sampling_plan["planned_n"] == 20
    assert audit.sampling_plan["eligible_group_count"] == 40
    assert audit.risk_targets == {
        **RISK_TARGETS, "gate_false_pass_rate": None, "joint_allocation": None, "unresolved_automatic_rule": "block",
    }
    assert audit.population_definition["weighting"] == "group-weighted"
    assert audit.population_definition["source_type"] == "PRODUCTION"
    window = audit.population_definition["time_window"]
    assert window["start"] and window["end"] and window["start"] <= window["end"]
    assert audit.report is None and audit.report_version == 0

    traces = {t.id: t for t in db_session.scalars(select(TraceSnapshot).where(TraceSnapshot.id.in_(audit.locked_sample_ids)))}
    assert len(traces) == 20
    assert len({t.group_id for t in traces.values()}) == 20  # one target response per group
    for t in traces.values():
        assert t.source_type == "PRODUCTION"
        assert assignment_for(db_session, fx.project, t.group_id).exposure_status == ExposureStatus.SEALED
    sealed_events = list(db_session.scalars(select(ExposureEvent).where(
        ExposureEvent.project_id == fx.project.id, ExposureEvent.kind == ExposureKind.AUDIT_SEALED)))
    assert len(sealed_events) == 20 and all(e.reference_id == audit.id for e in sealed_events)

    requests = list(db_session.scalars(select(ReviewRequest).where(ReviewRequest.audit_run_id == audit.id)))
    assert len(requests) == 20
    assert {r.trace_id for r in requests} == set(audit.locked_sample_ids)
    assert all(r.purpose == "AUDIT" and r.selection_category == "AUDIT" and r.batch_id == audit.id for r in requests)
    assert all(r.state == ReviewRequestState.OPEN and not review.reveal_allowed(r) for r in requests)

    assert job.kind == "AUDIT_GRADING" and job.state == JobState.QUEUED
    assert job.payload == {"audit_id": audit.id, "project_id": fx.project.id, "use_cache": False}
    assert job.idempotency_key == f"audit-grading:{audit.id}" and audit.grading_job_id == job.id

    # Idempotent on the caller's key: same audit, same job, no extra requests.
    again, job_again = lock(db_session, fx)
    assert again.id == audit.id and job_again.id == job.id
    assert db_session.scalar(select(func.count()).select_from(ReviewRequest)) == 20
    assert db_session.scalar(select(func.count()).select_from(AuditRun)) == 1

    # Sealed material is invisible to ordinary review and grading purposes.
    assert not set(t.id for t in review.eligible_traces(db_session, fx.project, "AUDIT_RESERVE")) & set(audit.locked_sample_ids)
    assert review.sealed_group_ids(db_session, fx.project.id) == {t.group_id for t in traces.values()}
    runtime = GraderRuntime.build(fx.project, fx.seed_grader, settings=settings)
    some_trace = next(iter(traces.values()))
    for purpose in ("PROBE", "POOL", "BULK"):
        with pytest.raises(SealedMaterial, match="SEALED"):
            grade_trace(db_session, fx.project, runtime, some_trace, purpose=purpose, settings=settings)

    # A second audit with different seed draws from the remaining untouched groups only.
    other, _ = lock(db_session, fx, seed=99, key="audit-2")
    assert not set(other.locked_sample_ids) & set(audit.locked_sample_ids)
    with pytest.raises(AuditError, match="only 0 eligible"):
        lock(db_session, fx, seed=5, planned_n=1, key="audit-3")

    # Machine grading: fresh calls with the frozen grader, tagged with the audit, never exposed as labels.
    assert run_grading(db_session, session_factory, settings) == 2
    job = db_session.get(Job, job.id)
    assert job.state == JobState.SUCCEEDED, job.error
    assert job.result["graded"] == 20 and job.result["use_cache"] is False
    runs = audit_runs(db_session, audit)
    assert set(runs) == set(audit.locked_sample_ids)
    assert all(r.purpose == "AUDIT" and r.status == GradingStatus.OK and r.cache_hit_of is None for r in runs.values())
    assert all(r.grader_id == fx.truthful_grader.id and r.job_id == job.id for r in runs.values())
    usage = db_session.scalar(select(UsageRecord).where(UsageRecord.job_id == job.id))
    assert usage is not None and usage.role == "grading" and usage.calls == 20
    # No human judgment was created by grading; sealed groups stay sealed.
    assert db_session.scalar(select(func.count()).select_from(HumanJudgment)) == 0
    assert all(assignment_for(db_session, fx.project, t.group_id).exposure_status == ExposureStatus.SEALED
               for t in traces.values())
    # Re-running the job is idempotent: nothing is regraded.
    db_session.expire_all()
    from eval_tinder.services import jobs as job_service

    j2 = job_service.enqueue(db_session, kind="AUDIT_GRADING", payload={"audit_id": audit.id}, idempotency_key="rerun")
    db_session.commit()
    run_grading(db_session, session_factory, settings)
    assert db_session.get(Job, j2.id).result["graded"] == 0
    assert db_session.scalar(select(func.count()).select_from(GradingRun).where(GradingRun.audit_run_id == audit.id)) == 20


def test_eligibility_never_shrinks_and_excludes_synthetic_inspected_graded_groups(db_session, settings):
    fx = build_project(db_session, settings, reserve=10)
    with pytest.raises(AuditError, match=r"only 10 eligible .* planned_n=11") as excinfo:
        lock(db_session, fx, planned_n=11)
    assert "never shrunk" in str(excinfo.value)
    assert db_session.scalar(select(func.count()).select_from(AuditRun)) == 0

    population = audits.validate_population({})
    assert len(audits.eligible_audit_targets(db_session, fx.project, population)) == 10

    # Synthetic material never enters an audit, even inside the reserve partition.
    synth_gid = group_ids_for(fx.project, "AUDIT_RESERVE", 1, "synth")[0]
    import_jsonl_sync(db_session, fx.project, json.dumps(make_record(f"{synth_gid}-1", synth_gid, CASES[0], day=3, source="SYNTHETIC")))
    assert assignment_for(db_session, fx.project, synth_gid).partition == "AUDIT_RESERVE"
    targets = audits.eligible_audit_targets(db_session, fx.project, population)
    assert len(targets) == 10 and synth_gid not in {t.group_id for t in targets}

    # A previously inspected group is not fresh.
    review.record_exposure(db_session, fx.project.id, fx.reserve_groups[0], ExposureKind.TRAIN_REVIEW, None)
    targets = audits.eligible_audit_targets(db_session, fx.project, population)
    assert len(targets) == 9 and fx.reserve_groups[0] not in {t.group_id for t in targets}

    # A group that was ever graded (even for an experiment) is not fresh.
    runtime = GraderRuntime.build(fx.project, fx.seed_grader, settings=settings)
    victim = db_session.scalar(select(TraceSnapshot).where(TraceSnapshot.group_id == fx.reserve_groups[1]))
    grade_trace(db_session, fx.project, runtime, victim, purpose="EXPERIMENT", settings=settings)
    targets = audits.eligible_audit_targets(db_session, fx.project, population)
    assert len(targets) == 8 and fx.reserve_groups[1] not in {t.group_id for t in targets}

    # One designated target response per group: the latest revision with the greatest (timestamp, external_id).
    multi = group_ids_for(fx.project, "AUDIT_RESERVE", 1, "multi")[0]
    import_jsonl_sync(db_session, fx.project, "\n".join([
        json.dumps(make_record(f"{multi}-a", multi, CASES[0], day=5)),
        json.dumps(make_record(f"{multi}-b", multi, CASES[1], day=9)),
        json.dumps(make_record(f"{multi}-c", multi, CASES[2], day=9, timestamp=None)),
    ]))
    by_group = {t.group_id: t for t in audits.eligible_audit_targets(db_session, fx.project, population)}
    assert by_group[multi].external_id == f"{multi}-b"
    import_jsonl_sync(db_session, fx.project, json.dumps({**make_record(f"{multi}-b", multi, CASES[1], day=9), "output": "Revised: your request is processing."}))
    by_group = {t.group_id: t for t in audits.eligible_audit_targets(db_session, fx.project, population)}
    assert by_group[multi].external_id == f"{multi}-b" and by_group[multi].revision == 2 and by_group[multi].is_latest

    # Population filters: task types and time window narrow the eligible set; partition/source are fixed.
    refund_gids = group_ids_for(fx.project, "AUDIT_RESERVE", 3, "refund")
    import_jsonl_sync(db_session, fx.project, "\n".join(
        json.dumps(make_record(f"{g}-1", g, CASES[0], day=20, task_type="refund")) for g in refund_gids))
    refunds = audits.eligible_audit_targets(db_session, fx.project, audits.validate_population({"task_types": ["refund"]}))
    assert {t.group_id for t in refunds} == set(refund_gids)
    early = audits.eligible_audit_targets(db_session, fx.project, audits.validate_population(
        {"time_window": {"start": "2026-03-01T00:00:00Z", "end": "2026-03-03T23:59:59Z"}}))
    assert early and all(1 <= t.timestamp.day <= 3 for t in early)
    with pytest.raises(AuditError, match="synthetic"):
        audits.validate_population({"source_type": "SYNTHETIC"})
    with pytest.raises(AuditError, match="AUDIT_RESERVE"):
        audits.validate_population({"partition": "TRAIN"})
    with pytest.raises(AuditError, match="time_window"):
        audits.validate_population({"time_window": {"start": "yesterday"}})

    # Locking with a filtered population stores the effective definition and only samples inside it.
    audit, _ = lock(db_session, fx, planned_n=3, population={"task_types": ["refund"]}, key="refund-audit")
    assert audit.population_definition["task_types"] == ["refund"]
    sampled = list(db_session.scalars(select(TraceSnapshot).where(TraceSnapshot.id.in_(audit.locked_sample_ids))))
    assert {t.group_id for t in sampled} == set(refund_gids)
    assert audit.population_definition["eligible_group_count"] == 3


def test_risk_targets_and_sampling_plan_must_be_explicit(db_session, settings):
    fx = build_project(db_session, settings, reserve=5)
    with pytest.raises(AuditError) as excinfo:
        lock(db_session, fx, planned_n=2, risk_targets={})
    message = str(excinfo.value)
    for name in ("permitted_verdicts", "max_error_rate", "min_coverage", "confidence"):
        assert name in message
    assert "no default" in message
    with pytest.raises(AuditError, match="joint_allocation"):
        lock(db_session, fx, planned_n=2, risk_targets={**RISK_TARGETS, "gate_false_pass_rate": 0.1})
    with pytest.raises(AuditError, match="unresolved_automatic_rule"):
        lock(db_session, fx, planned_n=2, risk_targets={**RISK_TARGETS, "unresolved_automatic_rule": "ignore"})
    with pytest.raises(AuditError, match="permitted_verdicts"):
        lock(db_session, fx, planned_n=2, risk_targets={**RISK_TARGETS, "permitted_verdicts": ["REVIEW"]})
    with pytest.raises(AuditError, match="max_error_rate"):
        lock(db_session, fx, planned_n=2, risk_targets={**RISK_TARGETS, "max_error_rate": 1.5})
    with pytest.raises(AuditError, match="unknown risk target keys"):
        lock(db_session, fx, planned_n=2, risk_targets={**RISK_TARGETS, "max_error": 0.1})
    with pytest.raises(AuditError, match="unit must be 'group'"):
        lock(db_session, fx, planned_n=2, sampling_plan={**SAMPLING_PLAN, "unit": "trace"})
    with pytest.raises(AuditError, match="use_cache"):
        lock(db_session, fx, planned_n=2, sampling_plan={**SAMPLING_PLAN, "use_cache": True})
    with pytest.raises(AuditError, match="independence_note"):
        lock(db_session, fx, planned_n=2, sampling_plan={**SAMPLING_PLAN, "independence_note": ""})
    with pytest.raises(AuditError, match="planned_n"):
        lock(db_session, fx, planned_n=0)
    assert db_session.scalar(select(func.count()).select_from(AuditRun)) == 0
    assert db_session.scalar(select(func.count()).select_from(Job)) == 0
    assert all(a.exposure_status == ExposureStatus.UNTOUCHED for a in db_session.scalars(select(PartitionAssignment)))

    normalized = audits.validate_risk_targets(
        {**RISK_TARGETS, "gate_false_pass_rate": 0.1, "joint_allocation": "bonferroni",
         "unresolved_automatic_rule": "count_as_error", "permitted_verdicts": ["FAIL", "FAIL"]}
    )
    assert normalized["permitted_verdicts"] == ["FAIL"] and normalized["joint_allocation"] == "bonferroni"


# ----------------------------------------------------------------- fixed sample


def test_sample_is_fixed_skips_are_reported_and_never_replaced(db_session, session_factory, settings):
    fx = build_project(db_session, settings)
    audit, _ = lock(db_session, fx)
    run_grading(db_session, session_factory, settings)
    audit = db_session.get(AuditRun, audit.id)
    judge_all(db_session, audit, truth_labeler(fx), limit=5)
    skipped_req = audits.next_audit_review(db_session, audit, owner="expert", lease_seconds=600)
    review.skip(db_session, skipped_req.id, owner="expert")

    report = audits.compute_report(db_session, audit)
    assert report["kind"] == "AUDIT_EVIDENCE"
    assert report["complete"] is False
    assert (report["planned_n"], report["locked_n"], report["judged_n"], report["skipped_n"], report["pending_n"]) == (20, 20, 5, 1, 14)
    skipped_entries = [e for e in report["human_unresolved"] if e["request_state"] == "SKIPPED"]
    assert len(skipped_entries) == 1 and skipped_entries[0]["trace_id"] == skipped_req.trace_id
    assert "skipped" in skipped_entries[0]["reason"] and "no replacement" in skipped_entries[0]["reason"]
    assert sum(1 for e in report["human_unresolved"] if e["reason"] == "no judgment") == 14
    assert report["gate"]["passed"] is False and "audit_complete" in report["gate"]["failed"]

    audits.persist_report(db_session, audit)
    assert audit.state == AuditState.IN_REVIEW and audit.report_version == 1 and audit.completed_at is None

    # No replacement is drawn: the locked ids and the request set never change.
    assert db_session.scalar(select(func.count()).select_from(ReviewRequest).where(ReviewRequest.audit_run_id == audit.id)) == 20
    served = judge_all(db_session, audit, truth_labeler(fx))
    served_trace_ids = {j.trace_id for j in served}
    assert len(served) == 15 and skipped_req.trace_id in served_trace_ids  # the skipped case came back last
    assert served_trace_ids <= set(audit.locked_sample_ids)
    assert audits.next_audit_review(db_session, audit, owner="expert", lease_seconds=600) is None
    audits.persist_report(db_session, audit)
    assert audit.state == AuditState.COMPLETE and audit.report["complete"] is True and audit.report_version == 2
    assert audit.report_history[0]["report_version"] == 1 and audit.report_history[0]["report"]["complete"] is False
    assert db_session.scalar(select(func.count()).select_from(TraceSnapshot).where(TraceSnapshot.id.in_(audit.locked_sample_ids))) == 20


# ----------------------------------------------------------------- metrics


def test_report_metrics_match_hand_computation(db_session, session_factory, settings):
    fx = build_project(db_session, settings)
    audit, _ = lock(db_session, fx)
    run_grading(db_session, session_factory, settings)
    audit = db_session.get(AuditRun, audit.id)
    runs = audit_runs(db_session, audit)
    assert all(r.verdict in ("PASS", "FAIL") for r in runs.values())  # truthful grader decides every case
    flipped: list[str] = []

    def one_disagreement(trace):
        label = fx.truth[trace.external_id]
        if not flipped and label == "PASS":
            flipped.append(trace.id)
            return "FAIL", None
        return label, None

    judge_all(db_session, audit, one_disagreement)
    assert len(flipped) == 1
    report = audits.compute_report(db_session, audit)
    metrics, table = report["metrics"], report["table"]
    human_pass = sum(1 for tid in audit.locked_sample_ids if fx.truth[db_session.get(TraceSnapshot, tid).external_id] == "PASS") - 1
    human_fail = 20 - human_pass
    assert report["counts"]["human_pass"] == human_pass and report["counts"]["human_fail"] == human_fail
    assert table == {
        "PASS": {"PASS": human_pass, "FAIL": 0, "REVIEW": 0},
        "FAIL": {"PASS": 1, "FAIL": human_fail - 1, "REVIEW": 0},
    }
    assert metrics["agreement"]["value"] == pytest.approx(0.95)
    assert (metrics["agreement"]["numerator"], metrics["agreement"]["denominator"]) == (19, 20)
    assert metrics["automatic_error_rate"]["value"] == pytest.approx(1 / 20)
    assert metrics["automatic_coverage_determinate"]["value"] == 1.0
    assert metrics["false_pass_rate_among_accepted"]["value"] == pytest.approx(1 / (human_pass + 1))
    assert metrics["failure_recall"]["value"] == pytest.approx((human_fail - 1) / human_fail)
    assert report["baselines"]["always_pass"]["agreement"]["value"] == pytest.approx(human_pass / 20)
    assert report["baselines"]["always_pass"]["failure_recall"]["value"] == 0.0
    intervals = report["intervals"]
    assert intervals["supported"] is True and intervals["per_bound_confidence"] == 0.95 and intervals["n_bounds"] == 1
    assert (intervals["automatic_error_rate_k"], intervals["automatic_error_rate_n"]) == (1, 20)
    assert intervals["automatic_error_rate_upper"] == pytest.approx(error_upper_bound(1, 20, 0.95))
    assert (intervals["false_pass_k"], intervals["false_pass_n"]) == (1, human_pass + 1)
    assert intervals["false_pass_rate_upper"] == pytest.approx(error_upper_bound(1, human_pass + 1, 0.95))
    assert report["complete"] is True and report["human_unresolved"] == [] and report["operational_failures"] == []
    # k=1 in n=20 gives a 95% upper bound above 0.2, so the predeclared gate must fail on the bound.
    assert error_upper_bound(1, 20, 0.95) > 0.2
    assert report["gate"]["failed"] == ["automatic_error_rate_bound"]
    assert report["scope"]["weighting"] == "group-weighted" and report["scope"]["pipeline_hash"] == audit.pipeline_hash
    assert any("Development agreement is not audit evidence" in n for n in report["notes"])
    assert any("independent uniform group sampling" in n for n in report["notes"])


def test_zero_errors_gate_and_bonferroni_allocation(db_session, session_factory, settings):
    fx = build_project(db_session, settings)
    audit, _ = lock(db_session, fx)
    both, _ = lock(db_session, fx, seed=11, key="audit-both", risk_targets={
        **RISK_TARGETS, "gate_false_pass_rate": 0.25, "joint_allocation": "bonferroni"})
    run_grading(db_session, session_factory, settings)
    audit, both = db_session.get(AuditRun, audit.id), db_session.get(AuditRun, both.id)
    judge_all(db_session, audit, truth_labeler(fx))
    audits.persist_report(db_session, audit)
    report = audit.report
    assert audit.state == AuditState.COMPLETE and audit.completed_at is not None and audit.report_version == 1
    assert report["metrics"]["agreement"]["value"] == 1.0
    assert report["intervals"]["automatic_error_rate_k"] == 0 and report["intervals"]["automatic_error_rate_n"] == 20
    assert report["intervals"]["automatic_error_rate_upper"] == pytest.approx(1 - 0.05 ** (1 / 20))
    assert report["intervals"]["automatic_error_rate_upper"] == pytest.approx(0.1391, abs=1e-4)
    assert report["gate"]["passed"] is True and report["gate"]["failed"] == []
    assert {c["name"] for c in report["gate"]["checks"]} == {
        "audit_complete", "no_pending_correction", "audit_not_spent", "sampling_design_supported",
        "denominators_estimable", "automatic_error_rate_bound", "automatic_coverage", "unresolved_automatic_decisions",
    }
    # The same evidence fails a stricter predeclared target.
    audit.risk_targets = {**audit.risk_targets, "max_error_rate": 0.1}
    strict = audits.compute_report(db_session, audit)
    assert strict["gate"]["passed"] is False and strict["gate"]["failed"] == ["automatic_error_rate_bound"]

    judge_all(db_session, both, truth_labeler(fx))
    report = audits.compute_report(db_session, both)
    intervals = report["intervals"]
    assert intervals["n_bounds"] == 2 and intervals["joint_allocation"] == "bonferroni"
    assert intervals["per_bound_confidence"] == pytest.approx(bonferroni_confidence(0.95, 2))
    assert intervals["per_bound_confidence"] == pytest.approx(0.975)
    assert intervals["automatic_error_rate_upper"] == pytest.approx(error_upper_bound(0, 20, 0.975))
    assert intervals["false_pass_rate_upper"] == pytest.approx(error_upper_bound(0, intervals["false_pass_n"], 0.975))
    assert intervals["false_pass_k"] == 0 and intervals["false_pass_n"] == report["counts"]["table"]["PASS"]["PASS"]
    # Only a handful of accepted cases: the false-pass bound is honest about that and blocks a tight target.
    assert intervals["false_pass_n"] < 20 and intervals["false_pass_rate_upper"] > 0.25
    assert report["gate"]["failed"] == ["false_pass_rate_bound"]
    fp_check = next(c for c in report["gate"]["checks"] if c["name"] == "false_pass_rate_bound")
    assert f"n={intervals['false_pass_n']}" in fp_check["detail"] and "0.975" in fp_check["detail"]
    both.risk_targets = {**both.risk_targets, "gate_false_pass_rate": 0.5}
    loose = audits.compute_report(db_session, both)
    assert loose["intervals"]["false_pass_rate_upper"] <= 0.5 and loose["gate"]["passed"] is True
    assert {c["name"] for c in loose["gate"]["checks"]} >= {"false_pass_rate_bound", "automatic_error_rate_bound"}


def test_zero_denominators_are_not_estimable(db_session, session_factory, settings):
    fx = build_project(db_session, settings)
    audit, _ = lock(db_session, fx, risk_targets={**RISK_TARGETS, "gate_false_pass_rate": 0.2, "joint_allocation": "bonferroni"})
    run_grading(db_session, session_factory, settings)
    audit = db_session.get(AuditRun, audit.id)
    judge_all(db_session, audit, lambda trace: ("PASS", None))  # the expert saw no failures
    report = audits.compute_report(db_session, audit)
    assert report["counts"]["human_fail"] == 0
    assert report["metrics"]["failure_recall"]["value"] == NOT_ESTIMABLE
    assert report["baselines"]["always_fail"]["failure_recall"]["value"] == NOT_ESTIMABLE
    # Simulate a grader that never accepts: the false-pass rate has no denominator.
    for run in audit_runs(db_session, audit).values():
        run.verdict = "FAIL"
    db_session.flush()
    report = audits.compute_report(db_session, audit)
    assert report["metrics"]["false_pass_rate_among_accepted"]["value"] == NOT_ESTIMABLE
    assert report["intervals"]["false_pass_rate_upper"] == NOT_ESTIMABLE and report["intervals"]["false_pass_n"] == 0
    assert report["metrics"]["automatic_error_rate"]["value"] == 1.0
    assert "denominators_estimable" in report["gate"]["failed"]
    detail = next(c["detail"] for c in report["gate"]["checks"] if c["name"] == "denominators_estimable")
    assert "false_pass_rate" in detail and "NOT_ESTIMABLE" in detail


def test_cannot_judge_unresolved_rules_and_operational_failures(db_session, session_factory, settings):
    fx = build_project(db_session, settings)
    audit, _ = lock(db_session, fx)
    run_grading(db_session, session_factory, settings)
    audit = db_session.get(AuditRun, audit.id)
    unresolved: list[str] = []

    def label(trace):
        if not unresolved:
            unresolved.append(trace.id)
            return "CANNOT_JUDGE", "MISSING_CONTEXT"
        return fx.truth[trace.external_id], None

    judge_all(db_session, audit, label)
    runs = audit_runs(db_session, audit)
    assert runs[unresolved[0]].verdict in ("PASS", "FAIL")
    report = audits.compute_report(db_session, audit)
    assert report["complete"] is True
    assert report["counts"]["human_unresolved"] == 1 and report["counts"]["total_cases"] == 20
    assert report["metrics"]["human_unresolved_rate"]["value"] == pytest.approx(1 / 20)
    entry = next(e for e in report["human_unresolved"] if e["kind"] == "CANNOT_JUDGE")
    assert entry["trace_id"] == unresolved[0] and entry["reason"] == "MISSING_CONTEXT"
    assert report["unresolved_automatic_decisions"] == unresolved
    assert (report["intervals"]["automatic_error_rate_k"], report["intervals"]["automatic_error_rate_n"]) == (0, 19)
    assert report["gate"]["failed"] == ["unresolved_automatic_decisions"]
    assert report["intervals"]["unresolved_counted_as_errors"] == 0

    audit.risk_targets = {**audit.risk_targets, "unresolved_automatic_rule": "count_as_error"}
    counted = audits.compute_report(db_session, audit)
    assert (counted["intervals"]["automatic_error_rate_k"], counted["intervals"]["automatic_error_rate_n"]) == (1, 20)
    assert counted["intervals"]["unresolved_counted_as_errors"] == 1
    assert counted["intervals"]["automatic_error_rate_upper"] == pytest.approx(error_upper_bound(1, 20, 0.95))
    assert next(c for c in counted["gate"]["checks"] if c["name"] == "unresolved_automatic_decisions")["passed"] is True
    assert any("count_as_error" in n for n in counted["notes"])
    assert "unresolved_automatic_decisions" not in counted["gate"]["failed"]

    # Operational failures: an altered run and a missing run both count as REVIEW for coverage.
    determinate = [tid for tid in audit.locked_sample_ids if tid != unresolved[0]]
    broken, missing = determinate[0], determinate[1]
    runs[broken].status = "PROVIDER_ERROR"
    runs[broken].verdict = "REVIEW"
    runs[broken].error = "simulated provider outage"
    db_session.delete(runs[missing])
    db_session.flush()
    report = audits.compute_report(db_session, audit)
    failures = {f["trace_id"]: f for f in report["operational_failures"]}
    assert set(failures) == {broken, missing}
    assert failures[broken]["status"] == "PROVIDER_ERROR" and failures[broken]["effective_verdict"] == "REVIEW"
    assert failures[missing]["status"] == "MISSING_PREDICTION"
    assert report["counts"]["operational_failures"] == 2
    assert report["metrics"]["automatic_coverage_determinate"]["value"] == pytest.approx(17 / 19)
    assert report["metrics"]["automatic_coverage_all"]["value"] == pytest.approx(18 / 20)
    assert report["metrics"]["operational_failure_rate"]["value"] == pytest.approx(2 / 20)
    assert (report["intervals"]["automatic_error_rate_k"], report["intervals"]["automatic_error_rate_n"]) == (1, 18)
    assert any("operational failure" in n for n in report["notes"])


def test_unsupported_sampling_design_is_descriptive_only(db_session, session_factory, settings):
    fx = build_project(db_session, settings)
    audit, _ = lock(db_session, fx, sampling_plan={
        "unit": "group", "method": "uniform_random", "independence_assumption_documented": False, "independence_note": ""})
    assert audit.sampling_plan["independence_assumption_documented"] is False
    run_grading(db_session, session_factory, settings)
    audit = db_session.get(AuditRun, audit.id)
    judge_all(db_session, audit, truth_labeler(fx))
    report = audits.compute_report(db_session, audit)
    assert report["complete"] is True and report["metrics"]["agreement"]["value"] == 1.0
    assert report["intervals"]["supported"] is False
    assert "independence_assumption_documented" in report["intervals"]["reason"]
    assert report["intervals"]["automatic_error_rate_upper"] is None
    assert report["intervals"]["false_pass_rate_upper"] is None
    assert report["gate"]["failed"] == ["sampling_design_supported", "automatic_error_rate_bound"]
    assert any("descriptive metrics only" in n for n in report["notes"])


# ----------------------------------------------------------------- corrections, spending, release


def test_correction_invalidates_report_but_preserves_history(db_session, session_factory, settings):
    fx = build_project(db_session, settings)
    audit, _ = lock(db_session, fx)
    run_grading(db_session, session_factory, settings)
    audit = db_session.get(AuditRun, audit.id)
    judgments = judge_all(db_session, audit, truth_labeler(fx))
    audits.persist_report(db_session, audit)
    assert audit.state == AuditState.COMPLETE and audit.report["gate"]["passed"] is True
    original_report = dict(audit.report)
    target = next(j for j in judgments if j.verdict == "PASS")

    new = audits.correct_audit_judgment(
        db_session, audit, target.id, verdict="FAIL", explanation="on reflection the claim overstates the outcome",
        cannot_judge_reason=None, reviewer_id="expert", idempotency_key="fix-1",
    )
    assert new.supersedes_id == target.id and db_session.get(HumanJudgment, target.id).superseded_by_id == new.id
    assert audit.state == AuditState.INVALIDATED
    assert audit.report == original_report and audit.report_version == 1  # the derived report is preserved
    entry = audit.correction_history[0]
    assert len(audit.correction_history) == 1
    assert (entry["judgment_id"], entry["new_judgment_id"], entry["previous_report_version"]) == (target.id, new.id, 1)
    assert entry["recomputed_in_report_version"] is None and entry["at"]
    pending = audits.compute_report(db_session, audit)
    assert "no_pending_correction" in pending["gate"]["failed"]
    # Replaying the correction changes nothing.
    assert audits.correct_audit_judgment(
        db_session, audit, target.id, verdict="FAIL", explanation="", cannot_judge_reason=None, reviewer_id="expert",
        idempotency_key="fix-1").id == new.id
    assert len(audit.correction_history) == 1 and audit.state == AuditState.INVALIDATED
    with pytest.raises(AuditError, match="not found"):
        audits.correct_audit_judgment(db_session, audit, "nope", verdict="PASS", reviewer_id="expert", idempotency_key="fix-2")
    other, _ = lock(db_session, fx, seed=5, key="audit-other")
    foreign = judge_all(db_session, other, truth_labeler(fx), limit=1)[0]
    with pytest.raises(AuditError, match="does not belong"):
        audits.correct_audit_judgment(db_session, audit, foreign.id, verdict="PASS", reviewer_id="expert", idempotency_key="fix-3")

    audits.persist_report(db_session, audit)
    assert audit.state == AuditState.COMPLETE and audit.report_version == 2
    assert audit.report["intervals"]["automatic_error_rate_k"] == 1  # the corrected label now disagrees with the machine
    assert audit.report["counts"]["human_fail"] == original_report["counts"]["human_fail"] + 1
    assert [e["report_version"] for e in audit.report_history] == [1]
    assert audit.report_history[0]["report"] == original_report and audit.report_history[0]["state"] == "INVALIDATED"
    assert audit.correction_history[0]["recomputed_in_report_version"] == 2
    assert audit.report["gate"]["passed"] is False  # k=1, n=20 exceeds max_error_rate 0.2


def test_spent_audits_stay_historical_and_release_never_untouches(db_session, session_factory, settings):
    fx = build_project(db_session, settings)
    done, _ = lock(db_session, fx)
    ongoing, _ = lock(db_session, fx, seed=3, key="audit-ongoing")
    run_grading(db_session, session_factory, settings)
    done, ongoing = db_session.get(AuditRun, done.id), db_session.get(AuditRun, ongoing.id)
    judge_all(db_session, done, truth_labeler(fx))
    audits.persist_report(db_session, done)
    assert done.state == AuditState.COMPLETE and done.report["gate"]["passed"] is True
    with pytest.raises(AuditError, match="sealed while the audit is in review"):
        audits.release_audit_groups(db_session, ongoing)

    spent = audits.mark_completed_audits_spent(db_session, fx.project, reason="optimization run started")
    assert [a.id for a in spent] == [done.id]
    assert done.state == AuditState.SPENT and ongoing.state == AuditState.IN_REVIEW
    assert done.report["gate"]["passed"] is False and done.report["gate"]["failed"] == ["audit_not_spent"]
    assert done.report["state"] == "SPENT" and done.report_version == 2
    events = [e["event"] for e in done.report_history]
    assert events == ["spent", "report_superseded"]
    assert done.report_history[0]["reason"] == "optimization run started"
    assert done.report_history[1]["report"]["gate"]["passed"] is True  # the original evidence remains
    assert audits.next_audit_review(db_session, done, owner="expert", lease_seconds=60) is None
    audits.mark_spent(db_session, done, reason="again")  # idempotent
    assert done.report_version == 2

    audits.release_audit_groups(db_session, done)
    groups = {t.group_id for t in db_session.scalars(select(TraceSnapshot).where(TraceSnapshot.id.in_(done.locked_sample_ids)))}
    for gid in groups:
        assert assignment_for(db_session, fx.project, gid).exposure_status == ExposureStatus.INSPECTED
    released = list(db_session.scalars(select(ExposureEvent).where(ExposureEvent.kind == ExposureKind.AUDIT_RELEASED)))
    assert {e.group_id for e in released} == groups and all(e.reference_id == done.id for e in released)
    assert not set(t.group_id for t in audits.eligible_audit_targets(db_session, fx.project, audits.validate_population({}))) & groups
    assert done.report_history[-1]["event"] == "released"
    # Ordinary grading purposes still refuse AUDIT_RESERVE material after release.
    runtime = GraderRuntime.build(fx.project, fx.seed_grader, settings=settings)
    trace = db_session.get(TraceSnapshot, done.locked_sample_ids[0])
    with pytest.raises(SealedMaterial):
        grade_trace(db_session, fx.project, runtime, trace, purpose="BULK", settings=settings)
