"""Candidate-driven active review (M3).

A selection round:
1. gathers candidate graders evaluated on the latest frozen DEV snapshot (plus the
   active shadow grader, evaluated on that same snapshot),
2. shortlists them by a quality floor and de-duplicates manifests,
3. runs the shortlist on a fixed random unlabeled TRAIN-only probe and forms a
   small behaviorally diverse committee,
4. scores a stratified random unlabeled TRAIN pool by committee disagreement,
5. selects the next human batch (default 6 disagreement / 2 coverage / 2 random)
   with random slots drawn independently of any model output,
6. creates blind review requests whose selection reasons stay hidden until judged.

The committee selects *questions*; it is never a voting classifier. If fewer than
two usable members exist, the batch falls back to coverage/random exploration.
"""
from __future__ import annotations

import logging
import random
from collections import Counter
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from eval_tinder.config import Settings, get_settings
from eval_tinder.db.enums import GradingPurpose, GradingStatus, JobKind, Partition, ReviewPurpose, SelectionRoundState
from eval_tinder.db.models import (
    CandidateEvaluation,
    DatasetSnapshot,
    GraderVersion,
    Job,
    OptimizationRun,
    Project,
    SelectionRound,
    TraceSnapshot,
)
from eval_tinder.domain.committee import CandidateSummary, form_committee, shortlist_candidates
from eval_tinder.domain.disagreement import gini_disagreement, vote_summary
from eval_tinder.domain.manifest import GraderManifest
from eval_tinder.domain.metrics import NOT_ESTIMABLE
from eval_tinder.domain.rendering import estimate_reading_length
from eval_tinder.domain.selection import STRATEGY_VERSION, PoolCase, build_pool, select_batch, stratum_key
from eval_tinder.llm.budget import BudgetExhausted, BudgetGuard
from eval_tinder.llm.factory import new_budget_guard
from eval_tinder.services import jobs as job_service
from eval_tinder.services.grading import GraderRuntime, grade_many
from eval_tinder.services.optimization import evaluate_on_dev
from eval_tinder.services.projects import project_config
from eval_tinder.services.review import active_judgments, create_requests, eligible_traces, strata_of

log = logging.getLogger(__name__)


class SelectionError(ValueError):
    pass


def create_selection_round(
    session: Session, project: Project, *, seed: int | None, idempotency_key: str, settings: Settings | None = None
) -> tuple[SelectionRound, Job]:
    existing_job = session.scalar(select(Job).where(Job.idempotency_key == idempotency_key))
    if existing_job is not None:
        rnd = session.get(SelectionRound, existing_job.payload_ref)
        assert rnd is not None
        return rnd, existing_job
    active = session.scalar(
        select(SelectionRound).where(
            SelectionRound.project_id == project.id,
            SelectionRound.state.in_([SelectionRoundState.QUEUED, SelectionRoundState.RUNNING]),
        )
    )
    if active is not None:
        job = session.get(Job, active.job_id) if active.job_id else None
        if job is not None and job.state in job_service.TERMINAL_STATES:
            # Bug fix: a round whose job already finished without completing the round (cancelled while QUEUED
            # through the generic job endpoint, or failed after its last lease expired) can never run again, yet it
            # stayed QUEUED/RUNNING and blocked every later round for the project. Finalize it instead of refusing.
            active.state = SelectionRoundState.FAILED
            active.error = f"job {job.id} finished as {job.state} before the round completed; no requests were created"
            session.flush()
        else:
            raise SelectionError(f"selection round {active.id} is already {active.state}")
    rnd = SelectionRound(
        project_id=project.id,
        state=SelectionRoundState.QUEUED,
        strategy_version=STRATEGY_VERSION,
        seed=seed if seed is not None else random.SystemRandom().randrange(1, 2**31 - 1),
    )
    session.add(rnd)
    session.flush()
    job = job_service.enqueue(
        session, kind=JobKind.SELECTION, payload={"round_id": rnd.id, "project_id": project.id},
        idempotency_key=idempotency_key, project_id=project.id, payload_ref=rnd.id, max_attempts=1,
    )
    rnd.job_id = job.id
    session.flush()
    return rnd, job


# ------------------------------------------------------------------ candidate gathering


def latest_dev_snapshot(session: Session, project: Project) -> DatasetSnapshot | None:
    run = session.scalar(
        select(OptimizationRun)
        .where(OptimizationRun.project_id == project.id, OptimizationRun.policy_epoch == project.policy_epoch)
        .where(OptimizationRun.state.in_(["SUCCEEDED", "NO_IMPROVEMENT", "BUDGET_EXHAUSTED", "CANCELLED"]))
        .order_by(OptimizationRun.created_at.desc())
    )
    if run is None:
        return None
    return session.get(DatasetSnapshot, run.dev_snapshot_id)


def _agreement_value(ev: CandidateEvaluation | None) -> float | None:
    if ev is None:
        return None
    value = (ev.aggregate_metrics or {}).get("agreement")
    if value is None or value == NOT_ESTIMABLE:
        return None
    return float(value)


def gather_candidates(
    session: Session, project: Project, dev_snapshot: DatasetSnapshot, *, budget: BudgetGuard, job_id: str | None,
    settings: Settings,
) -> tuple[list[CandidateSummary], dict[str, GraderVersion], list[dict[str, Any]]]:
    """Candidates = graders evaluated on ``dev_snapshot`` plus the active shadow grader (evaluated if needed)."""
    notes: list[dict[str, Any]] = []
    evals = {
        ev.grader_id: ev
        for ev in session.scalars(
            select(CandidateEvaluation).where(CandidateEvaluation.dev_snapshot_id == dev_snapshot.id)
        )
    }
    graders = {g.id: g for g in session.scalars(select(GraderVersion).where(GraderVersion.id.in_(list(evals))))} if evals else {}
    if project.active_shadow_grader_id and project.active_shadow_grader_id not in evals:
        shadow = session.get(GraderVersion, project.active_shadow_grader_id)
        if shadow is not None:
            try:
                ev = evaluate_on_dev(
                    session, project, shadow, dev_snapshot, budget=budget, job_id=job_id, source="SELECTION",
                    settings=settings,
                )
                evals[shadow.id] = ev
                graders[shadow.id] = shadow
                notes.append({"event": "shadow_evaluated_on_dev", "grader_id": shadow.id, "complete": ev.complete})
            except BudgetExhausted as e:
                notes.append({"event": "shadow_evaluation_skipped", "reason": str(e)})
    summaries = []
    for gid, ev in evals.items():
        g = graders.get(gid)
        if g is None:
            continue
        usable = True
        note = ""
        try:
            GraderManifest.from_dict(g.manifest)
        except Exception as e:  # noqa: BLE001
            usable, note = False, f"unsupported manifest: {e}"
        if g.policy_epoch != project.policy_epoch:
            usable, note = False, "grader belongs to a previous policy epoch"
        summaries.append(
            CandidateSummary(
                grader_id=gid, manifest_hash=g.manifest_hash, dev_agreement=_agreement_value(ev),
                dev_complete=bool(ev.complete), prompt_length=len(g.instruction_text), usable=usable, note=note,
            )
        )
    return summaries, graders, notes


# ------------------------------------------------------------------ probe / pool helpers


def _one_per_group_sample(traces: list[TraceSnapshot], size: int, seed: int) -> list[TraceSnapshot]:
    by_group: dict[str, list[TraceSnapshot]] = {}
    for t in sorted(traces, key=lambda t: (t.group_id, t.external_id)):
        by_group.setdefault(t.group_id, []).append(t)
    rng = random.Random(seed)
    picks = [members[rng.randrange(len(members))] for _, members in sorted(by_group.items())]
    rng.shuffle(picks)
    return picks[:size]


def _predictions_map(runs) -> dict[str, str | None]:
    return {r.trace_id: (r.verdict if r.status == GradingStatus.OK else None) for r in runs}


def labeled_strata_counts(session: Session, project: Project) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for t, _j in active_judgments(session, project, Partition.TRAIN):
        counts[stratum_key(strata_of(t))] += 1
    return dict(counts)


def _note_budget_exhaustion(report: dict[str, Any], budget: BudgetGuard, stage: str, **detail: Any) -> None:
    """Record (once) that the provider budget ran out during ``stage``.

    Bug fix: ``grader.runtime.grade`` turns ``BudgetExhausted`` into BUDGET_EXHAUSTED grading runs instead of
    raising, so the ``except BudgetExhausted`` blocks in ``_run_round`` never fire for grading calls and an exhausted
    round used to report full results with ``partial=False``. The guard's own flag is the reliable signal. Grading
    is not aborted: cached verdicts remain valid votes, refused provider calls yield no vote, and a case without two
    valid votes keeps a ``None`` disagreement score (never an invented one).
    """
    if not budget.exhausted or any(str(n.get("event", "")).startswith("budget_exhausted") for n in report["notes"]):
        return
    usage = budget.snapshot()
    report["notes"].append(
        {
            "event": f"budget_exhausted_during_{stage}",
            "reason": (
                f"provider budget exhausted: calls={usage['calls']} (max {budget.max_calls}), "
                f"tokens={usage['total_tokens']} (max {budget.max_total_tokens})"
            ),
            **detail,
        }
    )


# ------------------------------------------------------------------ execution


def execute_selection_round(session_factory, round_id: str, *, ctx=None, settings: Settings | None = None, lm=None) -> dict[str, Any]:
    settings = settings or get_settings()
    with session_factory() as s:
        rnd = s.get(SelectionRound, round_id)
        if rnd is None:
            raise SelectionError(f"selection round {round_id} not found")
        if rnd.state not in (SelectionRoundState.QUEUED, SelectionRoundState.RUNNING):
            return {"round_id": round_id, "state": rnd.state, "note": "already finished"}
        rnd.state = SelectionRoundState.RUNNING
        s.commit()
    try:
        with session_factory() as s:
            rnd = s.get(SelectionRound, round_id)
            project = s.get(Project, rnd.project_id)
            assert project is not None
            result = _run_round(s, project, rnd, ctx=ctx, settings=settings, lm=lm)
            rnd.state = SelectionRoundState.COMPLETE
            s.commit()
            return result
    except Exception as e:  # noqa: BLE001
        with session_factory() as s:
            rnd = s.get(SelectionRound, round_id)
            rnd.state = SelectionRoundState.FAILED
            rnd.error = f"{type(e).__name__}: {e}"[:4000]
            s.commit()
        raise


def _run_round(s: Session, project: Project, rnd: SelectionRound, *, ctx, settings: Settings, lm) -> dict[str, Any]:
    cfg = project_config(project)
    job_id = ctx.job_id if ctx is not None else rnd.job_id
    budget = new_budget_guard(settings)
    report: dict[str, Any] = {"notes": [], "strategy_version": STRATEGY_VERSION, "seed": rnd.seed}
    committee_members: list[str] = []
    graders: dict[str, GraderVersion] = {}
    dev_snapshot = latest_dev_snapshot(s, project)
    if dev_snapshot is None:
        report["notes"].append({"event": "no_optimization_run", "detail": "no candidates yet; coverage/random exploration"})
    else:
        report["dev_snapshot_id"] = dev_snapshot.id
        candidates, graders, notes = gather_candidates(s, project, dev_snapshot, budget=budget, job_id=job_id, settings=settings)
        report["notes"].extend(notes)
        _note_budget_exhaustion(report, budget, "shadow_evaluation")
        shortlist = shortlist_candidates(
            candidates, quality_gap=cfg.committee_quality_gap, max_shortlist=cfg.committee_max_shortlist
        )
        report["shortlist"] = {
            "shortlisted": [c.grader_id for c in shortlist.shortlisted],
            "exclusions": shortlist.exclusions,
            "quality_floor": shortlist.quality_floor,
            "quality_gap": cfg.committee_quality_gap,
            "best_agreement": shortlist.best_agreement,
        }
        if ctx is not None:
            ctx.check_cancelled()
        # ---- probe: fixed random unlabeled TRAIN-only cases; labels are never requested for them
        train_pool = eligible_traces(s, project, Partition.TRAIN)
        probe = _one_per_group_sample(train_pool, cfg.committee_probe_size, rnd.seed + 11)
        rnd.probe_ids = [t.id for t in probe]
        probe_predictions: dict[str, dict[str, str | None]] = {}
        try:
            for c in shortlist.shortlisted:
                runtime = GraderRuntime.build(project, graders[c.grader_id], settings=settings, budget=budget, lm=lm)
                runs = grade_many(s, project, runtime, probe, purpose=GradingPurpose.PROBE, job_id=job_id, settings=settings)
                probe_predictions[c.grader_id] = _predictions_map(runs)
                _note_budget_exhaustion(report, budget, "probe", grader_id=c.grader_id)  # bug fix, see helper
                if ctx is not None:
                    ctx.check_cancelled()
                    ctx.progress(stage="probe", graded=len(probe_predictions))
        except BudgetExhausted:
            _note_budget_exhaustion(report, budget, "probe")
        committee = form_committee(shortlist.shortlisted, probe_predictions, size=cfg.committee_size, min_shared=5)
        committee_members = list(committee.members)
        report["committee"] = {
            "members": committee.members,
            "diversity_claimed": committee.diversity_claimed,
            "reason": committee.reason,
            "log": committee.log,
            "probe_size": len(probe),
        }
    rnd.committee_ids = committee_members
    # ---- pool: stratified random unlabeled TRAIN cases
    pool_traces = eligible_traces(s, project, Partition.TRAIN)
    pool_cases = [
        PoolCase(trace_id=t.id, group_id=t.group_id, strata=strata_of(t), reading_length=estimate_reading_length(t))
        for t in pool_traces
    ]
    pool = build_pool(pool_cases, size=cfg.selection_pool_size, seed=rnd.seed + 23)
    rnd.pool_ids = [p.trace_id for p in pool]
    trace_by_id = {t.id: t for t in pool_traces}
    disagreement: dict[str, float | None] = {}
    all_review: set[str] = set()
    votes_by_trace: dict[str, dict[str, Any]] = {}
    if len(committee_members) >= 2:
        pool_trace_objs = [trace_by_id[p.trace_id] for p in pool]
        votes: dict[str, list[str | None]] = {p.trace_id: [] for p in pool}
        try:
            for gid in committee_members:
                runtime = GraderRuntime.build(project, graders[gid], settings=settings, budget=budget, lm=lm)
                runs = grade_many(s, project, runtime, pool_trace_objs, purpose=GradingPurpose.POOL, job_id=job_id, settings=settings)
                for r in runs:
                    votes[r.trace_id].append(r.verdict if r.status == GradingStatus.OK else None)
                _note_budget_exhaustion(report, budget, "pool", grader_id=gid)  # bug fix, see helper
                if ctx is not None:
                    ctx.check_cancelled()
                    ctx.progress(stage="pool", graded_members=gid)
        except BudgetExhausted:
            _note_budget_exhaustion(report, budget, "pool")
        for tid, vs in votes.items():
            summary = vote_summary(vs)
            score = gini_disagreement(vs)
            disagreement[tid] = score
            votes_by_trace[tid] = {"summary": summary, "disagreement": score}
            if summary.get("all_review"):
                all_review.add(tid)
    else:
        report["notes"].append({"event": "committee_fallback", "detail": "fewer than two usable committee members; coverage/random exploration"})
    quotas = {
        "disagreement": cfg.review_batch.disagreement,
        "coverage": cfg.review_batch.coverage,
        "random": cfg.review_batch.random,
    }
    selection = select_batch(
        pool, disagreement=disagreement, all_review=all_review, labeled_strata_counts=labeled_strata_counts(s, project),
        quotas=quotas, seed=rnd.seed + 37,
    )
    reasons = {}
    for sc in selection.selected:
        reasons[sc.trace_id] = {
            "category": sc.category, "score": sc.score, "rank": sc.rank, **sc.reason,
            "committee_size": len(committee_members), "committee_votes_hidden_until_judged": True,
            "votes": votes_by_trace.get(sc.trace_id, {}).get("summary"),
        }
    picks = [trace_by_id[sc.trace_id] for sc in selection.selected]
    requests = []
    for sc, t in zip(selection.selected, picks, strict=True):
        requests.extend(
            create_requests(
                s, project, [t], purpose=ReviewPurpose.TRAIN, category=sc.category, batch_id=rnd.id,
                selection_round_id=rnd.id, reasons={t.id: reasons[t.id]},
            )
        )
    rnd.scores = {
        "disagreement": disagreement,
        "votes": votes_by_trace,
        "exhausted": selection.exhausted,
        "context_repair": selection.context_repair,
        "selection_log": selection.log,
    }
    rnd.selected_requests = [
        {"request_id": r.id, "trace_id": r.trace_id, "category": r.selection_category, "expected_reading_length": r.expected_reading_length}
        for r in requests
    ]
    rnd.batch_id = rnd.id
    report["usage"] = budget.snapshot()
    report["pool_size"] = len(pool)
    report["batch_size"] = len(requests)
    rnd.committee_report = report
    s.flush()
    return {
        "round_id": rnd.id,
        "committee": committee_members,
        "batch_size": len(requests),
        "exhausted": selection.exhausted,
        "context_repair": selection.context_repair,
        "partial": any(n.get("event", "").startswith("budget_exhausted") for n in report["notes"]),
    }


def selection_job_handler(job: Job, ctx) -> dict[str, Any]:
    return execute_selection_round(ctx.session_factory, job.payload["round_id"], ctx=ctx, settings=ctx.settings)


def round_view(rnd: SelectionRound) -> dict[str, Any]:
    report = rnd.committee_report or {}
    scores = rnd.scores or {}
    return {
        "id": rnd.id,
        "project_id": rnd.project_id,
        "state": rnd.state,
        "strategy_version": rnd.strategy_version,
        "seed": rnd.seed,
        "committee_ids": rnd.committee_ids or [],
        "committee_report": {
            "members": rnd.committee_ids or [],
            "diversity_claimed": (report.get("committee") or {}).get("diversity_claimed", False),
            "reason": (report.get("committee") or {}).get("reason", ""),
            "log": (report.get("committee") or {}).get("log", []),
            "shortlist": report.get("shortlist"),
            "notes": report.get("notes", []),
            "usage": report.get("usage"),
            "dev_snapshot_id": report.get("dev_snapshot_id"),
        },
        "probe_size": len(rnd.probe_ids or []),
        "pool_size": len(rnd.pool_ids or []),
        "selected_requests": rnd.selected_requests or [],
        "exhausted": scores.get("exhausted", {}),
        "context_repair": scores.get("context_repair", []),
        "batch_id": rnd.batch_id,
        "job_id": rnd.job_id,
        "error": rnd.error,
        "created_at": rnd.created_at,
        "note": "Disagreement ranks committee splits; it is not a calibrated error probability.",
    }
