"""Optimization runs: freeze snapshots, run GEPA in a job, persist candidates and DEV evaluations.

Rules
- One run at a time per project (no race to promote).
- TRAIN and DEV snapshots are frozen at creation; labels added later belong to the next run.
- Every returned candidate becomes an immutable GraderVersion with lineage.
- DEV metrics used for comparison are recomputed through the application's own
  grading path (same validation as probes and audits); GEPA's own subscores are
  stored as provenance. Missing scores are unknown, never zero.
- Recommendation rule (product rule, not a significance test): recommend a
  candidate only if agreement improves, false passes do not increase, and
  coverage does not decrease on the same DEV snapshot. Ties keep the incumbent.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from eval_tinder.config import Settings, get_settings
from eval_tinder.db.enums import (
    GraderOrigin,
    GradingPurpose,
    GradingStatus,
    JobKind,
    JobState,
    OptimizationState,
    Partition,
)
from eval_tinder.db.models import (
    CandidateEvaluation,
    DatasetSnapshot,
    GraderVersion,
    Job,
    OptimizationRun,
    Project,
)
from eval_tinder.domain.manifest import METRIC_VERSION, GraderManifest
from eval_tinder.domain.metrics import NOT_ESTIMABLE, baselines, build_confusion, compute_metrics
from eval_tinder.gepa.examples import build_examples
from eval_tinder.gepa.results import OptimizationOutcome, map_subscores_to_traces
from eval_tinder.gepa.service import (
    CancellationToken,
    GepaOptimizerService,
    OptimizerConfig,
    OptimizerService,
    PreflightError,
    preflight,
)
from eval_tinder.llm.budget import BudgetExhausted, BudgetGuard, estimate_cost_usd
from eval_tinder.llm.factory import build_grading_lm, build_reflection_lm
from eval_tinder.services import jobs as job_service
from eval_tinder.services.grading import GraderRuntime, grade_many
from eval_tinder.services.projects import (
    create_grader_version,
    get_grader,
    project_config,
    project_context_for,
    seed_grader_for,
)
from eval_tinder.services.snapshots import assert_disjoint, freeze_snapshot, snapshot_cases, snapshot_rows

log = logging.getLogger(__name__)


class OptimizationError(ValueError):
    pass


@dataclass
class RunRequest:
    max_metric_calls: int | None = None
    reflection_minibatch_size: int | None = None
    num_threads: int | None = None
    seed: int = 0
    seed_grader_id: str | None = None
    label: str = ""
    max_provider_calls: int | None = None
    max_total_tokens: int | None = None
    evaluate_all_candidates: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def _active_run(session: Session, project_id: str) -> OptimizationRun | None:
    return session.scalar(
        select(OptimizationRun).where(
            OptimizationRun.project_id == project_id,
            OptimizationRun.state.in_([OptimizationState.QUEUED, OptimizationState.RUNNING]),
        )
    )


def choose_seed_grader(session: Session, project: Project, explicit_id: str | None) -> tuple[GraderVersion, str]:
    if explicit_id:
        return get_grader(session, explicit_id), f"explicit:{explicit_id}"
    if project.active_shadow_grader_id:
        return get_grader(session, project.active_shadow_grader_id), "active_shadow_grader"
    last = session.scalar(
        select(OptimizationRun)
        .where(OptimizationRun.project_id == project.id, OptimizationRun.state == OptimizationState.SUCCEEDED)
        .order_by(OptimizationRun.created_at.desc())
    )
    if last is not None:
        rec = (last.result_summary or {}).get("recommended_grader_id")
        if rec:
            return get_grader(session, rec), f"previous_best_supported:{last.id}"
    return seed_grader_for(session, project), "generic_seed"


def cost_estimate(config: OptimizerConfig, train_n: int, dev_n: int, settings: Settings, grader_model: str, reflection_model: str) -> dict[str, Any]:
    """An estimate, not a guarantee: metric calls ~ grading calls; reflections ~ metric_calls / (2*minibatch)."""
    grading_calls = config.max_metric_calls + dev_n * 8  # plus DEV re-evaluation of up to ~8 candidates
    reflection_calls = max(1, config.max_metric_calls // max(1, 2 * config.reflection_minibatch_size))
    est = {
        "grading_calls": grading_calls,
        "reflection_calls": reflection_calls,
        "note": "Estimate only. Actual usage is measured and reported separately for grading and reflection.",
    }
    g = estimate_cost_usd(grader_model, {"prompt_tokens": grading_calls * 1500, "completion_tokens": grading_calls * 200}, settings.pricing_table)
    r = estimate_cost_usd(reflection_model, {"prompt_tokens": reflection_calls * 6000, "completion_tokens": reflection_calls * 800}, settings.pricing_table)
    est["cost_usd"] = (g or 0.0) + (r or 0.0) if (g is not None or r is not None) else None
    est["cost_note"] = "Configure PRICING_TABLE to see a cost estimate." if est["cost_usd"] is None else "Based on the configured pricing table and rough token assumptions."
    return est


def create_run(
    session: Session,
    project: Project,
    request: RunRequest,
    *,
    idempotency_key: str,
    settings: Settings | None = None,
) -> tuple[OptimizationRun, Job]:
    settings = settings or get_settings()
    existing_job = session.scalar(select(Job).where(Job.idempotency_key == idempotency_key))
    if existing_job is not None:
        run = session.get(OptimizationRun, existing_job.payload_ref)
        assert run is not None
        return run, existing_job
    active = _active_run(session, project.id)
    if active is not None:
        raise OptimizationError(f"run {active.id} is already {active.state}; wait for it to finish")
    # A completed audit's results can now influence this revision: it becomes SPENT (historical report only).
    from eval_tinder.services import audits as audit_service

    audit_service.mark_completed_audits_spent(
        session, project, reason=f"optimization run requested (idempotency_key={idempotency_key})"
    )
    cfg = project_config(project)
    config = OptimizerConfig(
        max_metric_calls=request.max_metric_calls or cfg.gepa_max_metric_calls,
        reflection_minibatch_size=request.reflection_minibatch_size or cfg.gepa_reflection_minibatch_size,
        num_threads=request.num_threads or cfg.gepa_num_threads,
        seed=request.seed,
    )
    train_snapshot = freeze_snapshot(session, project, Partition.TRAIN)
    dev_snapshot = freeze_snapshot(session, project, Partition.DEV)
    assert_disjoint(train_snapshot, dev_snapshot)
    seed_grader, seed_choice = choose_seed_grader(session, project, request.seed_grader_id)
    context = project_context_for(project, seed_grader)
    train_examples = build_examples(context, snapshot_cases(session, train_snapshot))
    dev_examples = build_examples(context, snapshot_cases(session, dev_snapshot))
    try:
        pre = preflight(train_examples, dev_examples, config)
    except PreflightError as e:
        raise OptimizationError(f"preflight failed: {e}") from e
    manifest = GraderManifest.from_dict(seed_grader.manifest)
    budgets = {
        "max_metric_calls": config.max_metric_calls,
        "max_provider_calls": request.max_provider_calls or settings.max_provider_calls_per_job,
        "max_total_tokens": request.max_total_tokens or settings.max_total_tokens_per_job,
        "max_tokens_per_call": settings.max_tokens_per_call,
        "estimate": cost_estimate(config, len(train_examples), len(dev_examples), settings,
                                  manifest.model_config_.model, settings.reflection_model or "fake-reflection"),
    }
    run = OptimizationRun(
        project_id=project.id,
        seed_grader_id=seed_grader.id,
        seed_choice=seed_choice,
        train_snapshot_id=train_snapshot.id,
        dev_snapshot_id=dev_snapshot.id,
        policy_epoch=project.policy_epoch,
        metric_version=METRIC_VERSION,
        config={**config.as_dict(), "label": request.label, "preflight": pre,
                "evaluate_all_candidates": request.evaluate_all_candidates, **request.extra},
        state=OptimizationState.QUEUED,
        budgets=budgets,
    )
    session.add(run)
    session.flush()
    run.artifact_path = str(Path(settings.artifact_path) / "optimization" / run.id)
    job = job_service.enqueue(
        session, kind=JobKind.OPTIMIZATION, payload={"run_id": run.id, "project_id": project.id},
        idempotency_key=idempotency_key, project_id=project.id, payload_ref=run.id, max_attempts=1,
    )
    run.job_id = job.id
    session.flush()
    return run, job


# ----------------------------------------------------------------- DEV evaluation


def evaluate_on_dev(
    session: Session,
    project: Project,
    grader: GraderVersion,
    dev_snapshot: DatasetSnapshot,
    *,
    budget: BudgetGuard | None,
    job_id: str | None,
    run_id: str | None = None,
    source: str = "APP",
    lm: Any | None = None,
    gepa_subscores: dict[str, float | None] | None = None,
    settings: Settings | None = None,
) -> CandidateEvaluation:
    """Compute (or return the existing) evaluation of a grader on one frozen DEV snapshot."""
    settings = settings or get_settings()
    existing = session.scalar(
        select(CandidateEvaluation).where(
            CandidateEvaluation.grader_id == grader.id, CandidateEvaluation.dev_snapshot_id == dev_snapshot.id
        )
    )
    if existing is not None and existing.complete:
        return existing
    rows = snapshot_rows(session, dev_snapshot)
    runtime = GraderRuntime.build(project, grader, settings=settings, budget=budget, lm=lm)
    verdicts: dict[str, Any] = dict(existing.verdicts) if existing is not None else {}
    complete = True
    partial_reason = None
    try:
        # Resume an incomplete evaluation: a BUDGET_EXHAUSTED entry is an unknown, not a result, so it is
        # graded again; recorded operational failures (REVIEW votes) are kept as they were.
        pending = [
            (t, j) for t, j in rows
            if t.id not in verdicts or verdicts[t.id]["status"] == GradingStatus.BUDGET_EXHAUSTED
        ]
        runs = grade_many(
            session, project, runtime, [t for t, _ in pending], purpose=GradingPurpose.DEV_EVALUATION, job_id=job_id,
            settings=settings,
        )
        for (t, _), r in zip(pending, runs, strict=True):
            verdicts[t.id] = {"verdict": r.verdict, "status": r.status, "grading_run_id": r.id}
            if r.status == GradingStatus.BUDGET_EXHAUSTED:
                complete = False
                partial_reason = "budget exhausted"
    except BudgetExhausted as e:
        complete = False
        partial_reason = str(e)
    labels = {t.id: j.verdict for t, j in rows}
    per_case: dict[str, float | None] = {}
    confusion_rows = []
    for t, j in rows:
        v = verdicts.get(t.id)
        if v is None or v["status"] == GradingStatus.BUDGET_EXHAUSTED:
            per_case[t.id] = None
            complete = False
            continue
        per_case[t.id] = float(v["status"] == GradingStatus.OK and v["verdict"] == j.verdict)
        confusion_rows.append((j.verdict, v["verdict"], v["status"]))
    table = build_confusion(confusion_rows)
    metrics = compute_metrics(table)
    agg: dict[str, Any] = {
        "metrics": metrics,
        "baselines": baselines(table),
        "agreement": metrics["agreement"]["value"],
        "false_passes": table.fail_pass,
        "coverage": metrics["automatic_coverage_determinate"]["value"],
        "failure_recall": metrics["failure_recall"]["value"],
        "human_classes": {"PASS": table.human_pass, "FAIL": table.human_fail},
        "insufficient_class_coverage": table.human_pass == 0 or table.human_fail == 0,
        "evaluated_cases": len(confusion_rows),
        "dev_size": len(rows),
        "complete": complete,
        "partial_reason": partial_reason,
        "gepa_val_score": None,
    }
    if gepa_subscores is not None:
        known = [s for s in gepa_subscores.values() if s is not None]
        agg["gepa_val_score"] = (sum(known) / len(known)) if known else None
        agg["gepa_subscores_missing"] = sum(1 for s in gepa_subscores.values() if s is None)
    if existing is None:
        existing = CandidateEvaluation(
            project_id=project.id, run_id=run_id, grader_id=grader.id, dev_snapshot_id=dev_snapshot.id, source=source
        )
        session.add(existing)
    existing.per_case_scores = {**(gepa_subscores and {"gepa": gepa_subscores} or {}), "app": per_case}
    existing.verdicts = verdicts
    existing.aggregate_metrics = agg
    existing.complete = complete
    if run_id is not None:
        existing.run_id = run_id
    session.flush()
    _ = labels
    return existing


# ----------------------------------------------------------------- comparison / recommendation


def compare_evaluations(incumbent: CandidateEvaluation, candidate: CandidateEvaluation) -> dict[str, Any]:
    if incumbent.dev_snapshot_id != candidate.dev_snapshot_id:
        raise OptimizationError("evaluations come from different DEV snapshots; recompute before comparing")
    a, b = incumbent.aggregate_metrics or {}, candidate.aggregate_metrics or {}
    result: dict[str, Any] = {
        "dev_snapshot_id": incumbent.dev_snapshot_id,
        "incumbent": {"grader_id": incumbent.grader_id, "agreement": a.get("agreement"), "false_passes": a.get("false_passes"),
                      "coverage": a.get("coverage"), "failure_recall": a.get("failure_recall")},
        "candidate": {"grader_id": candidate.grader_id, "agreement": b.get("agreement"), "false_passes": b.get("false_passes"),
                      "coverage": b.get("coverage"), "failure_recall": b.get("failure_recall")},
        "insufficient_class_coverage": bool(a.get("insufficient_class_coverage") or b.get("insufficient_class_coverage")),
        "rule": "recommend only if agreement improves, false passes do not increase, and coverage does not decrease; ties keep the incumbent",
    }
    if not (incumbent.complete and candidate.complete):
        result.update(recommend=False, reason="incomplete DEV evaluation")
        return result
    if a.get("agreement") == NOT_ESTIMABLE or b.get("agreement") == NOT_ESTIMABLE:
        result.update(recommend=False, reason="agreement not estimable")
        return result
    improves = float(b["agreement"]) > float(a["agreement"])
    no_new_false_passes = int(b.get("false_passes", 0)) <= int(a.get("false_passes", 0))
    cov_a, cov_b = a.get("coverage"), b.get("coverage")
    coverage_ok = not (isinstance(cov_a, (int, float)) and isinstance(cov_b, (int, float)) and cov_b < cov_a)
    recommend = improves and no_new_false_passes and coverage_ok
    reasons = []
    if not improves:
        reasons.append("agreement does not improve (ties keep the incumbent)")
    if not no_new_false_passes:
        reasons.append("adds false passes on this DEV snapshot")
    if not coverage_ok:
        reasons.append("coverage decreases")
    if result["insufficient_class_coverage"]:
        reasons.append("insufficient class coverage: DEV has only one human class; failure detection cannot be inferred")
    result.update(recommend=recommend, reason="; ".join(reasons) if reasons else "meets the conservative rule")
    return result


# ----------------------------------------------------------------- execution (worker handler)


class _HeartbeatThread(threading.Thread):
    def __init__(self, ctx, token: CancellationToken, interval: float):
        super().__init__(daemon=True)
        self.ctx, self.token, self.interval = ctx, token, interval
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.ctx.heartbeat(force=True)
                if self.ctx.cancel_requested:
                    self.token.cancel()
            except Exception:  # noqa: BLE001
                log.exception("heartbeat failed")

    def stop(self) -> None:
        self._stop.set()


def execute_run(
    session_factory,
    run_id: str,
    *,
    ctx=None,
    optimizer: OptimizerService | None = None,
    settings: Settings | None = None,
    grading_lm: Any | None = None,
    reflection_lm: Any | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    optimizer = optimizer or GepaOptimizerService()
    with session_factory() as s:
        run = s.get(OptimizationRun, run_id)
        if run is None:
            raise OptimizationError(f"run {run_id} not found")
        if run.state not in (OptimizationState.QUEUED, OptimizationState.RUNNING):
            return {"run_id": run_id, "state": run.state, "note": "already finished"}
        run.state = OptimizationState.RUNNING
        project = s.get(Project, run.project_id)
        seed_grader = s.get(GraderVersion, run.seed_grader_id)
        train_snapshot = s.get(DatasetSnapshot, run.train_snapshot_id)
        dev_snapshot = s.get(DatasetSnapshot, run.dev_snapshot_id)
        assert project and seed_grader and train_snapshot and dev_snapshot
        context = project_context_for(project, seed_grader)
        train_examples = build_examples(context, snapshot_cases(s, train_snapshot))
        dev_examples = build_examples(context, snapshot_cases(s, dev_snapshot))
        config = OptimizerConfig(
            max_metric_calls=run.config["max_metric_calls"],
            reflection_minibatch_size=run.config["reflection_minibatch_size"],
            num_threads=run.config["num_threads"],
            seed=run.config.get("seed", 0),
        )
        manifest = GraderManifest.from_dict(seed_grader.manifest)
        budgets = dict(run.budgets)
        run_dir = Path(run.artifact_path or (Path(settings.artifact_path) / "optimization" / run.id))
        evaluate_all = bool(run.config.get("evaluate_all_candidates"))
        s.commit()

    budget = BudgetGuard(
        max_calls=int(budgets["max_provider_calls"]),
        max_total_tokens=int(budgets["max_total_tokens"]),
        max_tokens_per_call=int(budgets["max_tokens_per_call"]),
    )
    grading = grading_lm or build_grading_lm(manifest.model_config_, settings=settings)
    reflection = reflection_lm or build_reflection_lm(settings=settings)
    token = CancellationToken()
    hb = _HeartbeatThread(ctx, token, interval=max(1.0, ctx.lease_seconds / 4)) if ctx is not None else None
    if hb:
        hb.start()
    started = time.perf_counter()
    try:
        outcome = optimizer.run(
            seed_instructions=manifest.instruction_text, train=train_examples, dev=dev_examples,
            grading_lm=grading, reflection_lm=reflection, config=config, run_dir=run_dir, budget=budget,
            cancellation=token,
        )
    except PreflightError as e:
        with session_factory() as s:
            run = s.get(OptimizationRun, run_id)
            run.state = OptimizationState.FAILED
            run.error = f"preflight: {e}"
            s.commit()
        raise
    except Exception as e:  # noqa: BLE001
        with session_factory() as s:
            run = s.get(OptimizationRun, run_id)
            run.state = OptimizationState.FAILED
            run.error = f"{type(e).__name__}: {e}"[:4000]
            run.usage = budget.snapshot()
            s.commit()
        raise
    finally:
        if hb:
            hb.stop()
    elapsed = time.perf_counter() - started

    with session_factory() as s:
        run = s.get(OptimizationRun, run_id)
        project = s.get(Project, run.project_id)
        seed_grader = s.get(GraderVersion, run.seed_grader_id)
        dev_snapshot = s.get(DatasetSnapshot, run.dev_snapshot_id)
        summary = _persist_outcome(
            s, project, run, seed_grader, dev_snapshot, outcome, budget=budget, job_id=ctx.job_id if ctx else None,
            settings=settings, evaluate_all=evaluate_all, lm=grading,
        )
        summary["elapsed_seconds"] = round(elapsed, 1)
        run.usage = budget.snapshot()
        run.result_summary = summary
        if outcome.partial and "budget" in (outcome.partial_reason or "").lower():
            run.state = OptimizationState.BUDGET_EXHAUSTED
        elif outcome.partial and "cancel" in (outcome.partial_reason or "").lower():
            run.state = OptimizationState.CANCELLED
        elif summary.get("improved"):
            run.state = OptimizationState.SUCCEEDED
        else:
            run.state = OptimizationState.NO_IMPROVEMENT
        run.error = outcome.partial_reason
        from eval_tinder.ids import utcnow

        run.finished_at = utcnow()
        s.commit()
        return {"run_id": run_id, "state": run.state, **summary}


def _persist_outcome(
    s: Session,
    project: Project,
    run: OptimizationRun,
    seed_grader: GraderVersion,
    dev_snapshot: DatasetSnapshot,
    outcome: OptimizationOutcome,
    *,
    budget: BudgetGuard,
    job_id: str | None,
    settings: Settings,
    evaluate_all: bool,
    lm: Any,
) -> dict[str, Any]:
    dev_ids = list(dev_snapshot.ordered_trace_ids)
    index_to_grader: dict[int, GraderVersion] = {}
    for cand in outcome.candidates:
        if cand.index == 0 and cand.instruction_text == seed_grader.instruction_text:
            index_to_grader[0] = seed_grader
            continue
        parents = [index_to_grader[p].id for p in cand.parent_indices if p in index_to_grader]
        grader = create_grader_version(
            s, project, instruction_text=cand.instruction_text, origin=GraderOrigin.GEPA, parent_ids=parents or [seed_grader.id],
            optimization_run_id=run.id, candidate_index=cand.index,
            label=f"run {run.id[:8]} candidate {cand.index}", model_config=GraderManifest.from_dict(seed_grader.manifest).model_config_,
            immutable_policy_context=seed_grader.immutable_policy_context, settings=settings,
        )
        index_to_grader[cand.index] = grader

    members = set(outcome.member_indices())
    cfg = project_config(project)
    to_evaluate = sorted(members | {0}) if not evaluate_all else [c.index for c in outcome.candidates]
    to_evaluate = to_evaluate[: max(cfg.committee_max_shortlist, 2)] if not evaluate_all else to_evaluate
    evaluations: dict[int, CandidateEvaluation] = {}
    exhausted = False
    for idx in to_evaluate:
        cand = outcome.candidates[idx]
        gepa_scores = map_subscores_to_traces(cand.val_subscores, dev_ids) if cand.val_subscores else None
        if exhausted:
            # record the candidate with GEPA provenance only; evaluation stays incomplete
            ev = s.scalar(select(CandidateEvaluation).where(
                CandidateEvaluation.grader_id == index_to_grader[idx].id, CandidateEvaluation.dev_snapshot_id == dev_snapshot.id))
            if ev is None:
                ev = CandidateEvaluation(project_id=project.id, run_id=run.id, grader_id=index_to_grader[idx].id,
                                         dev_snapshot_id=dev_snapshot.id, per_case_scores={"gepa": gepa_scores or {}},
                                         verdicts={}, aggregate_metrics={"complete": False, "partial_reason": "budget exhausted"},
                                         complete=False, source="GEPA")
                s.add(ev)
                s.flush()
            evaluations[idx] = ev
            continue
        ev = evaluate_on_dev(
            s, project, index_to_grader[idx], dev_snapshot, budget=budget, job_id=job_id, run_id=run.id,
            source="GEPA", lm=lm, gepa_subscores=gepa_scores, settings=settings,
        )
        evaluations[idx] = ev
        if not ev.complete and budget.exhausted:
            exhausted = True
    # candidates not evaluated: store GEPA provenance only (incomplete)
    for cand in outcome.candidates:
        if cand.index in evaluations:
            continue
        gepa_scores = map_subscores_to_traces(cand.val_subscores, dev_ids) if cand.val_subscores else {}
        ev = CandidateEvaluation(
            project_id=project.id, run_id=run.id, grader_id=index_to_grader[cand.index].id, dev_snapshot_id=dev_snapshot.id,
            per_case_scores={"gepa": gepa_scores}, verdicts={},
            aggregate_metrics={"complete": False, "partial_reason": "not evaluated through the application path",
                               "gepa_val_score": cand.val_aggregate_score},
            complete=False, source="GEPA",
        )
        s.add(ev)
        s.flush()
        evaluations[cand.index] = ev

    seed_eval = evaluations.get(0)
    best_idx = None
    best_agreement = None
    for idx, ev in evaluations.items():
        if not ev.complete:
            continue
        agr = (ev.aggregate_metrics or {}).get("agreement")
        if agr == NOT_ESTIMABLE or agr is None:
            continue
        key = (float(agr), -len(outcome.candidates[idx].instruction_text), -idx)
        if best_idx is None or key > (best_agreement, -len(outcome.candidates[best_idx].instruction_text), -best_idx):
            best_idx, best_agreement = idx, float(agr)
    comparison = None
    recommended = None
    improved = False
    if seed_eval is not None and best_idx is not None and best_idx != 0 and seed_eval.complete:
        comparison = compare_evaluations(seed_eval, evaluations[best_idx])
        if comparison["recommend"]:
            recommended = index_to_grader[best_idx].id
            improved = True
    summary = {
        "n_candidates": len(outcome.candidates),
        "candidate_grader_ids": {str(i): g.id for i, g in index_to_grader.items()},
        "instance_best_members": {str(k): v for k, v in outcome.instance_best_members.items()},
        "member_indices": sorted(members),
        "gepa_best_index": outcome.best_index,
        "seed_agreement": (seed_eval.aggregate_metrics or {}).get("agreement") if seed_eval else None,
        "best_index": best_idx,
        "best_agreement": best_agreement,
        "recommended_grader_id": recommended,
        "comparison": comparison,
        "improved": improved,
        "partial": outcome.partial,
        "partial_reason": outcome.partial_reason,
        "total_metric_calls": outcome.total_metric_calls,
        "num_full_val_evals": outcome.num_full_val_evals,
        "usage": budget.snapshot(),
        "evaluated_indices": sorted(evaluations),
        "note": "DEV agreement is a development result on a frozen snapshot, not evidence of production accuracy.",
    }
    return summary


def optimization_job_handler(job: Job, ctx) -> dict[str, Any]:
    result = execute_run(ctx.session_factory, job.payload["run_id"], ctx=ctx, settings=ctx.settings)
    if result.get("state") == OptimizationState.BUDGET_EXHAUSTED:
        # The run is already persisted as BUDGET_EXHAUSTED (partial, no invented scores). Surface the same
        # explicit state on the job: a spent budget must never be reported as a SUCCEEDED job.
        raise BudgetExhausted(result.get("partial_reason") or "provider budget exhausted")
    return result


def cancel_run(session: Session, run: OptimizationRun) -> OptimizationRun:
    """Request cancellation of a run through its job.

    A QUEUED job is cancelled immediately, so its run is finalized here as CANCELLED (it will never execute;
    leaving it QUEUED would also block every later ``create_run`` via ``_active_run``). A RUNNING job only
    gets a cancel request: the worker stops GEPA cooperatively and records the run's final state itself.
    """
    if run.state not in (OptimizationState.QUEUED, OptimizationState.RUNNING):
        return run
    if run.job_id is not None:
        job = job_service.request_cancel(session, run.job_id)
        if job.state == JobState.CANCELLED and run.state == OptimizationState.QUEUED:
            from eval_tinder.ids import utcnow

            run.state = OptimizationState.CANCELLED
            run.error = "cancelled before execution"
            run.finished_at = utcnow()
    session.flush()
    return run


# ----------------------------------------------------------------- shadow selection


def select_shadow(session: Session, project: Project, grader_id: str, *, reason: str, user: str) -> Project:
    grader = get_grader(session, grader_id)
    if grader.project_id != project.id:
        raise OptimizationError("grader belongs to another project")
    from eval_tinder.ids import utcnow

    history = list((project.configuration or {}).get("shadow_history", []))
    history.append({"grader_id": grader_id, "reason": reason, "user": user, "at": utcnow().isoformat(),
                    "previous": project.active_shadow_grader_id})
    project.configuration = {**(project.configuration or {}), "shadow_history": history}
    project.active_shadow_grader_id = grader_id
    session.flush()
    # Enablement is bound to one exact pipeline hash: a different active pipeline invalidates it.
    from eval_tinder.services import automation as automation_service

    automation_service.invalidate_if_pipeline_changed(session, project)
    return project


def clear_shadow(session: Session, project: Project, *, reason: str, user: str) -> Project:
    from eval_tinder.ids import utcnow

    history = list((project.configuration or {}).get("shadow_history", []))
    history.append({"grader_id": None, "reason": reason, "user": user, "at": utcnow().isoformat(),
                    "previous": project.active_shadow_grader_id})
    project.configuration = {**(project.configuration or {}), "shadow_history": history}
    project.active_shadow_grader_id = None
    session.flush()
    return project


def run_readiness(session: Session, project: Project) -> dict[str, Any]:
    """Bootstrap/next-round counters shown in the UI (counts only, not guarantees)."""
    from eval_tinder.services.review import resolved_label_counts

    cfg = project_config(project)
    counts = resolved_label_counts(session, project)
    train, dev = counts["TRAIN"]["resolved"], counts["DEV"]["resolved"]
    last = session.scalar(
        select(OptimizationRun).where(OptimizationRun.project_id == project.id).order_by(OptimizationRun.created_at.desc())
    )
    labels_at_last = 0
    if last is not None:
        ts = session.get(DatasetSnapshot, last.train_snapshot_id)
        labels_at_last = len(ts.ordered_trace_ids) if ts else 0
    new_since = max(0, train - labels_at_last)
    from eval_tinder.services.review import dev_topup_target

    return {
        "resolved_train": train,
        "resolved_dev": dev,
        "bootstrap_train_labels": cfg.bootstrap_train_labels,
        "bootstrap_dev_labels": cfg.bootstrap_dev_labels,
        "bootstrap_ready": train >= cfg.bootstrap_train_labels and dev >= cfg.bootstrap_dev_labels,
        "new_train_labels_since_last_run": new_since,
        "ready_to_optimize_again": last is not None and new_since >= cfg.new_train_labels_per_round,
        "dev_topup_target": dev_topup_target(train, cap=cfg.dev_topup_cap, floor=cfg.bootstrap_dev_labels),
        "last_run_id": last.id if last else None,
        "active_run": (_active_run(session, project.id) or last).id if _active_run(session, project.id) else None,
        "automatic_optimization": cfg.automatic_optimization,
        "note": "These are bootstrap counts, not sample-size guarantees.",
    }


# ----------------------------------------------------------------- opt-in automatic rounds


def maybe_auto_optimize(session: Session, project: Project, *, settings: Settings | None = None) -> OptimizationRun | None:
    """Enqueue a new run when the project opted in and enough new TRAIN labels arrived.

    Off by default (``automatic_optimization=false``). Respects configured budgets
    and never runs while another run is queued or running. Returns the run when one
    was created, else None.
    """
    cfg = project_config(project)
    if not cfg.automatic_optimization:
        return None
    readiness = run_readiness(session, project)
    if not readiness["bootstrap_ready"]:
        return None
    if readiness["last_run_id"] is not None and not readiness["ready_to_optimize_again"]:
        return None
    if _active_run(session, project.id) is not None:
        return None
    from eval_tinder.services.review import resolved_label_counts

    counts = resolved_label_counts(session, project)
    key = f"auto:{project.id}:{project.policy_epoch}:{counts['TRAIN']['resolved']}:{counts['DEV']['resolved']}"
    try:
        run, _job = create_run(session, project, RunRequest(label="automatic"), idempotency_key=key, settings=settings)
    except OptimizationError as e:
        log.info("automatic optimization skipped for project %s: %s", project.id, e)
        return None
    return run
