"""M5 learning experiment: random selection versus committee selection under a fixed labeling budget.

Design
- One isolated project per strategy, same fixture, same partition seed, same bootstrap.
- A simulated expert (the fixture truth table) answers review requests; its
  minutes are a reading-length proxy unless real expert minutes are supplied.
- Each round: select the next TRAIN batch (RANDOM batch, or a committee-driven
  selection round), label it, then run a new optimization round seeded from the
  recommended grader.
- The benchmark is the experiment project's own AUDIT_RESERVE groups with truth
  labels: reserved for this experiment, never a release audit. Results are
  reported as measured, including ties and regressions.
"""
from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from eval_tinder.config import Settings, get_settings
from eval_tinder.db.enums import GradingPurpose, Partition, ReviewPurpose
from eval_tinder.db.models import GraderVersion, Job, OptimizationRun, PartitionAssignment, Project, TraceSnapshot
from eval_tinder.demo import DEMO_JSONL, DEMO_TRUTH, SimulatedExpert, load_truth, seed_demo
from eval_tinder.domain.metrics import build_confusion, compute_metrics
from eval_tinder.gepa.service import OptimizerService
from eval_tinder.services import optimization as opt_service
from eval_tinder.services import selection as selection_service
from eval_tinder.services.grading import GraderRuntime, grade_many
from eval_tinder.services.review import BatchSpec, create_review_batch, resolved_label_counts

log = logging.getLogger(__name__)

STRATEGIES = ("random", "committee")


def _benchmark_traces(session: Session, project: Project) -> list[TraceSnapshot]:
    stmt = (
        select(TraceSnapshot)
        .join(
            PartitionAssignment,
            (PartitionAssignment.project_id == TraceSnapshot.project_id)
            & (PartitionAssignment.group_id == TraceSnapshot.group_id),
        )
        .where(
            TraceSnapshot.project_id == project.id,
            TraceSnapshot.is_latest.is_(True),
            PartitionAssignment.partition == Partition.AUDIT_RESERVE,
        )
        .order_by(TraceSnapshot.group_id, TraceSnapshot.external_id)
    )
    return list(session.scalars(stmt))


def evaluate_on_benchmark(
    session: Session, project: Project, grader: GraderVersion, truth: dict[str, dict[str, Any]], *, settings: Settings
) -> dict[str, Any]:
    traces = [t for t in _benchmark_traces(session, project) if t.external_id in truth]
    runtime = GraderRuntime.build(project, grader, settings=settings)
    runs = grade_many(session, project, runtime, traces, purpose=GradingPurpose.EXPERIMENT, settings=settings)
    rows = [(truth[t.external_id]["verdict"], r.verdict, r.status) for t, r in zip(traces, runs, strict=True)]
    table = build_confusion(rows)
    metrics = compute_metrics(table)
    return {
        "benchmark_size": len(rows),
        "agreement": metrics["agreement"]["value"],
        "failure_recall": metrics["failure_recall"]["value"],
        "false_pass_rate_among_accepted": metrics["false_pass_rate_among_accepted"]["value"],
        "coverage": metrics["automatic_coverage_determinate"]["value"],
        "table": table.as_table(),
        "note": "Experiment benchmark on reserved synthetic groups with truth labels; not a release audit.",
    }


def _run_optimization(
    session_factory, project_id: str, *, optimizer: OptimizerService | None, settings: Settings, max_metric_calls: int,
    label: str, seed: int, grading_lm=None, reflection_lm=None,
) -> dict[str, Any]:
    with session_factory() as s:
        project = s.get(Project, project_id)
        run, _job = opt_service.create_run(
            s, project, opt_service.RunRequest(max_metric_calls=max_metric_calls, label=label, seed=seed),
            idempotency_key=f"exp:{project_id}:{label}:{seed}", settings=settings,
        )
        run_id = run.id
        s.commit()
    result = opt_service.execute_run(
        session_factory, run_id, optimizer=optimizer, settings=settings, grading_lm=grading_lm, reflection_lm=reflection_lm
    )
    return result


def _current_grader(session: Session, project: Project) -> GraderVersion:
    if project.active_shadow_grader_id:
        g = session.get(GraderVersion, project.active_shadow_grader_id)
        if g is not None:
            return g
    last = session.scalar(
        select(OptimizationRun).where(OptimizationRun.project_id == project.id).order_by(OptimizationRun.created_at.desc())
    )
    if last is not None and (last.result_summary or {}).get("recommended_grader_id"):
        return session.get(GraderVersion, last.result_summary["recommended_grader_id"])
    from eval_tinder.services.projects import seed_grader_for

    return seed_grader_for(session, project)


def batch_quotas(batch_size: int) -> dict[str, int]:
    """Scale the default 6/2/2 split to ``batch_size`` so both strategies label the same number per round."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    disagreement = math.ceil(0.6 * batch_size)
    coverage = math.floor(0.2 * batch_size)
    random_slots = batch_size - disagreement - coverage
    return {"disagreement": disagreement, "coverage": coverage, "random": max(0, random_slots)}


def run_experiment(
    session_factory,
    *,
    labeling_budget: int = 32,
    batch_size: int = 10,
    max_metric_calls: int = 60,
    seed: int = 1,
    strategies: tuple[str, ...] = STRATEGIES,
    fixture: Path = DEMO_JSONL,
    truth_path: Path = DEMO_TRUTH,
    optimizer: OptimizerService | None = None,
    settings: Settings | None = None,
    grading_lm=None,
    reflection_lm=None,
    expert_minutes: dict[str, float] | None = None,
    on_progress=None,
) -> dict[str, Any]:
    """Run both strategies under the same labeling budget and report measured results."""
    settings = settings or get_settings()
    truth = load_truth(truth_path)
    rng = random.Random(seed)
    results: dict[str, Any] = {"labeling_budget": labeling_budget, "batch_size": batch_size, "seed": seed, "strategies": {}}
    for strategy in strategies:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}")
        expert = SimulatedExpert(truth)
        with session_factory() as s:
            seeded = seed_demo(
                s, name=f"experiment/{strategy}/{seed}", fixture=fixture, partition_seed=20260910 + seed,
                simulate_expert=False, settings=settings, batch_seed=seed,
                configuration={"review_batch": batch_quotas(batch_size)},
            )
            project = s.get(Project, seeded["project_id"])
            expert.answer_open_requests(s, project)
            s.commit()
            project_id = project.id
        trajectory: list[dict[str, Any]] = []
        round_index = 0
        with session_factory() as s:
            project = s.get(Project, project_id)
            counts = resolved_label_counts(s, project)
            labels_used = counts["TRAIN"]["resolved"] + counts["DEV"]["resolved"]
        opt = _run_optimization(
            session_factory, project_id, optimizer=optimizer, settings=settings, max_metric_calls=max_metric_calls,
            label=f"{strategy}-r0", seed=seed, grading_lm=grading_lm, reflection_lm=reflection_lm,
        )
        with session_factory() as s:
            project = s.get(Project, project_id)
            if opt.get("recommended_grader_id"):
                opt_service.select_shadow(s, project, opt["recommended_grader_id"], reason="experiment", user="experiment")
            trajectory.append({"round": 0, "labels_used": labels_used, "run": {k: opt.get(k) for k in ("state", "seed_agreement", "best_agreement", "recommended_grader_id")}})
            s.commit()
        while labels_used + batch_size <= labeling_budget:
            round_index += 1
            with session_factory() as s:
                project = s.get(Project, project_id)
                if strategy == "committee":
                    rnd, _job = selection_service.create_selection_round(
                        s, project, seed=rng.randrange(1, 2**31 - 1), idempotency_key=f"exp:{project_id}:sel:{round_index}",
                        settings=settings,
                    )
                    round_id = rnd.id
                    s.commit()
                    sel = selection_service.execute_selection_round(session_factory, round_id, settings=settings, lm=grading_lm)
                    committee_size = len(sel.get("committee", []))
                else:
                    create_review_batch(
                        s, project, BatchSpec(purpose=ReviewPurpose.TRAIN, kind="RANDOM", size=batch_size, seed=rng.randrange(1, 2**31 - 1))
                    )
                    committee_size = 0
                    s.commit()
            with session_factory() as s:
                project = s.get(Project, project_id)
                answered = expert.answer_open_requests(s, project, purpose=ReviewPurpose.TRAIN)
                s.commit()
                counts = resolved_label_counts(s, project)
                labels_used = counts["TRAIN"]["resolved"] + counts["DEV"]["resolved"]
            opt = _run_optimization(
                session_factory, project_id, optimizer=optimizer, settings=settings, max_metric_calls=max_metric_calls,
                label=f"{strategy}-r{round_index}", seed=seed + round_index, grading_lm=grading_lm, reflection_lm=reflection_lm,
            )
            with session_factory() as s:
                project = s.get(Project, project_id)
                if opt.get("recommended_grader_id"):
                    opt_service.select_shadow(s, project, opt["recommended_grader_id"], reason="experiment", user="experiment")
                s.commit()
            trajectory.append({
                "round": round_index, "labels_used": labels_used, "answered": answered, "committee_size": committee_size,
                "run": {k: opt.get(k) for k in ("state", "seed_agreement", "best_agreement", "recommended_grader_id")},
            })
            if on_progress is not None:
                on_progress(strategy, round_index, labels_used)
            if answered == 0:
                break
        with session_factory() as s:
            project = s.get(Project, project_id)
            grader = _current_grader(s, project)
            benchmark = evaluate_on_benchmark(s, project, grader, truth, settings=settings)
            s.commit()
        results["strategies"][strategy] = {
            "project_id": project_id,
            "labels_used": labels_used,
            "rounds": round_index,
            "final_grader_id": grader.id,
            "final_grader_origin": grader.origin,
            "simulated_expert_minutes": round(expert.minutes, 2),
            "actual_expert_minutes": (expert_minutes or {}).get(strategy),
            "trajectory": trajectory,
            "benchmark": benchmark,
        }
    results["comparison"] = _compare(results["strategies"])
    return results


def _compare(strategies: dict[str, Any]) -> dict[str, Any]:
    if "random" not in strategies or "committee" not in strategies:
        return {"note": "both strategies are needed for a comparison"}
    a = strategies["random"]["benchmark"]
    b = strategies["committee"]["benchmark"]
    out: dict[str, Any] = {}
    for key in ("agreement", "failure_recall", "false_pass_rate_among_accepted", "coverage"):
        va, vb = a.get(key), b.get(key)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = vb - va
            out[key] = {"random": va, "committee": vb, "delta_committee_minus_random": round(delta, 4),
                        "outcome": "committee_better" if delta > 0 else ("tie" if delta == 0 else "random_better")}
        else:
            out[key] = {"random": va, "committee": vb, "outcome": "not_estimable"}
    out["note"] = (
        "Measured on the reserved experiment benchmark with a simulated expert. Ties and regressions are reported as such; "
        "neither strategy is required to beat the other."
    )
    return out


def experiment_job_handler(job: Job, ctx) -> dict[str, Any]:
    payload = job.payload or {}
    result = run_experiment(
        ctx.session_factory,
        labeling_budget=int(payload.get("labeling_budget", 32)),
        batch_size=int(payload.get("batch_size", 10)),
        max_metric_calls=int(payload.get("max_metric_calls", 60)),
        seed=int(payload.get("seed", 1)),
        settings=ctx.settings,
        on_progress=lambda strategy, r, n: ctx.progress(strategy=strategy, round=r, labels_used=n),
    )
    out_path = Path(ctx.settings.artifact_path) / "experiments" / f"{job.id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1, default=str))
    return {"path": str(out_path), "comparison": result["comparison"]}
