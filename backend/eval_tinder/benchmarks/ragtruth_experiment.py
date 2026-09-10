"""RAGTruth end-to-end experiment: random versus committee selection, learning curve, audit, benchmark.

Runs in-process against the configured provider. Every stage is recorded in
``state.json`` so an interrupted run can resume. Human labels from RAGTruth are
replayed through the ordinary review path (leases, blindness, idempotency).
"""
from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

from sqlalchemy import select

from eval_tinder.benchmarks.ragtruth import PROJECT_DESCRIPTION
from eval_tinder.benchmarks.scorer import load_jsonl, score_benchmark
from eval_tinder.config import Settings, get_settings
from eval_tinder.db.enums import ReviewPurpose
from eval_tinder.db.models import AuditRun, GraderVersion, OptimizationRun, PartitionAssignment, Project
from eval_tinder.demo import SimulatedExpert
from eval_tinder.services import audits, optimization, selection
from eval_tinder.services.imports import import_jsonl_sync
from eval_tinder.services.projects import create_project, seed_grader_for
from eval_tinder.services.review import BatchSpec, create_review_batch, dev_topup_target, resolved_label_counts, submit_judgment
from eval_tinder.worker.handlers import build_handlers
from eval_tinder.worker.main import Worker, drain

log = logging.getLogger(__name__)

DEFAULT_SCHEDULE = (12, 32, 72, 152)  # resolved TRAIN labels before each optimization round
STRATEGIES = ("random", "committee")


class State:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {"strategies": {}, "log": []}

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=str))
        tmp.replace(self.path)

    def strategy(self, name: str) -> dict[str, Any]:
        return self.data["strategies"].setdefault(name, {"rounds": [], "benchmarks": {}, "audit": None})

    def note(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.data["log"].append(f"[{stamp}] {msg}")
        print(f"[{stamp}] {msg}", flush=True)
        self.save()


class HumanLabelReplay(SimulatedExpert):
    """RAGTruth's human labels replayed through the review path (marked as replayed, not as an expert session)."""

    def answer(self, session, request, *, reviewer_id: str = "ragtruth-annotators") -> bool:  # type: ignore[override]
        from eval_tinder.db.models import TraceSnapshot

        trace = session.get(TraceSnapshot, request.trace_id)
        label = self.truth.get(trace.external_id)
        if label is None:
            return False
        submit_judgment(
            session, request.id, verdict=label["verdict"], explanation=f"[RAGTruth human label] {label.get('explanation', '')}",
            cannot_judge_reason=label.get("cannot_judge_reason"), reviewer_id=reviewer_id,
            shown_context_hash=trace.content_hash, active_review_ms=0, idempotency_key=f"rt:{request.id}",
        )
        self.judged += 1
        return True


def _counts(session, project) -> dict[str, int]:
    c = resolved_label_counts(session, project)
    return {"train": c["TRAIN"]["resolved"], "dev": c["DEV"]["resolved"], "train_cannot_judge": c["TRAIN"]["CANNOT_JUDGE"],
            "total_labels": sum(c[p]["resolved"] + c[p]["CANNOT_JUDGE"] for p in ("TRAIN", "DEV"))}


def _current_grader(session, project) -> GraderVersion:
    if project.active_shadow_grader_id:
        g = session.get(GraderVersion, project.active_shadow_grader_id)
        if g is not None:
            return g
    return seed_grader_for(session, project)


def run(
    session_factory,
    *,
    data_dir: Path,
    out_dir: Path,
    strategies: tuple[str, ...] = STRATEGIES,
    schedule: tuple[int, ...] = DEFAULT_SCHEDULE,
    metric_calls: int = 300,
    seed: int = 1,
    settings: Settings | None = None,
    project_config: dict[str, Any] | None = None,
    subsample_name: str = "test_subsample.jsonl",
    full_name: str = "test_full.jsonl",
    audit_planned_n: int = 40,
    score_concurrency: int = 8,
) -> dict[str, Any]:
    settings = settings or get_settings()
    out_dir.mkdir(parents=True, exist_ok=True)
    state = State(out_dir / "state.json")
    truth = json.load(open(data_dir / "train_truth.json"))["labels"]
    test_truth = json.load(open(data_dir / "test_truth.json"))["labels"]
    subsample = load_jsonl(data_dir / subsample_name)
    full = load_jsonl(data_dir / full_name)
    cache_dir = out_dir / "score_cache"
    cache_dir.mkdir(exist_ok=True)
    config = {
        "gepa_max_metric_calls": metric_calls, "gepa_num_threads": 8, "committee_max_shortlist": 6, "committee_size": 4,
        "committee_probe_size": 30, "selection_pool_size": 120,
        "review_batch": {"disagreement": 12, "coverage": 4, "random": 4},
        "bootstrap_train_labels": schedule[0], "bootstrap_dev_labels": 8, "new_train_labels_per_round": 10,
        **(project_config or {}),
    }
    rng = random.Random(seed)

    def score(strategy: str, grader: GraderVersion, which: str) -> dict[str, Any]:
        st = state.strategy(strategy)
        key = f"{which}:{grader.manifest_hash}"
        if key in st["benchmarks"]:
            return st["benchmarks"][key]
        with session_factory() as s:
            project = s.get(Project, st["project_id"])
            g = s.get(GraderVersion, grader.id)
            records = subsample if which == "subsample" else full
            result = score_benchmark(project, g, records, test_truth, name=f"RAGTruth {which}", cache_dir=cache_dir,
                                     settings=settings, concurrency=score_concurrency)
        st["benchmarks"][key] = result
        o = result["overall"]
        state.note(f"{strategy}: {which} score for {grader.label or grader.id[:8]}: agreement={o['agreement']} recall={o['failure_recall']} "
                   f"f1={o['fail_detection']['f1']} coverage={o['coverage']} n={o['n']} ({result['elapsed_seconds']}s)")
        return result

    for strategy in strategies:
        st = state.strategy(strategy)
        expert = HumanLabelReplay(truth)
        # ---------------------------------------------------------------- project + import
        if "project_id" not in st:
            with session_factory() as s:
                project = create_project(s, name=f"RAGTruth / {strategy} / seed {seed}", description=PROJECT_DESCRIPTION,
                                         partition_seed=20260910 + seed, configuration=config, settings=settings)
                batch = import_jsonl_sync(s, project, (data_dir / "train_sample.jsonl").read_bytes(), filename="ragtruth_train_sample.jsonl")
                parts = {}
                for row in s.execute(select(PartitionAssignment.partition, PartitionAssignment.group_id).where(PartitionAssignment.project_id == project.id)):
                    parts[row[0]] = parts.get(row[0], 0) + 1
                s.commit()
                st["project_id"] = project.id
                st["import"] = {"counts": {k: v for k, v in batch.counts.items() if k != "warnings"}, "partitions": parts}
            state.note(f"{strategy}: project {st['project_id']} imported {st['import']['counts']} partitions={parts}")
        pid = st["project_id"]
        # ---------------------------------------------------------------- bootstrap
        if not st.get("bootstrap_done"):
            with session_factory() as s:
                project = s.get(Project, pid)
                create_review_batch(s, project, BatchSpec(purpose=ReviewPurpose.TRAIN, kind="SEED", size=schedule[0], seed=rng.randrange(1, 2**31)))
                create_review_batch(s, project, BatchSpec(purpose=ReviewPurpose.DEV, kind="DEV_RANDOM", size=8, seed=rng.randrange(1, 2**31)))
                expert.answer_open_requests(s, project)
                s.commit()
                st["bootstrap_done"] = True
                st["bootstrap"] = _counts(s, project)
            state.note(f"{strategy}: bootstrap labels {st['bootstrap']}")
        # ---------------------------------------------------------------- rounds
        for round_index, target in enumerate(schedule):
            if any(r.get("round") == round_index and r.get("done") for r in st["rounds"]):
                continue
            rec: dict[str, Any] = {"round": round_index, "target_train": target, "selection_rounds": [], "started": time.time()}
            # (a) reach the TRAIN target
            guard_iterations = 0
            while True:
                with session_factory() as s:
                    project = s.get(Project, pid)
                    counts = _counts(s, project)
                if counts["train"] >= target or guard_iterations >= 12:
                    break
                guard_iterations += 1
                missing = target - counts["train"]
                if strategy == "committee":
                    with session_factory() as s:
                        project = s.get(Project, pid)
                        rnd, _ = selection.create_selection_round(s, project, seed=rng.randrange(1, 2**31), idempotency_key=f"rt:{pid}:sel:{round_index}:{guard_iterations}", settings=settings)
                        rid = rnd.id
                        s.commit()
                    result = selection.execute_selection_round(session_factory, rid, settings=settings)
                    with session_factory() as s:
                        project = s.get(Project, pid)
                        rnd = s.get(type(rnd), rid)
                        view = selection.round_view(rnd, s)
                        answered = expert.answer_open_requests(s, project, purpose=ReviewPurpose.TRAIN)
                        s.commit()
                    rec["selection_rounds"].append({"round_id": rid, "committee": result.get("committee"), "batch_size": result.get("batch_size"),
                                                    "exhausted": result.get("exhausted"), "category_counts": view.get("category_counts"),
                                                    "diversity_claimed": view["committee_report"].get("diversity_claimed"),
                                                    "reason": view["committee_report"].get("reason"), "answered": answered,
                                                    "usage": (view["committee_report"].get("usage") or {}).get("calls")})
                    state.note(f"{strategy} r{round_index}: selection round committee={len(result.get('committee', []))} batch={result.get('batch_size')} answered={answered} exhausted={result.get('exhausted')}")
                else:
                    with session_factory() as s:
                        project = s.get(Project, pid)
                        create_review_batch(s, project, BatchSpec(purpose=ReviewPurpose.TRAIN, kind="RANDOM", size=missing, seed=rng.randrange(1, 2**31)))
                        answered = expert.answer_open_requests(s, project, purpose=ReviewPurpose.TRAIN)
                        s.commit()
                    rec["selection_rounds"].append({"random_batch": missing, "answered": answered})
                    state.note(f"{strategy} r{round_index}: random batch {missing} answered={answered}")
                if answered == 0:
                    break
            # (b) DEV top-up
            with session_factory() as s:
                project = s.get(Project, pid)
                counts = _counts(s, project)
                dev_target = dev_topup_target(counts["train"], cap=40, floor=8)
                if counts["dev"] < dev_target:
                    create_review_batch(s, project, BatchSpec(purpose=ReviewPurpose.DEV, kind="DEV_RANDOM", size=dev_target - counts["dev"], seed=rng.randrange(1, 2**31)))
                    expert.answer_open_requests(s, project, purpose=ReviewPurpose.DEV)
                s.commit()
                rec["labels_before_run"] = _counts(s, project)
            # (c) optimization round
            with session_factory() as s:
                project = s.get(Project, pid)
                run_obj, _ = optimization.create_run(s, project, optimization.RunRequest(max_metric_calls=metric_calls, label=f"{strategy}-r{round_index}", seed=seed + round_index),
                                                     idempotency_key=f"rt:{pid}:run:{round_index}", settings=settings)
                run_id = run_obj.id
                s.commit()
            state.note(f"{strategy} r{round_index}: optimization run {run_id[:8]} with {rec['labels_before_run']}")
            optimization.execute_run(session_factory, run_id, settings=settings)
            with session_factory() as s:
                run_obj = s.get(OptimizationRun, run_id)
                project = s.get(Project, pid)
                cands = []
                for idx_str, gid in sorted((run_obj.result_summary.get("candidate_grader_ids") or {}).items(), key=lambda kv: int(kv[0])):
                    from eval_tinder.db.models import CandidateEvaluation
                    ev = s.scalar(select(CandidateEvaluation).where(CandidateEvaluation.grader_id == gid, CandidateEvaluation.dev_snapshot_id == run_obj.dev_snapshot_id))
                    cands.append({"index": int(idx_str), "grader_id": gid, "agreement": (ev.aggregate_metrics or {}).get("agreement") if ev else None,
                                  "false_passes": (ev.aggregate_metrics or {}).get("false_passes") if ev else None,
                                  "coverage": (ev.aggregate_metrics or {}).get("coverage") if ev else None, "complete": bool(ev and ev.complete)})
                rec["run"] = {"run_id": run_id, "state": run_obj.state, "seed_choice": run_obj.seed_choice, "n_candidates": run_obj.result_summary.get("n_candidates"),
                              "seed_agreement": run_obj.result_summary.get("seed_agreement"), "best_agreement": run_obj.result_summary.get("best_agreement"),
                              "recommended_grader_id": run_obj.result_summary.get("recommended_grader_id"), "comparison": run_obj.result_summary.get("comparison"),
                              "usage": run_obj.usage, "elapsed_seconds": run_obj.result_summary.get("elapsed_seconds"), "candidates": cands,
                              "partial": run_obj.result_summary.get("partial"), "error": run_obj.error}
                incumbent_agreement = run_obj.result_summary.get("seed_agreement")
                refused = [c for c in cands if c["complete"] and isinstance(c["agreement"], float) and isinstance(incumbent_agreement, float)
                           and c["agreement"] > incumbent_agreement and c["grader_id"] != run_obj.result_summary.get("recommended_grader_id") and c["index"] != 0]
                rec["refused_better_on_dev"] = refused
                if run_obj.result_summary.get("recommended_grader_id"):
                    optimization.select_shadow(s, project, run_obj.result_summary["recommended_grader_id"], reason=f"recommended by round {round_index}", user="experiment")
                s.commit()
                current = _current_grader(s, project)
                rec["shadow_after_round"] = current.id
            state.note(f"{strategy} r{round_index}: run {run_obj.state} seed={rec['run']['seed_agreement']} best={rec['run']['best_agreement']} recommended={rec['run']['recommended_grader_id']} refused_better={len(refused)} calls={run_obj.usage.get('calls')}")
            # (d) benchmark the current grader (and any refused-but-better-on-DEV candidate) on the subsample
            with session_factory() as s:
                current = s.get(GraderVersion, rec["shadow_after_round"])
            rec["subsample_score"] = score(strategy, current, "subsample")
            rec["refused_scores"] = []
            for c in refused[:1]:
                with session_factory() as s:
                    g = s.get(GraderVersion, c["grader_id"])
                rec["refused_scores"].append({"grader_id": c["grader_id"], "dev_agreement": c["agreement"], "score": score(strategy, g, "subsample")})
            rec["done"] = True
            rec["elapsed_seconds"] = round(time.time() - rec["started"], 1)
            st["rounds"] = [r for r in st["rounds"] if r.get("round") != round_index] + [rec]
            state.save()
        # ---------------------------------------------------------------- final benchmark on the full test split
        with session_factory() as s:
            project = s.get(Project, pid)
            final = _current_grader(s, project)
            seed_g = seed_grader_for(s, project)
        st["final_grader_id"] = final.id
        st["final_full_score"] = score(strategy, final, "full")
        st["seed_full_score"] = score(strategy, seed_g, "full")
        # ---------------------------------------------------------------- audit on the untouched reserve
        if not st.get("audit"):
            try:
                with session_factory() as s:
                    project = s.get(Project, pid)
                    final = s.get(GraderVersion, st["final_grader_id"])
                    population = {"source_type": "PRODUCTION"}
                    eligible = len(audits.eligible_audit_targets(s, project, audits.validate_population(population)))
                    n = min(audit_planned_n, eligible)
                    audit, job = audits.lock_audit(
                        s, project, grader_id=final.id, planned_n=n, seed=seed, population=population,
                        sampling_plan={"unit": "group", "method": "uniform_random", "independence_assumption_documented": True,
                                       "independence_note": "Each RAGTruth source document is an independent draw from the benchmark train split; all answers to one document share a group and only one designated answer per group is audited."},
                        risk_targets={"permitted_verdicts": ["PASS", "FAIL"], "max_error_rate": 0.15, "min_coverage": 0.8, "confidence": 0.95,
                                      "unresolved_automatic_rule": "block"},
                        idempotency_key=f"rt:{pid}:audit", settings=settings,
                    )
                    aid = audit.id
                    s.commit()
                state.note(f"{strategy}: audit {aid[:8]} locked n={n} of {eligible} eligible groups")
                worker = Worker(build_handlers(), settings=settings, worker_id="rt-audit", session_factory=session_factory)
                drain(worker)
                with session_factory() as s:
                    audit = s.get(AuditRun, aid)
                    project = s.get(Project, pid)
                    while True:
                        req = audits.next_audit_review(s, audit, owner="ragtruth-annotators", lease_seconds=600)
                        if req is None:
                            break
                        if not expert.answer(s, req):
                            from eval_tinder.services.review import skip
                            skip(s, req.id, owner="ragtruth-annotators")
                    audits.persist_report(s, audit)
                    s.commit()
                    report = audit.report or {}
                    st["audit"] = {"audit_id": aid, "state": audit.state, "planned_n": n, "eligible_groups": eligible,
                                   "metrics": {k: v.get("value") for k, v in (report.get("metrics") or {}).items()},
                                   "intervals": report.get("intervals"), "gate": report.get("gate"), "counts": report.get("counts"),
                                   "table": report.get("table")}
                state.note(f"{strategy}: audit {audit.state} gate={ (report.get('gate') or {}).get('passed') } bound={(report.get('intervals') or {}).get('automatic_error_rate_upper')} error={(report.get('metrics') or {}).get('automatic_error_rate', {}).get('value')}")
            except Exception as e:  # noqa: BLE001
                st["audit"] = {"error": f"{type(e).__name__}: {e}"}
                state.note(f"{strategy}: audit failed: {e}")
        state.save()
    state.data["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    state.save()
    return state.data
