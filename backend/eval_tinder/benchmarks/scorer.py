"""Score a grader version on an external benchmark file that never enters the application.

Records use the import JSONL shape; ``truth`` maps external_id -> {"verdict", "task_type", ...}.
Provider calls run in a thread pool and every (manifest_hash, external_id) result is cached
on disk, so re-scoring the same grader is free. Metrics come from the same confusion-table
code the audit uses, plus precision/recall/F1 for hallucination (FAIL) detection.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from eval_tinder.config import Settings, get_settings
from eval_tinder.db.models import GraderVersion, Project
from eval_tinder.domain.metrics import NOT_ESTIMABLE, baselines, build_confusion, compute_metrics
from eval_tinder.domain.rendering import render_case
from eval_tinder.llm.budget import BudgetGuard
from eval_tinder.services.grading import GraderRuntime


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class ResultCache:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.data: dict[str, dict[str, Any]] = json.loads(path.read_text()) if path.exists() else {}

    def get(self, key: str) -> dict[str, Any] | None:
        return self.data.get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self.data[key] = value

    def save(self) -> None:
        with self._lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data))
            tmp.replace(self.path)


def _fail_detection(rows: list[tuple[str, str, str]]) -> dict[str, Any]:
    """Precision / recall / F1 for the FAIL class among human-determinate cases (REVIEW counts as a miss)."""
    tp = sum(1 for h, m, s in rows if h == "FAIL" and m == "FAIL" and s == "OK")
    fp = sum(1 for h, m, s in rows if h == "PASS" and m == "FAIL" and s == "OK")
    fn = sum(1 for h, m, s in rows if h == "FAIL" and not (m == "FAIL" and s == "OK"))
    precision = tp / (tp + fp) if tp + fp else NOT_ESTIMABLE
    recall = tp / (tp + fn) if tp + fn else NOT_ESTIMABLE
    f1 = (2 * precision * recall / (precision + recall)) if isinstance(precision, float) and isinstance(recall, float) and (precision + recall) else NOT_ESTIMABLE
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def summarize(rows: list[tuple[str, str, str]]) -> dict[str, Any]:
    table = build_confusion(rows)
    metrics = compute_metrics(table)
    return {
        "n": len(rows),
        "agreement": metrics["agreement"]["value"],
        "coverage": metrics["automatic_coverage_determinate"]["value"],
        "automatic_error_rate": metrics["automatic_error_rate"]["value"],
        "false_pass_rate_among_accepted": metrics["false_pass_rate_among_accepted"]["value"],
        "failure_recall": metrics["failure_recall"]["value"],
        "fail_detection": _fail_detection(rows),
        "table": table.as_table(),
        "operational_failures": table.operational_failures,
        "baselines": baselines(table),
    }


def score_benchmark(
    project: Project,
    grader: GraderVersion,
    records: list[dict[str, Any]],
    truth: dict[str, dict[str, Any]],
    *,
    name: str,
    cache_dir: Path,
    settings: Settings | None = None,
    concurrency: int = 8,
    max_calls: int = 20_000,
) -> dict[str, Any]:
    settings = settings or get_settings()
    cache = ResultCache(Path(cache_dir) / f"scores_{grader.manifest_hash[:16]}.json")
    guard = BudgetGuard(max_calls=max_calls, max_total_tokens=10**9, max_tokens_per_call=settings.max_tokens_per_call)
    runtime = GraderRuntime.build(project, grader, settings=settings, budget=guard)
    started = time.perf_counter()

    def one(rec: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        key = rec["external_id"]
        hit = cache.get(key)
        if hit is not None:
            return key, hit
        case = render_case(
            input_text=rec["input"], output_text=rec["output"], context=rec.get("context"),
            tool_calls=rec.get("tool_calls"), metadata=rec.get("metadata") or {},
        )
        result = runtime.grade_case(case, max_case_chars=settings.max_case_chars)
        value = {"verdict": result.verdict, "status": result.status, "explanation": result.explanation[:400],
                 "evidence": result.evidence[:4], "error": result.error, "latency_ms": result.latency_ms}
        cache.put(key, value)
        return key, value

    results: dict[str, dict[str, Any]] = {}
    pending = [r for r in records if truth.get(r["external_id"], {}).get("verdict") in ("PASS", "FAIL")]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one, r) for r in pending]
        for i, fut in enumerate(as_completed(futures)):
            key, value = fut.result()
            results[key] = value
            if (i + 1) % 50 == 0:
                cache.save()
    cache.save()
    rows_all: list[tuple[str, str, str]] = []
    rows_by_task: dict[str, list[tuple[str, str, str]]] = {}
    rows_by_model: dict[str, list[tuple[str, str, str]]] = {}
    for rec in pending:
        ext = rec["external_id"]
        t = truth[ext]
        row = (t["verdict"], results[ext]["verdict"], results[ext]["status"])
        rows_all.append(row)
        rows_by_task.setdefault(t.get("task_type", "?"), []).append(row)
        rows_by_model.setdefault(t.get("author_model", "?"), []).append(row)
    return {
        "benchmark": name,
        "grader_id": grader.id,
        "grader_label": grader.label,
        "manifest_hash": grader.manifest_hash,
        "overall": summarize(rows_all),
        "by_task": {k: summarize(v) for k, v in sorted(rows_by_task.items())},
        "by_author_model": {k: {"n": len(v), "agreement": summarize(v)["agreement"], "failure_recall": summarize(v)["failure_recall"]} for k, v in sorted(rows_by_model.items())},
        "usage": guard.snapshot(),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "note": "Scored on an external benchmark split that never entered the application (no labels, no selection, no audit).",
    }
