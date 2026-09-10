"""Job-kind -> handler registry. Feature areas register their handlers here."""
from __future__ import annotations

from typing import Any

from eval_tinder.db.enums import JobKind
from eval_tinder.db.models import ImportBatch, Job
from eval_tinder.services.imports import run_import
from eval_tinder.services.optimization import optimization_job_handler
from eval_tinder.worker.main import JobContext


def import_handler(job: Job, ctx: JobContext) -> dict[str, Any]:
    with ctx.session() as s:
        batch = s.get(ImportBatch, job.payload["batch_id"])
        if batch is None:
            raise RuntimeError(f"import batch {job.payload['batch_id']} not found")
        batch.state = "RUNNING"
        s.commit()
        try:
            run_import(s, batch)
            s.commit()
        except Exception:
            s.rollback()
            batch = s.get(ImportBatch, job.payload["batch_id"])
            batch.state = "FAILED"
            s.commit()
            raise
        return {"batch_id": batch.id, "counts": batch.counts, "line_errors": len(batch.line_errors or [])}


def build_handlers() -> dict[str, Any]:
    handlers: dict[str, Any] = {
        JobKind.IMPORT: import_handler,
        JobKind.OPTIMIZATION: optimization_job_handler,
    }
    for module_name, attr in (
        ("eval_tinder.services.selection", "selection_job_handler"),
        ("eval_tinder.services.bulk_grading", "bulk_grading_job_handler"),
        ("eval_tinder.services.audits", "audit_grading_job_handler"),
        ("eval_tinder.services.exports", "export_job_handler"),
        ("eval_tinder.experiments.selection_experiment", "experiment_job_handler"),
    ):
        try:
            module = __import__(module_name, fromlist=[attr])
        except ModuleNotFoundError as e:
            if e.name == module_name:
                continue
            raise
        handler = getattr(module, attr, None)
        if handler is not None:
            kind = {
                "selection_job_handler": JobKind.SELECTION,
                "bulk_grading_job_handler": JobKind.BULK_GRADING,
                "audit_grading_job_handler": JobKind.AUDIT_GRADING,
                "export_job_handler": JobKind.EXPORT,
                "experiment_job_handler": JobKind.EXPERIMENT,
            }[attr]
            handlers[kind] = handler
    return handlers
