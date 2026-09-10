"""Persisted, leased job queue on PostgreSQL.

States: QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED, BUDGET_EXHAUSTED.
Idempotency keys make enqueue safe to retry. Leases with heartbeats let a
restarted worker reclaim abandoned jobs; finalization is transactional and only
the lease owner may finalize, so a duplicate worker cannot double-publish.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from eval_tinder.db.enums import JobState
from eval_tinder.db.models import Job
from eval_tinder.ids import utcnow

TERMINAL_STATES = {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED, JobState.BUDGET_EXHAUSTED}


class JobError(RuntimeError):
    pass


class LeaseLost(JobError):
    pass


def find_existing(session: Session, idempotency_key: str, *, project_id: str | None, kind: str) -> Job | None:
    """Return the job previously created with this key, or raise when the key belongs elsewhere.

    Idempotency keys are scoped to a project and job kind: replaying a key from another project or
    another kind must never hand back a foreign job.
    """
    existing = session.scalar(select(Job).where(Job.idempotency_key == idempotency_key))
    if existing is None:
        return None
    if existing.project_id != project_id or existing.kind != kind:
        raise JobError(
            f"idempotency_key {idempotency_key!r} was already used for a {existing.kind} job"
            + (" of another project" if existing.project_id != project_id else "")
        )
    return existing


def enqueue(
    session: Session,
    *,
    kind: str,
    payload: dict[str, Any],
    idempotency_key: str,
    project_id: str | None = None,
    payload_ref: str | None = None,
    max_attempts: int = 3,
) -> Job:
    existing = find_existing(session, idempotency_key, project_id=project_id, kind=kind)
    if existing is not None:
        return existing
    job = Job(
        kind=kind, payload=payload, idempotency_key=idempotency_key, project_id=project_id,
        payload_ref=payload_ref, max_attempts=max_attempts, state=JobState.QUEUED,
    )
    session.add(job)
    session.flush()
    return job


def claim_next(session: Session, *, worker_id: str, lease_seconds: int, kinds: list[str] | None = None) -> Job | None:
    """Atomically lease the oldest runnable job (QUEUED and available, or RUNNING with an expired lease)."""
    now = utcnow()
    stmt = (
        select(Job)
        .where(
            or_(
                (Job.state == JobState.QUEUED) & (or_(Job.lease_expiry.is_(None), Job.lease_expiry <= now)),
                (Job.state == JobState.RUNNING) & (Job.lease_expiry <= now),
            )
        )
        .order_by(Job.created_at)
        .with_for_update(skip_locked=True)
        .limit(1)
    )
    if kinds:
        stmt = stmt.where(Job.kind.in_(kinds))
    job = session.scalar(stmt)
    if job is None:
        return None
    if job.attempts >= job.max_attempts and job.state == JobState.RUNNING:
        job.state = JobState.FAILED
        job.error = (job.error or "") + " | lease expired after max attempts"
        job.finished_at = now
        session.flush()
        return None
    job.state = JobState.RUNNING
    job.attempts += 1
    job.lease_owner = worker_id
    job.lease_expiry = now + timedelta(seconds=lease_seconds)
    job.heartbeat_at = now
    job.started_at = job.started_at or now
    session.flush()
    return job


def heartbeat(session: Session, job_id: str, worker_id: str, lease_seconds: int) -> Job:
    job = session.get(Job, job_id, with_for_update=True)
    if job is None or job.lease_owner != worker_id or job.state != JobState.RUNNING:
        raise LeaseLost(f"job {job_id} is no longer leased by {worker_id}")
    now = utcnow()
    job.heartbeat_at = now
    job.lease_expiry = now + timedelta(seconds=lease_seconds)
    session.flush()
    return job


def update_progress(session: Session, job_id: str, worker_id: str, progress: dict[str, Any]) -> None:
    job = session.get(Job, job_id, with_for_update=True)
    if job is None or job.lease_owner != worker_id or job.state != JobState.RUNNING:
        raise LeaseLost(f"job {job_id} is no longer leased by {worker_id}")
    job.progress = {**(job.progress or {}), **progress}
    session.flush()


def finalize(
    session: Session,
    job_id: str,
    worker_id: str,
    state: str,
    *,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> Job:
    """Transactional finalization by the lease owner only."""
    if state not in TERMINAL_STATES:
        raise JobError(f"{state} is not a terminal state")
    job = session.get(Job, job_id, with_for_update=True)
    if job is None:
        raise JobError(f"job {job_id} not found")
    if job.state in TERMINAL_STATES:
        raise LeaseLost(f"job {job_id} already finalized as {job.state}")
    if job.lease_owner != worker_id:
        raise LeaseLost(f"job {job_id} is leased by {job.lease_owner}, not {worker_id}")
    job.state = state
    job.result = result or {}
    job.error = error
    job.finished_at = utcnow()
    job.lease_expiry = None
    session.flush()
    return job


def requeue_for_retry(session: Session, job_id: str, worker_id: str, *, error: str, backoff_seconds: int) -> Job:
    job = session.get(Job, job_id, with_for_update=True)
    if job is None or job.lease_owner != worker_id or job.state != JobState.RUNNING:
        raise LeaseLost(f"job {job_id} is no longer leased by {worker_id}")
    if job.attempts >= job.max_attempts:
        job.state = JobState.FAILED
        job.error = error
        job.finished_at = utcnow()
        job.lease_expiry = None
    else:
        job.state = JobState.QUEUED
        job.error = error
        job.lease_owner = None
        job.lease_expiry = utcnow() + timedelta(seconds=backoff_seconds)
    session.flush()
    return job


def request_cancel(session: Session, job_id: str) -> Job:
    job = session.get(Job, job_id, with_for_update=True)
    if job is None:
        raise JobError(f"job {job_id} not found")
    if job.state == JobState.QUEUED:
        job.state = JobState.CANCELLED
        job.finished_at = utcnow()
    elif job.state == JobState.RUNNING:
        job.cancel_requested = True
    session.flush()
    return job


def is_cancel_requested(session: Session, job_id: str) -> bool:
    job = session.get(Job, job_id)
    return bool(job and job.cancel_requested)
