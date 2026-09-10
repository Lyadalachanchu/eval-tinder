from __future__ import annotations

from datetime import timedelta

from eval_tinder.db.enums import JobState
from eval_tinder.db.models import Job
from eval_tinder.ids import utcnow
from eval_tinder.services import jobs
from eval_tinder.worker.main import JobCancelled, Worker, drain
from eval_tinder.llm.budget import BudgetExhausted


def test_enqueue_is_idempotent(db_session):
    a = jobs.enqueue(db_session, kind="IMPORT", payload={"x": 1}, idempotency_key="k1")
    b = jobs.enqueue(db_session, kind="IMPORT", payload={"x": 2}, idempotency_key="k1")
    assert a.id == b.id
    assert a.payload == {"x": 1}


def test_claim_lease_and_finalize_by_owner_only(db_session):
    job = jobs.enqueue(db_session, kind="IMPORT", payload={}, idempotency_key="k2")
    db_session.commit()
    claimed = jobs.claim_next(db_session, worker_id="w1", lease_seconds=60, kinds=["IMPORT"])
    assert claimed is not None and claimed.id == job.id and claimed.state == JobState.RUNNING
    assert jobs.claim_next(db_session, worker_id="w2", lease_seconds=60) is None  # leased
    try:
        jobs.finalize(db_session, job.id, "w2", JobState.SUCCEEDED)
        raise AssertionError("non-owner finalized the job")
    except jobs.LeaseLost:
        pass
    jobs.finalize(db_session, job.id, "w1", JobState.SUCCEEDED, result={"ok": True})
    db_session.commit()
    assert db_session.get(Job, job.id).state == JobState.SUCCEEDED
    try:
        jobs.finalize(db_session, job.id, "w1", JobState.FAILED)
        raise AssertionError("double finalization")
    except jobs.LeaseLost:
        pass


def test_expired_lease_is_reclaimed_by_restarted_worker(db_session):
    job = jobs.enqueue(db_session, kind="IMPORT", payload={}, idempotency_key="k3")
    db_session.commit()
    first = jobs.claim_next(db_session, worker_id="w1", lease_seconds=60)
    first.lease_expiry = utcnow() - timedelta(seconds=1)
    db_session.commit()
    second = jobs.claim_next(db_session, worker_id="w2", lease_seconds=60)
    assert second is not None and second.id == job.id and second.lease_owner == "w2" and second.attempts == 2
    try:
        jobs.heartbeat(db_session, job.id, "w1", 60)
        raise AssertionError("stale worker heartbeat accepted")
    except jobs.LeaseLost:
        pass


def test_worker_dispatch_states(db_session, session_factory, settings):
    calls = []

    def ok(job, ctx):
        calls.append(job.id)
        ctx.progress(step=1)
        return {"done": True}

    def cancelled(job, ctx):
        raise JobCancelled("stop")

    def exhausted(job, ctx):
        raise BudgetExhausted("out of tokens")

    def broken(job, ctx):
        raise RuntimeError("boom")

    for kind, key in [("A", "a"), ("B", "b"), ("C", "c"), ("D", "d")]:
        jobs.enqueue(db_session, kind=kind, payload={}, idempotency_key=key, max_attempts=2)
    db_session.commit()
    worker = Worker({"A": ok, "B": cancelled, "C": exhausted, "D": broken}, settings=settings,
                    worker_id="wt", session_factory=session_factory)
    n = drain(worker)
    assert n == 4
    states = {j.idempotency_key: (j.state, j.attempts) for j in db_session.query(Job).all()}
    assert states["a"] == (JobState.SUCCEEDED, 1)
    assert states["b"] == (JobState.CANCELLED, 1)
    assert states["c"] == (JobState.BUDGET_EXHAUSTED, 1)
    assert states["d"][0] == JobState.QUEUED and states["d"][1] == 1  # retried with backoff
    failing = db_session.query(Job).filter_by(idempotency_key="d").one()
    assert "boom" in (failing.error or "")
    failing.lease_expiry = utcnow() - timedelta(seconds=1)
    db_session.commit()
    drain(worker)
    db_session.expire_all()
    assert db_session.query(Job).filter_by(idempotency_key="d").one().state == JobState.FAILED
    assert db_session.query(Job).filter_by(idempotency_key="a").one().progress == {"step": 1}


def test_cancel_request_reaches_running_handler(db_session, session_factory, settings):
    def handler(job, ctx):
        with ctx.session() as s:
            jobs.request_cancel(s, job.id)
            s.commit()
        ctx.heartbeat(force=True)
        assert ctx.cancel_requested
        ctx.check_cancelled()
        return {}

    jobs.enqueue(db_session, kind="X", payload={}, idempotency_key="x1")
    db_session.commit()
    worker = Worker({"X": handler}, settings=settings, worker_id="wc", session_factory=session_factory)
    drain(worker)
    assert db_session.query(Job).filter_by(idempotency_key="x1").one().state == JobState.CANCELLED


def test_cancel_queued_job(db_session):
    job = jobs.enqueue(db_session, kind="X", payload={}, idempotency_key="x2")
    jobs.request_cancel(db_session, job.id)
    assert job.state == JobState.CANCELLED
    assert jobs.claim_next(db_session, worker_id="w", lease_seconds=10) is None
