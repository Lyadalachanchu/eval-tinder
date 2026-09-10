"""Worker loop: lease a job, dispatch to its handler, finalize transactionally.

Handlers receive a ``JobContext`` and return a result dict. They may raise
``BudgetExhausted`` (-> BUDGET_EXHAUSTED), ``JobCancelled`` (-> CANCELLED), or any
exception (-> retry with backoff, then FAILED). A handler must never mark a
partially completed run as complete.
"""
from __future__ import annotations

import logging
import socket
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from eval_tinder.config import Settings, get_settings
from eval_tinder.db.base import get_session_factory
from eval_tinder.db.enums import JobState
from eval_tinder.db.models import Job
from eval_tinder.ids import new_id
from eval_tinder.llm.budget import BudgetExhausted
from eval_tinder.services import jobs as job_service

log = logging.getLogger(__name__)


class JobCancelled(Exception):
    pass


@dataclass
class JobContext:
    job_id: str
    worker_id: str
    settings: Settings
    session_factory: Callable[[], Session]
    lease_seconds: int
    _last_heartbeat: float = field(default_factory=time.monotonic)
    _cancel_flag: threading.Event = field(default_factory=threading.Event)

    def session(self) -> Session:
        return self.session_factory()

    def heartbeat(self, force: bool = False) -> None:
        """Extend the lease and poll for cancellation; cheap to call often."""
        if not force and time.monotonic() - self._last_heartbeat < max(1.0, self.lease_seconds / 4):
            return
        with self.session() as s:
            job_service.heartbeat(s, self.job_id, self.worker_id, self.lease_seconds)
            if job_service.is_cancel_requested(s, self.job_id):
                self._cancel_flag.set()
            s.commit()
        self._last_heartbeat = time.monotonic()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_flag.is_set()

    def check_cancelled(self) -> None:
        self.heartbeat()
        if self.cancel_requested:
            raise JobCancelled(f"job {self.job_id} cancelled")

    def progress(self, **fields: Any) -> None:
        with self.session() as s:
            job_service.update_progress(s, self.job_id, self.worker_id, fields)
            s.commit()


Handler = Callable[[Job, JobContext], dict[str, Any]]


class Worker:
    def __init__(
        self,
        handlers: dict[str, Handler],
        *,
        settings: Settings | None = None,
        worker_id: str | None = None,
        poll_interval: float = 1.0,
        session_factory: Callable[[], Session] | None = None,
    ):
        self.handlers = handlers
        self.settings = settings or get_settings()
        self.worker_id = worker_id or f"{socket.gethostname()}-{new_id()[:8]}"
        self.poll_interval = poll_interval
        self.session_factory = session_factory or get_session_factory()
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run_once(self) -> Job | None:
        """Claim and execute at most one job. Returns the job, or None when nothing was runnable."""
        lease = self.settings.job_lease_seconds
        with self.session_factory() as s:
            job = job_service.claim_next(s, worker_id=self.worker_id, lease_seconds=lease, kinds=list(self.handlers))
            s.commit()
            if job is None:
                return None
            job_id, kind = job.id, job.kind
        ctx = JobContext(
            job_id=job_id, worker_id=self.worker_id, settings=self.settings,
            session_factory=self.session_factory, lease_seconds=lease,
        )
        handler = self.handlers[kind]
        try:
            with self.session_factory() as s:
                job = s.get(Job, job_id)
                assert job is not None
                s.expunge(job)
            result = handler(job, ctx)
            state, error = JobState.SUCCEEDED, None
        except JobCancelled as e:
            result, state, error = {}, JobState.CANCELLED, str(e)
        except BudgetExhausted as e:
            result, state, error = {"partial": True}, JobState.BUDGET_EXHAUSTED, str(e)
        except job_service.LeaseLost as e:
            log.warning("lease lost for job %s: %s", job_id, e)
            return None
        except Exception as e:  # noqa: BLE001 - any handler failure is retried then failed
            err = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"
            log.exception("job %s failed", job_id)
            with self.session_factory() as s:
                try:
                    job_service.requeue_for_retry(s, job_id, self.worker_id, error=err, backoff_seconds=self._backoff(job))
                    s.commit()
                except job_service.LeaseLost:
                    s.rollback()
            return job
        with self.session_factory() as s:
            try:
                job_service.finalize(s, job_id, self.worker_id, state, result=result, error=error)
                s.commit()
            except job_service.LeaseLost as e:
                s.rollback()
                log.warning("could not finalize job %s: %s", job_id, e)
        return job

    @staticmethod
    def _backoff(job: Job) -> int:
        return min(300, 2 ** max(0, job.attempts))

    def run_forever(self) -> None:
        log.info("worker %s started; handlers=%s", self.worker_id, sorted(self.handlers))
        while not self._stop.is_set():
            try:
                job = self.run_once()
            except Exception:  # noqa: BLE001
                log.exception("worker loop error")
                job = None
            if job is None:
                self._stop.wait(self.poll_interval)


def drain(worker: Worker, *, max_jobs: int = 100) -> int:
    """Run jobs until the queue is empty (tests and the demo CLI)."""
    n = 0
    while n < max_jobs:
        if worker.run_once() is None:
            break
        n += 1
    return n
