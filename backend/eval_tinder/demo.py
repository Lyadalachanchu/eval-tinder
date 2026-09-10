"""Demo seeding: create a project from the synthetic cancellation fixture.

With ``simulate_expert=True`` the fixture's ground-truth file answers review
requests. Those judgments are stored with reviewer id ``simulated-expert`` and an
explanation prefix ``[SIMULATED]``; they are a demonstration device, not expert
evidence. Real GEPA may learn the distinction, already know it, or fail to
improve; the application reports the measured outcome.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from eval_tinder.config import Settings, get_settings
from eval_tinder.db.enums import ReviewPurpose, ReviewRequestState
from eval_tinder.db.models import Project, ReviewRequest, TraceSnapshot
from eval_tinder.services.imports import import_jsonl_sync
from eval_tinder.services.projects import create_project, project_config
from eval_tinder.services.review import BatchSpec, create_review_batch, submit_judgment

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"
DEMO_JSONL = FIXTURE_DIR / "demo_cancellation.jsonl"
DEMO_TRUTH = FIXTURE_DIR / "demo_cancellation_truth.json"

SIMULATED_REVIEWER = "simulated-expert"


def load_truth(path: Path = DEMO_TRUTH) -> dict[str, dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    return data["labels"] if "labels" in data else data


class SimulatedExpert:
    """Answers review requests from a truth table. Clearly marked as simulated."""

    def __init__(self, truth: dict[str, dict[str, Any]], *, active_ms_per_char: float = 8.0):
        self.truth = truth
        self.active_ms_per_char = active_ms_per_char
        self.judged = 0
        self.minutes = 0.0

    def answer(self, session: Session, request: ReviewRequest, *, reviewer_id: str = SIMULATED_REVIEWER) -> bool:
        trace = session.get(TraceSnapshot, request.trace_id)
        assert trace is not None
        label = self.truth.get(trace.external_id)
        if label is None:
            return False
        active_ms = int(self.active_ms_per_char * (len(trace.input) + len(trace.output)))
        submit_judgment(
            session,
            request.id,
            verdict=label["verdict"],
            explanation=f"[SIMULATED] {label.get('explanation', '')}".strip(),
            cannot_judge_reason=label.get("cannot_judge_reason"),
            reviewer_id=reviewer_id,
            shown_context_hash=trace.content_hash,
            active_review_ms=active_ms,
            idempotency_key=f"sim:{request.id}",
        )
        self.judged += 1
        self.minutes += active_ms / 60000.0
        return True

    def answer_open_requests(self, session: Session, project: Project, *, purpose: str | None = None) -> int:
        stmt = select(ReviewRequest).where(
            ReviewRequest.project_id == project.id, ReviewRequest.state == ReviewRequestState.OPEN
        )
        if purpose is not None:
            stmt = stmt.where(ReviewRequest.purpose == purpose)
        n = 0
        for req in list(session.scalars(stmt.order_by(ReviewRequest.created_at))):
            if self.answer(session, req):
                n += 1
        return n


def seed_demo(
    session: Session,
    *,
    name: str = "Cancellation assistant (demo)",
    fixture: Path = DEMO_JSONL,
    partition_seed: int = 20260910,
    simulate_expert: bool = False,
    truth_path: Path = DEMO_TRUTH,
    settings: Settings | None = None,
    batch_seed: int = 7,
    configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    project = create_project(
        session,
        name=name,
        description=(
            "A customer-support assistant for a subscription product. Users ask it to cancel subscriptions; "
            "it can call a cancellation tool and then answers the user. All demo traces are SYNTHETIC."
        ),
        partition_seed=partition_seed,
        configuration=configuration,
        settings=settings,
    )
    batch = import_jsonl_sync(session, project, Path(fixture).read_bytes(), filename=Path(fixture).name)
    cfg = project_config(project)
    train_requests = create_review_batch(
        session, project, BatchSpec(purpose=ReviewPurpose.TRAIN, kind="SEED", size=cfg.bootstrap_train_labels, seed=batch_seed)
    )
    dev_requests = create_review_batch(
        session, project, BatchSpec(purpose=ReviewPurpose.DEV, kind="DEV_RANDOM", size=cfg.bootstrap_dev_labels, seed=batch_seed + 1)
    )
    result: dict[str, Any] = {
        "project_id": project.id,
        "import_counts": batch.counts,
        "line_errors": batch.line_errors,
        "seed_train_requests": len(train_requests),
        "dev_requests": len(dev_requests),
        "simulated_expert": simulate_expert,
    }
    if simulate_expert:
        expert = SimulatedExpert(load_truth(truth_path))
        result["simulated_judgments"] = expert.answer_open_requests(session, project)
        result["simulated_minutes"] = round(expert.minutes, 2)
    session.flush()
    return result


def random_batch_seed(rng: random.Random | None = None) -> int:
    rng = rng or random.SystemRandom()
    return rng.randrange(1, 2**31 - 1)
