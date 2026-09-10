"""Fixtures for API tests: the migrated test database, a ``TestClient`` over ``create_app()``, and a worker.

``tests/integration/conftest.py`` owns the migrated-database fixtures. It cannot be pulled in through
``pytest_plugins`` (pytest refuses to register the same conftest module twice once both directories are collected
in one run), so its fixture functions are re-exported here and ``test_database_url`` is overridden so this directory
gets its own private database: two session-scoped engines must never drop each other's tables.

The helpers at the bottom build JSONL records whose *groups* land in a chosen partition under the project's
seeded assignment, and look up the fixture's expert label for a trace returned by the API. Expert labels
follow the truthful-reporting policy of ``tests.cases``; they are never derived from machine predictions.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eval_tinder.api.app import create_app
from eval_tinder.config import get_settings, reset_settings_cache
from eval_tinder.domain.partitions import DEFAULT_SPLIT, assign_partition
from eval_tinder.worker.handlers import build_handlers
from eval_tinder.worker.main import Worker, drain
from tests.cases import DEV_CANARY, DEV_CASES, TRAIN_CANARY, TRAIN_CASES, DevCase
from tests.integration.conftest import db_session, migrated_engine, session_factory, settings  # noqa: F401

SEED = 20260910
CASES: list[DevCase] = TRAIN_CASES + DEV_CASES
REVIEWER = "local-expert"  # Settings.reviewer_id default: the identity the API attributes judgments to


@pytest.fixture(scope="session")
def test_database_url() -> str:
    base = os.environ.get("TEST_DATABASE_URL") or get_settings().test_database_url
    return f"{base}_api"


@pytest.fixture
def client(db_session) -> Iterator[TestClient]:  # noqa: F811 - fixture parameters shadow the imported names
    """HTTP client over the real app. Depends on ``db_session`` so every test starts on truncated tables."""
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture
def worker(session_factory, settings) -> Worker:  # noqa: F811 - fixture parameters shadow the imported names
    """A worker with the real handler registry; drain it with ``eval_tinder.worker.main.drain``."""
    return Worker(build_handlers(), settings=settings, worker_id="api-test-worker", session_factory=session_factory)


@pytest.fixture
def api_token(monkeypatch) -> Iterator[str]:
    """Configure ``API_TOKEN`` for the duration of a test (the app reads settings per request)."""
    token = "test-bearer-token-7f3a9c"
    monkeypatch.setenv("API_TOKEN", token)
    reset_settings_cache()
    yield token
    monkeypatch.delenv("API_TOKEN", raising=False)
    reset_settings_cache()


# ---------------------------------------------------------------- record builders


def group_ids_for(partition: str, n: int, *, seed: int = SEED, split: dict[str, float] = DEFAULT_SPLIT) -> list[str]:
    """Deterministically find ``n`` group ids that the seeded hash maps to ``partition`` (never rearranged)."""
    found: list[str] = []
    i = 0
    while len(found) < n:
        candidate = f"grp-{partition.lower()}-{i}"
        if assign_partition(candidate, seed, split) == partition:
            found.append(candidate)
        i += 1
    return found


def record(external_id: str, group_id: str, case: DevCase, **overrides: Any) -> dict[str, Any]:
    """One JSONL record; the context carries the external id so every record's content hash is unique."""
    rec: dict[str, Any] = {
        "external_id": external_id,
        "group_id": group_id,
        "timestamp": "2026-08-12T10:30:00Z",
        "input": case.input,
        "context": {"subscription_id": f"s-{external_id}"},
        "tool_calls": case.tool_calls(),
        "output": case.output,
        "metadata": {"task_type": "cancellation", "language": "en"},
        "source_type": "PRODUCTION",
    }
    rec.update(overrides)
    return rec


def partitioned_records(
    *, train: list[DevCase] = (), dev: list[DevCase] = (), audit: list[DevCase] = ()
) -> dict[str, list[dict[str, Any]]]:
    """One record per case, grouped so the seeded assignment puts it in the wanted partition."""
    out: dict[str, list[dict[str, Any]]] = {}
    for partition, cases in (("TRAIN", list(train)), ("DEV", list(dev)), ("AUDIT_RESERVE", list(audit))):
        groups = group_ids_for(partition, len(cases))
        out[partition] = [record(f"{gid}-{c.key}", gid, c) for gid, c in zip(groups, cases, strict=True)]
    return out


def jsonl(records: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(r) + "\n" for r in records).encode("utf-8")


def expert_case(trace: dict[str, Any]) -> DevCase:
    """The fixture case behind a trace returned by the API, i.e. the expert's truthful-reporting label."""
    status = None
    for call in trace.get("tool_calls") or []:
        status = (call.get("result") or {}).get("status")
    for c in CASES:
        if c.input == trace["input"] and c.output == trace["output"] and c.status == status:
            return c
    raise LookupError(f"no fixture case for trace {trace.get('external_id')!r}")


# ---------------------------------------------------------------- API-level helpers


def create_project(client: TestClient, *, name: str, **body: Any) -> dict[str, Any]:
    payload = {"name": name, "partition_seed": SEED, **body}
    resp = client.post("/projects", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


def upload(client: TestClient, project_id: str, data: bytes, *, key: str, filename: str = "cases.jsonl"):
    files = {"file": (filename, data, "application/x-ndjson")}
    return client.post(f"/projects/{project_id}/imports", files=files, data={"idempotency_key": key})


def import_records(
    client: TestClient, worker: Worker, project_id: str, records: list[dict[str, Any]], *, key: str
) -> dict[str, Any]:
    """Upload, drain the import job, and return the finished import."""
    accepted = upload(client, project_id, jsonl(records), key=key)
    assert accepted.status_code == 202, accepted.text
    assert drain(worker) == 1
    done = client.get(f"/imports/{accepted.json()['id']}")
    assert done.status_code == 200, done.text
    body = done.json()
    assert body["state"] == "SUCCEEDED" and body["line_errors"] == [], body
    assert body["counts"]["inserted"] == len(records), body["counts"]
    return body


def create_batch(client: TestClient, project_id: str, **spec: Any) -> list[dict[str, Any]]:
    resp = client.post(f"/projects/{project_id}/review-batches", json=spec)
    assert resp.status_code == 201, resp.text
    return resp.json()


def next_review(client: TestClient, project_id: str, purpose: str | None = None) -> dict[str, Any] | None:
    params = {"purpose": purpose} if purpose else {}
    resp = client.get(f"/projects/{project_id}/next-review", params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()


def submit(
    client: TestClient,
    case: dict[str, Any],
    *,
    key: str,
    verdict: str | None = None,
    explanation: str = "",
    cannot_judge_reason: str | None = None,
    shown_context_hash: str | None = None,
    active_review_ms: int = 1200,
):
    """Submit a judgment for a review case. Defaults to the fixture's expert label for the shown trace."""
    body = {
        "verdict": verdict or expert_case(case["trace"]).label,
        "explanation": explanation,
        "cannot_judge_reason": cannot_judge_reason,
        "shown_context_hash": shown_context_hash or case["shown_context_hash"],
        "active_review_ms": active_review_ms,
        "idempotency_key": key,
    }
    return client.post(f"/review-requests/{case['request']['id']}/judgments", json=body)


def label_all(client: TestClient, project_id: str, purpose: str) -> list[dict[str, Any]]:
    """Claim and judge every open request of ``purpose`` through the blind review endpoints."""
    canary = TRAIN_CANARY if purpose == "TRAIN" else DEV_CANARY
    judgments = []
    for _ in range(200):
        case = next_review(client, project_id, purpose)
        if case is None:
            break
        c = expert_case(case["trace"])
        resp = submit(
            client, case, key=f"{purpose}-{case['request']['id']}", verdict=c.label,
            explanation=f"{c.explanation} {canary}",
        )
        assert resp.status_code == 201, resp.text
        judgments.append(resp.json())
    else:  # pragma: no cover - guards against an endless claim loop
        raise AssertionError("next-review never ran out of requests")
    return judgments


@dataclass
class SeededProject:
    project: dict[str, Any]
    imported: dict[str, Any]
    train_judgments: list[dict[str, Any]]
    dev_judgments: list[dict[str, Any]]

    @property
    def id(self) -> str:
        return self.project["id"]


def seed_labeled_project(client: TestClient, worker: Worker, *, name: str = "m2") -> SeededProject:
    """A project with 6 TRAIN and 4 DEV truthful-policy labels collected through the API."""
    project = create_project(
        client, name=name, description="A subscription support assistant that cancels plans on request.",
        configuration={"bootstrap_train_labels": len(TRAIN_CASES), "bootstrap_dev_labels": len(DEV_CASES)},
    )
    records = partitioned_records(train=TRAIN_CASES, dev=DEV_CASES)
    imported = import_records(client, worker, project["id"], records["TRAIN"] + records["DEV"], key=f"{name}-import")
    train_batch = create_batch(
        client, project["id"], purpose="TRAIN", kind="SEED", size=len(TRAIN_CASES), seed=1, idempotency_key=f"{name}-seed"
    )
    dev_batch = create_batch(
        client, project["id"], purpose="DEV", kind="DEV_RANDOM", size=len(DEV_CASES), seed=1, idempotency_key=f"{name}-dev"
    )
    assert len(train_batch) == len(TRAIN_CASES) and len(dev_batch) == len(DEV_CASES)
    train_judgments = label_all(client, project["id"], "TRAIN")
    dev_judgments = label_all(client, project["id"], "DEV")
    assert len(train_judgments) == len(TRAIN_CASES) and len(dev_judgments) == len(DEV_CASES)
    return SeededProject(project, imported, train_judgments, dev_judgments)
