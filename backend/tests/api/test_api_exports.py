"""HTTP surface for bulk grading jobs, predictions, exports, and grader bundles."""
from __future__ import annotations

import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from eval_tinder.api.app import create_app
from eval_tinder.worker.handlers import build_handlers
from eval_tinder.worker.main import Worker, drain
from tests.integration.test_bulk_grading import import_partitioned

# The database fixtures (db_session, session_factory, settings) come from tests/api/conftest.py, which
# re-exports the integration fixtures. The client and worker below are local so this module does not depend
# on that conftest's other fixture names.


@pytest.fixture
def export_client(db_session, settings):
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture
def export_worker(session_factory, settings) -> Worker:
    return Worker(build_handlers(), settings=settings, worker_id="export-api-w", session_factory=session_factory)


def test_export_job_lifecycle_and_download(export_client, export_worker, db_session, settings):
    client, worker = export_client, export_worker
    fx = import_partitioned(db_session, settings, train=2, dev=1, reserve=1, truthful_grader=True)
    db_session.commit()
    res = client.post(f"/projects/{fx.project.id}/exports",
                          json={"kind": "FULL", "grader_id": fx.grader.id, "idempotency_key": "api-exp-1"})
    assert res.status_code == 202, res.text
    body = res.json()
    assert body["kind"] == "FULL" and body["state"] == "QUEUED" and body["download_url"] is None
    assert body["project_id"] == fx.project.id and body["job_id"] and body["manifest"] == {}
    export_id = body["id"]
    assert client.get(f"/exports/{export_id}/download").status_code == 404
    replay = client.post(f"/projects/{fx.project.id}/exports",
                             json={"kind": "FULL", "grader_id": fx.grader.id, "idempotency_key": "api-exp-1"})
    assert replay.status_code == 202 and replay.json()["id"] == export_id

    assert drain(worker) == 1
    res = client.get(f"/exports/{export_id}")
    assert res.status_code == 200
    body = res.json()
    assert body["state"] == "SUCCEEDED" and body["download_url"] == f"/exports/{export_id}/download"
    assert "traces.jsonl" in body["manifest"]["files"] and body["manifest"]["sha256"]
    assert client.get(f"/jobs/{body['job_id']}").json()["state"] == "SUCCEEDED"
    listed = client.get(f"/projects/{fx.project.id}/exports").json()
    assert [e["id"] for e in listed] == [export_id]

    download = client.get(body["download_url"])
    assert download.status_code == 200 and download.headers["content-type"].startswith("application/zip")
    with zipfile.ZipFile(io.BytesIO(download.content)) as zf:
        names = zf.namelist()
        assert set(body["manifest"]["files"]) == set(names)
        assert not any(n.lower().endswith((".pkl", ".bin")) for n in names)
        blob = "\n".join(zf.read(n).decode("utf-8", errors="replace") for n in names)
        assert "api_key" not in blob.lower()
        for t in fx.reserve:
            assert t.external_id not in blob
    assert client.get("/exports/does-not-exist").status_code == 404
    bad = client.post(f"/projects/{fx.project.id}/exports",
                          json={"kind": "GRADER", "idempotency_key": "api-exp-nograder"})
    assert bad.status_code == 400 and "grader_id" in bad.json()["detail"]


def test_grader_bundle_endpoint_is_credential_free_and_versioned(export_client, db_session, settings):
    client = export_client
    fx = import_partitioned(db_session, settings, train=1, dev=1, reserve=1)
    db_session.commit()
    res = client.get(f"/graders/{fx.grader.id}/bundle")
    assert res.status_code == 200
    bundle = res.json()
    assert bundle["kind"] == "GRADER_BUNDLE" and bundle["grader_id"] == fx.grader.id
    assert bundle["manifest_hash"] == fx.grader.manifest_hash and bundle["pipeline_hash"]
    assert set(bundle["dependency_versions"]) == {"dspy", "gepa", "eval_tinder"}
    assert all(bundle["dependency_versions"].values())
    assert bundle["audits"] == [] and bundle["automation"]["state"] == "DISABLED"
    assert bundle["audit_status"] == "UNAUDITED" and bundle["automation_status"] == "DISABLED"
    text = json.dumps(bundle).lower()
    assert "api_key" not in text and "authorization" not in text and "sk-" not in text
    assert "extra" in bundle["manifest"]["model_config"] and bundle["manifest"]["model_config"]["extra"] == {}
    assert client.get("/graders/missing/bundle").status_code == 404


def test_grading_jobs_refuse_audit_reserve_and_predictions_are_machine(export_client, export_worker, db_session,
                                                                        settings):
    client, worker = export_client, export_worker
    fx = import_partitioned(db_session, settings, train=3, dev=2, reserve=2, truthful_grader=True)
    db_session.commit()
    url = f"/projects/{fx.project.id}/grading-jobs"
    res = client.post(url, json={"grader_id": fx.grader.id, "partition": "AUDIT_RESERVE",
                                     "idempotency_key": "api-bulk-reserve"})
    assert res.status_code == 400 and "sealed" in res.json()["detail"]
    res = client.post(url, json={"grader_id": fx.grader.id, "trace_ids": [fx.reserve[0].id],
                                     "idempotency_key": "api-bulk-reserve-ids"})
    assert res.status_code == 400 and "AUDIT_RESERVE" in res.json()["detail"]
    res = client.post(url, json={"grader_id": fx.grader.id, "partition": "OTHER", "idempotency_key": "x"})
    assert res.status_code == 422
    assert client.get(f"/projects/{fx.project.id}/predictions").status_code == 400  # no shadow grader

    res = client.post(url, json={"grader_id": fx.grader.id, "idempotency_key": "api-bulk-1"})
    assert res.status_code == 202, res.text
    job = res.json()
    assert job["kind"] == "BULK_GRADING" and job["state"] == "QUEUED"
    assert drain(worker) == 1
    assert client.get(f"/jobs/{job['id']}").json()["state"] == "SUCCEEDED"

    res = client.get(f"/projects/{fx.project.id}/predictions", params={"grader_id": fx.grader.id, "limit": 3})
    assert res.status_code == 200
    page = res.json()
    assert page["total"] == len(fx.browsable) and len(page["items"]) == 3 and page["grader_id"] == fx.grader.id
    rest = client.get(f"/projects/{fx.project.id}/predictions",
                          params={"grader_id": fx.grader.id, "limit": 10, "offset": 3}).json()
    ids = {p["trace_id"] for p in page["items"]} | {p["trace_id"] for p in rest["items"]}
    assert ids == {t.id for t in fx.browsable}
    for p in page["items"] + rest["items"]:
        assert p["kind"] == "MACHINE" and p["provisional"] is True
        assert p["audit_status"] == "UNAUDITED" and p["automation_status"] == "DISABLED"
        assert p["grader_id"] == fx.grader.id and not any("confidence" in k for k in p)
    assert client.get("/projects/nope/predictions", params={"grader_id": fx.grader.id}).status_code == 404
