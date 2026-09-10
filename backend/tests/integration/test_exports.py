"""Exports: sealed audit material, MACHINE tagging, judgment history, secret/pickle guards, CLI round trip."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from sqlalchemy import select

from eval_tinder.db.enums import GradingPurpose, JobState, ReviewPurpose, SelectionCategory
from eval_tinder.db.models import ExportBundle, GraderVersion, HumanJudgment, Job, Project, ReviewRequest
from eval_tinder.services import exports
from eval_tinder.services.exports import (
    ExportError,
    assert_safe_members,
    create_export,
    export_job_handler,
    grader_bundle,
    write_bundle_zip,
)
from eval_tinder.services.review import correct_judgment, create_requests, submit_judgment
from eval_tinder.worker.handlers import build_handlers
from eval_tinder.worker.main import Worker, drain
from tests.cases import TRAIN_CASES
from tests.integration.test_bulk_grading import (
    RESERVE_CANARY,
    Imported,
    bulk_worker,
    import_partitioned,
)
from eval_tinder.services.bulk_grading import enqueue_bulk_grading

REVIEWER = "expert-7c1d"


# ----------------------------------------------------------------- helpers


def read_zip(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


def jsonl(data: bytes) -> list[dict]:
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def label_train(session, fx: Imported) -> list[HumanJudgment]:
    """Label two TRAIN traces blind, then correct the first so a superseded judgment exists."""
    picks = fx.train[:2]
    requests = create_requests(session, fx.project, picks, purpose=ReviewPurpose.TRAIN, category=SelectionCategory.SEED)
    judgments = []
    for i, (t, req) in enumerate(zip(picks, requests, strict=True)):
        judgments.append(
            submit_judgment(
                session, req.id, verdict=TRAIN_CASES[i].label, explanation=f"expert note {i}", reviewer_id=REVIEWER,
                shown_context_hash=t.content_hash, idempotency_key=f"j-{req.id}",
            )
        )
    correction = correct_judgment(
        session, judgments[0].id, verdict="CANNOT_JUDGE", cannot_judge_reason="MISSING_CONTEXT",
        explanation="on reflection the record lacks context", reviewer_id=REVIEWER, idempotency_key="corr-1",
    )
    return judgments + [correction]


def bulk_grade(session, session_factory, settings, fx: Imported, key: str = "bulk-export") -> Job:
    job = enqueue_bulk_grading(session, fx.project, grader_id=fx.grader.id, partition=None, trace_ids=None,
                               idempotency_key=key, settings=settings)
    session.commit()
    drain(bulk_worker(settings, session_factory))
    session.expire_all()
    return session.get(Job, job.id)


def all_text(members: dict[str, bytes]) -> str:
    return "\n".join(v.decode("utf-8", errors="replace") for v in members.values())


# ----------------------------------------------------------------- FULL export without a released audit


def test_full_export_contents_exclude_sealed_audit_material(db_session, session_factory, settings, tmp_path):
    fx = import_partitioned(db_session, settings, truthful_grader=True)
    judgments = label_train(db_session, fx)
    bulk_grade(db_session, session_factory, settings, fx)
    project = db_session.get(Project, fx.project.id)
    manifest = write_bundle_zip(db_session, project, kind="FULL", grader_id=fx.grader.id, path=tmp_path / "full.zip")
    members = read_zip(tmp_path / "full.zip")
    graders = list(db_session.scalars(select(GraderVersion).where(GraderVersion.project_id == project.id)))
    expected = {
        "traces.jsonl", "human_judgments.jsonl", "machine_predictions.jsonl", "partitions.jsonl",
        "dataset_snapshots.json", "optimization_runs.json", "policy_epoch.json", "summary.json", "README.md",
        *(f"graders/{g.id}.json" for g in graders),
    }
    assert set(members) == expected == set(manifest["files"])
    assert manifest["sha256"] and manifest["sizes"]["traces.jsonl"] == len(members["traces.jsonl"])
    assert not any(name.startswith("audit_samples/") for name in members)
    assert not any(name.lower().endswith((".pkl", ".bin")) for name in members)

    text = all_text(members)
    assert RESERVE_CANARY not in text
    for t in fx.reserve:
        assert t.external_id not in text and t.id not in text
    assert "api_key" not in text.lower() and "sk-" not in text

    traces = jsonl(members["traces.jsonl"])
    assert {r["external_id"] for r in traces} == {t.external_id for t in fx.browsable}
    assert {r["partition"] for r in traces} == {"TRAIN", "DEV"}
    partitions = jsonl(members["partitions.jsonl"])
    assert {p["partition"] for p in partitions} == {"TRAIN", "DEV"}
    assert all({"group_id", "partition", "seed", "exposure_status"} <= set(p) for p in partitions)
    summary = json.loads(members["summary.json"])
    assert summary["counts"]["audit_reserve_groups_withheld"] == len(fx.reserve)
    assert summary["counts"]["audits_released"] == 0

    rows = jsonl(members["human_judgments.jsonl"])
    by_id = {r["id"]: r for r in rows}
    assert set(by_id) == {j.id for j in judgments}  # superseded judgments are kept
    original, correction = by_id[judgments[0].id], by_id[judgments[-1].id]
    assert original["superseded_by"] == correction["id"] and correction["supersedes"] == original["id"]
    assert original["is_active"] is False and correction["is_active"] is True
    assert all(r["kind"] == "HUMAN" and r["purpose"] == "TRAIN" and r["policy_epoch"] == 1 for r in rows)

    preds = jsonl(members["machine_predictions.jsonl"])
    assert len(preds) == len(fx.browsable)
    for p in preds:
        assert p["kind"] == "MACHINE" and p["grader_id"] == fx.grader.id
        assert p["manifest_hash"] == fx.grader.manifest_hash and p["purpose"] == GradingPurpose.BULK
        assert p["audit_status"] == "UNAUDITED" and p["automation_status"] == "DISABLED" and p["provisional"] is True
        assert {"status", "verdict", "cache_hit_of", "evidence", "explanation"} <= set(p)
        assert not any("confidence" in k for k in p)
    readme = members["README.md"].decode("utf-8")
    assert "not human labels" in readme and "traces.jsonl" in readme and "machine_predictions.jsonl" in readme
    bundle = json.loads(members[f"graders/{fx.grader.id}.json"])
    assert bundle["kind"] == "GRADER_BUNDLE" and bundle["manifest_hash"] == fx.grader.manifest_hash
    assert set(bundle["dependency_versions"]) == {"dspy", "gepa", "eval_tinder"}
    assert bundle["audits"] == [] and bundle["automation"]["state"] == "DISABLED"
    snapshots = json.loads(members["dataset_snapshots.json"])
    runs = json.loads(members["optimization_runs.json"])
    assert snapshots == [] and runs["runs"] == [] and runs["dependency_versions"]["dspy"]
    assert json.loads(members["policy_epoch.json"])["policy_epoch"] == 1


# ----------------------------------------------------------------- FULL export with a COMPLETE audit


def _complete_audit(db_session, session_factory, settings, fx: Imported, *, planned_n: int, key: str):
    try:
        from eval_tinder.services import audits
    except ImportError:  # pragma: no cover - audits area not present in this checkout
        pytest.skip("services.audits is not available")
    audit, _job = audits.lock_audit(
        db_session, fx.project, grader_id=fx.grader.id, planned_n=planned_n, seed=5, population=None,
        sampling_plan={"unit": "group", "method": "uniform_random", "independence_assumption_documented": True,
                       "independence_note": "one production group per customer; groups are independent"},
        risk_targets={"permitted_verdicts": ["PASS", "FAIL"], "max_error_rate": 0.5, "min_coverage": 0.1,
                      "confidence": 0.9},
        idempotency_key=key, settings=settings,
    )
    db_session.commit()
    drain(Worker(build_handlers(), settings=settings, worker_id="audit-w", session_factory=session_factory))
    db_session.expire_all()
    audit = db_session.get(type(audit), audit.id)
    return audit, audits


def _judge_audit(db_session, audit, *, verdict: str = "PASS") -> list[HumanJudgment]:
    out = []
    for req in db_session.scalars(select(ReviewRequest).where(ReviewRequest.audit_run_id == audit.id)):
        out.append(
            submit_judgment(
                db_session, req.id, verdict=verdict, explanation="blind audit judgment", reviewer_id=REVIEWER,
                shown_context_hash=req.trace.content_hash, idempotency_key=f"audit-j-{req.id}",
            )
        )
    return out


def test_full_export_includes_only_the_completed_audit_sample(db_session, session_factory, settings, tmp_path):
    fx = import_partitioned(db_session, settings, train=3, dev=2, reserve=8, reserve_source="PRODUCTION",
                            truthful_grader=True)
    bulk_grade(db_session, session_factory, settings, fx)
    done, audits = _complete_audit(db_session, session_factory, settings, fx, planned_n=3, key="audit-done")
    _judge_audit(db_session, done)
    audits.persist_report(db_session, done)
    db_session.commit()
    db_session.expire_all()
    done = db_session.get(type(done), done.id)
    assert done.state == "COMPLETE" and done.report["gate"]["passed"] in (True, False)
    pending, _ = _complete_audit(db_session, session_factory, settings, fx, planned_n=2, key="audit-pending")
    assert pending.state == "IN_REVIEW"
    project = db_session.get(Project, fx.project.id)

    write_bundle_zip(db_session, project, kind="FULL", grader_id=None, path=tmp_path / "full-audit.zip")
    members = read_zip(tmp_path / "full-audit.zip")
    folder = f"audit_samples/{done.id}"
    assert {f"{folder}/report.json", f"{folder}/traces.jsonl", f"{folder}/human_judgments.jsonl"} <= set(members)
    assert not any(name.startswith(f"audit_samples/{pending.id}") for name in members)

    sample = jsonl(members[f"{folder}/traces.jsonl"])
    assert {r["id"] for r in sample} == set(done.locked_sample_ids) and len(sample) == 3
    assert all(r["partition"] == "AUDIT_RESERVE" for r in sample)
    audit_rows = jsonl(members[f"{folder}/human_judgments.jsonl"])
    assert len(audit_rows) == 3 and all(r["purpose"] == "AUDIT" and r["kind"] == "HUMAN" for r in audit_rows)
    report = json.loads(members[f"{folder}/report.json"])
    assert report["state"] == "COMPLETE" and report["report"]["gate"]["passed"] == done.report["gate"]["passed"]
    assert report["planned_sample_size"] == 3 and report["exported_sample_size"] == 3

    # the main files never carry audit material, released or not
    assert all(r["purpose"] != "AUDIT" for r in jsonl(members["human_judgments.jsonl"]))
    assert {r["external_id"] for r in jsonl(members["traces.jsonl"])} == {t.external_id for t in fx.browsable}
    released = set(done.locked_sample_ids)
    withheld = [t for t in fx.reserve if t.id not in released]
    text = all_text(members)
    for t in withheld:
        assert t.external_id not in text and t.id not in text
    for t in fx.reserve:
        if t.id in released:
            assert t.external_id in members[f"{folder}/traces.jsonl"].decode()
    preds = jsonl(members["machine_predictions.jsonl"])
    audit_preds = [p for p in preds if p["purpose"] == "AUDIT"]
    assert audit_preds and {p["audit_run_id"] for p in audit_preds} == {done.id}
    assert {p["trace_id"] for p in audit_preds} == released
    assert all(p["kind"] == "MACHINE" for p in preds)
    partitions = jsonl(members["partitions.jsonl"])
    reserve_rows = [p for p in partitions if p["partition"] == "AUDIT_RESERVE"]
    assert {p["group_id"] for p in reserve_rows} == {t.group_id for t in fx.reserve if t.id in released}
    summary = json.loads(members["summary.json"])
    assert summary["counts"]["audits_released"] == 1 and summary["counts"]["audits_withheld"] == 1
    assert summary["counts"]["audit_reserve_groups_withheld"] == len(withheld)
    bundle = json.loads(members[f"graders/{fx.grader.id}.json"])
    assert [a["audit_id"] for a in bundle["audits"]] == [done.id]
    assert bundle["audits"][0]["state"] == "COMPLETE" and bundle["audits"][0]["gate_passed"] == done.report["gate"]["passed"]
    assert bundle["audit_status"] == ("AUDITED" if done.report["gate"]["passed"] else "AUDIT_GATE_FAILED")


# ----------------------------------------------------------------- guards


def test_export_guards_refuse_pickles_and_credentials():
    with pytest.raises(ExportError, match="serialized"):
        assert_safe_members({"graders/g.pkl": b"\x80\x04"})
    with pytest.raises(ExportError, match="serialized"):
        assert_safe_members({"model.bin": b"\x00"})
    with pytest.raises(ExportError, match="credential"):
        assert_safe_members({"a.json": json.dumps({"api_key": "x"}).encode()})
    with pytest.raises(ExportError, match="credential"):
        assert_safe_members({"a.jsonl": b'{"note": "token sk-abcdefghijklmnop"}\n'})
    assert_safe_members({"a.json": json.dumps({"task-1": 1, "desk-lamp": "risk-free", "asks": "ok"}).encode()})


def test_grader_bundle_cannot_carry_credentials(db_session, settings):
    fx = import_partitioned(db_session, settings, train=1, dev=1, reserve=1)
    grader = db_session.get(GraderVersion, fx.grader.id)
    grader.manifest = {**grader.manifest, "model_config": {**grader.manifest["model_config"],
                                                            "extra": {"api_key": "sk-secret-secret-secret"}}}
    db_session.flush()
    with pytest.raises(ValueError, match="credential"):
        grader_bundle(db_session, grader)


# ----------------------------------------------------------------- GRADER export, jobs, CLI round trip


def test_grader_export_job_and_cli_round_trip(db_session, session_factory, settings, tmp_path):
    fx = import_partitioned(db_session, settings, train=1, dev=1, reserve=1, truthful_grader=True)
    bundle, job = create_export(db_session, fx.project, kind="GRADER", grader_id=fx.grader.id,
                                idempotency_key="exp-grader", settings=settings)
    assert bundle.kind == "GRADER" and bundle.job_id == job.id and bundle.manifest == {}
    assert Path(bundle.path) == Path(settings.artifact_path) / "exports" / f"{bundle.id}.zip"
    same_bundle, same_job = create_export(db_session, fx.project, kind="GRADER", grader_id=fx.grader.id,
                                          idempotency_key="exp-grader", settings=settings)
    assert (same_bundle.id, same_job.id) == (bundle.id, job.id)
    with pytest.raises(ExportError, match="requires grader_id"):
        create_export(db_session, fx.project, kind="GRADER", grader_id=None, idempotency_key="exp-nograder",
                      settings=settings)
    db_session.commit()
    worker = Worker({"EXPORT": export_job_handler}, settings=settings, worker_id="exp-w", session_factory=session_factory)
    assert drain(worker) == 1
    db_session.expire_all()
    job = db_session.get(Job, job.id)
    bundle = db_session.get(ExportBundle, bundle.id)
    assert job.state == JobState.SUCCEEDED and job.result["sha256"] == bundle.manifest["sha256"]
    assert set(bundle.manifest["files"]) == {f"graders/{fx.grader.id}.json", "README.md"}
    members = read_zip(Path(bundle.path))
    assert set(members) == set(bundle.manifest["files"])
    exported = json.loads(members[f"graders/{fx.grader.id}.json"])
    assert exported["instruction_text"] == fx.grader.instruction_text
    assert "api_key" not in json.dumps(exported).lower()

    # CLI: export-grader (database) -> grade --provider fake on a fresh JSONL -> MACHINE predictions
    from typer.testing import CliRunner

    from eval_tinder.cli import app

    runner = CliRunner()
    bundle_path = tmp_path / "grader.json"
    res = runner.invoke(app, ["export-grader", "--grader-id", fx.grader.id, "--output", str(bundle_path)])
    assert res.exit_code == 0, res.output
    cli_bundle = json.loads(bundle_path.read_text())
    assert cli_bundle["manifest_hash"] == exported["manifest_hash"] == fx.grader.manifest_hash
    assert cli_bundle["pipeline_hash"] == exported["pipeline_hash"]
    input_path = tmp_path / "new.jsonl"
    input_path.write_text(
        "\n".join(
            json.dumps({"external_id": c.key, "input": c.input, "output": c.output, "tool_calls": c.tool_calls(),
                        "context": {"subscription_id": "s-demo"}})
            for c in TRAIN_CASES
        )
    )
    output_path = tmp_path / "preds.jsonl"
    res = runner.invoke(app, ["grade", "--bundle", str(bundle_path), "--input", str(input_path), "--output",
                              str(output_path), "--provider", "fake"])
    assert res.exit_code == 0, res.output
    rows = jsonl(output_path.read_bytes())
    assert [r["external_id"] for r in rows] == [c.key for c in TRAIN_CASES]
    for c, r in zip(TRAIN_CASES, rows, strict=True):
        assert r["status"] == "OK" and r["verdict"] == c.label, (c.key, r)
        assert r["kind"] == "MACHINE" and r["provisional"] is True and r["grader_id"] == fx.grader.id
        assert r["manifest_hash"] == fx.grader.manifest_hash and r["pipeline_matches_bundle"] is True
        assert r["audit_status"] == "UNAUDITED" and r["automation_status"] == "DISABLED"
    assert "OK=6" in res.stderr


def test_full_export_job_manifest_and_missing_grader(db_session, session_factory, settings):
    fx = import_partitioned(db_session, settings, train=2, dev=1, reserve=1)
    with pytest.raises(LookupError):
        create_export(db_session, fx.project, kind="FULL", grader_id="nope", idempotency_key="exp-missing",
                      settings=settings)
    with pytest.raises(ExportError, match="unknown export kind"):
        create_export(db_session, fx.project, kind="PICKLE", grader_id=None, idempotency_key="exp-kind",
                      settings=settings)
    bundle, job = create_export(db_session, fx.project, kind="FULL", grader_id=None, idempotency_key="exp-full",
                                settings=settings)
    db_session.commit()
    drain(Worker({"EXPORT": exports.export_job_handler}, settings=settings, worker_id="exp-w",
                 session_factory=session_factory))
    db_session.expire_all()
    bundle = db_session.get(ExportBundle, bundle.id)
    job = db_session.get(Job, job.id)
    assert job.state == JobState.SUCCEEDED
    assert "traces.jsonl" in bundle.manifest["files"] and bundle.manifest["bytes"] == Path(bundle.path).stat().st_size
    with zipfile.ZipFile(io.BytesIO(Path(bundle.path).read_bytes())) as zf:
        assert zf.testzip() is None
