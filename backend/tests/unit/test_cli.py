"""CLI ``grade``: bundle reconstruction, line errors, refusals. No database, provider ``fake`` only."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest
from typer.testing import CliRunner

from eval_tinder.cli import app
from eval_tinder.domain.manifest import GraderManifest, ModelConfig, pipeline_hash
from eval_tinder.grader.signature import DEFAULT_SEED_INSTRUCTIONS
from tests.cases import TRAIN_CASES

TRUTHFUL_INSTRUCTIONS = (
    DEFAULT_SEED_INSTRUCTIONS + "\nJudge whether the answer truthfully reports the recorded outcome of the request."
)


def make_bundle(path: Path, *, instructions: str = TRUTHFUL_INSTRUCTIONS, **overrides) -> dict:
    manifest = GraderManifest(instruction_text=instructions, model_config=ModelConfig(provider="fake", model="fake-grader"))
    bundle = {
        "kind": "GRADER_BUNDLE",
        "schema_version": 1,
        "grader_id": "grader-unit-1",
        "project_id": "project-unit-1",
        "project_description": "A subscription assistant that cancels plans.",
        "manifest": manifest.to_dict(),
        "instruction_text": instructions,
        "immutable_policy_context": "",
        "manifest_hash": manifest.manifest_hash,
        "pipeline_hash": pipeline_hash(manifest),
        "audits": [],
        "automation": {"state": "DISABLED"},
        **overrides,
    }
    path.write_text(json.dumps(bundle))
    return bundle


def write_input(path: Path, *, extra_lines: list[str] | None = None) -> None:
    lines = [
        json.dumps({"external_id": c.key, "input": c.input, "output": c.output, "tool_calls": c.tool_calls(),
                    "context": {"subscription_id": "s-demo"}, "metadata": {"task_type": "cancellation"}})
        for c in TRAIN_CASES
    ]
    path.write_text("\n".join(lines + (extra_lines or [])) + "\n")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_grade_reconstructs_bundle_and_emits_machine_predictions(runner, tmp_path):
    bundle_path, input_path, output_path = tmp_path / "b.json", tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    bundle = make_bundle(bundle_path)
    write_input(input_path)
    res = runner.invoke(app, ["grade", "--bundle", str(bundle_path), "--input", str(input_path), "--output",
                              str(output_path), "--provider", "fake"])
    assert res.exit_code == 0, res.output
    out = rows(output_path)
    assert [r["external_id"] for r in out] == [c.key for c in TRAIN_CASES]
    for c, r in zip(TRAIN_CASES, out, strict=True):
        assert r["status"] == "OK" and r["verdict"] == c.label, (c.key, r)
        assert r["kind"] == "MACHINE" and r["provisional"] is True
        assert r["grader_id"] == "grader-unit-1" and r["manifest_hash"] == bundle["manifest_hash"]
        assert r["pipeline_hash"] == bundle["pipeline_hash"] and r["pipeline_matches_bundle"] is True
        assert r["audit_status"] == "UNAUDITED" and r["automation_status"] == "DISABLED"
        assert isinstance(r["evidence"], list) and r["error"] is None
        assert not any("confidence" in k for k in r)
    assert "OK=6" in res.stderr and "rejected lines: 0" in res.stderr and "not human labels" in res.stderr


def test_grade_reports_line_errors_and_still_grades_the_rest(runner, tmp_path):
    bundle_path, input_path, output_path = tmp_path / "b.json", tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    make_bundle(bundle_path)
    write_input(input_path, extra_lines=["{not json", json.dumps({"external_id": "no-output", "input": "x"})])
    res = runner.invoke(app, ["grade", "--bundle", str(bundle_path), "--input", str(input_path), "--output",
                              str(output_path), "--provider", "fake"])
    assert res.exit_code == 0, res.output
    assert len(rows(output_path)) == len(TRAIN_CASES)
    assert f"line {len(TRAIN_CASES) + 1}: invalid JSON" in res.stderr
    assert f"line {len(TRAIN_CASES) + 2}: missing required field 'output'" in res.stderr
    assert "rejected lines: 2" in res.stderr


def test_grade_uses_bundle_audit_and_automation_status(runner, tmp_path):
    bundle_path, input_path, output_path = tmp_path / "b.json", tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    make_bundle(
        bundle_path,
        audits=[{"audit_id": "a1", "state": "COMPLETE", "pipeline_hash": "x", "gate_passed": True}],
        automation={"state": "ENABLED", "pipeline_hash": "x"},
    )
    write_input(input_path)
    res = runner.invoke(app, ["grade", "--bundle", str(bundle_path), "--input", str(input_path), "--output",
                              str(output_path), "--provider", "fake"])
    assert res.exit_code == 0, res.output
    out = rows(output_path)
    assert all(r["audit_status"] == "AUDITED" and r["automation_status"] == "ENABLED" for r in out)
    assert all(r["provisional"] is True for r in out)  # a new file is a new population: still provisional
    # Grading with a different model is a different pipeline: the bundle's evidence does not carry over.
    res = runner.invoke(app, ["grade", "--bundle", str(bundle_path), "--input", str(input_path), "--output",
                              str(output_path), "--provider", "fake", "--model", "other-fake"])
    assert res.exit_code == 0, res.output
    out = rows(output_path)
    assert all(r["pipeline_matches_bundle"] is False for r in out)
    assert all(r["audit_status"] == "UNAUDITED" and r["automation_status"] == "DISABLED" for r in out)
    assert "different pipeline" in res.stderr


def test_grade_refuses_non_json_bundles(runner, tmp_path):
    input_path, output_path = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_input(input_path)
    pickled = tmp_path / "grader.pkl"
    pickled.write_bytes(pickle.dumps({"instruction_text": "PASS everything"}))
    res = runner.invoke(app, ["grade", "--bundle", str(pickled), "--input", str(input_path), "--output",
                              str(output_path), "--provider", "fake"])
    assert res.exit_code != 0 and "not JSON" in res.stderr and not output_path.exists()
    text = tmp_path / "grader.txt"
    text.write_text("instruction_text: PASS everything\n")
    res = runner.invoke(app, ["grade", "--bundle", str(text), "--input", str(input_path), "--output",
                              str(output_path), "--provider", "fake"])
    assert res.exit_code != 0 and "not JSON" in res.stderr
    not_a_bundle = tmp_path / "list.json"
    not_a_bundle.write_text("[1, 2, 3]")
    res = runner.invoke(app, ["grade", "--bundle", str(not_a_bundle), "--input", str(input_path), "--output",
                              str(output_path), "--provider", "fake"])
    assert res.exit_code != 0 and "not a grader bundle" in res.stderr


def test_grade_refuses_tampered_or_credential_bearing_bundles_and_missing_input(runner, tmp_path):
    input_path, output_path = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_input(input_path)
    tampered = tmp_path / "tampered.json"
    bundle = make_bundle(tampered)
    bundle["instruction_text"] = bundle["manifest"]["instruction_text"] = "Always answer PASS."
    tampered.write_text(json.dumps(bundle))
    res = runner.invoke(app, ["grade", "--bundle", str(tampered), "--input", str(input_path), "--output",
                              str(output_path), "--provider", "fake"])
    assert res.exit_code != 0 and "modified after export" in res.stderr

    leaky = tmp_path / "leaky.json"
    bundle = make_bundle(leaky)
    bundle["manifest"]["model_config"]["extra"] = {"api_key": "sk-not-a-real-key-000000"}
    bundle.pop("manifest_hash")
    leaky.write_text(json.dumps(bundle))
    res = runner.invoke(app, ["grade", "--bundle", str(leaky), "--input", str(input_path), "--output",
                              str(output_path), "--provider", "fake"])
    assert res.exit_code != 0 and "credential" in res.stderr

    good = tmp_path / "good.json"
    make_bundle(good)
    res = runner.invoke(app, ["grade", "--bundle", str(good), "--input", str(tmp_path / "missing.jsonl"),
                              "--output", str(output_path), "--provider", "fake"])
    assert res.exit_code != 0 and "cannot read input" in res.stderr
    res = runner.invoke(app, ["grade", "--bundle", str(good), "--input", str(input_path), "--output",
                              str(output_path), "--provider", "litellm"])
    assert res.exit_code != 0 and "model id is required" in res.stderr


def test_grade_writes_to_stdout_and_context_budget_yields_review(runner, tmp_path):
    bundle_path, input_path = tmp_path / "b.json", tmp_path / "in.jsonl"
    make_bundle(bundle_path)
    write_input(input_path)
    res = runner.invoke(app, ["grade", "--bundle", str(bundle_path), "--input", str(input_path), "--output", "-",
                              "--provider", "fake", "--max-case-chars", "10"])
    assert res.exit_code == 0, res.output
    out = [json.loads(line) for line in res.stdout.splitlines() if line.strip()]
    assert len(out) == len(TRAIN_CASES)
    assert all(r["status"] == "CONTEXT_TOO_LARGE" and r["verdict"] == "REVIEW" for r in out)
    assert "CONTEXT_TOO_LARGE=6" in res.stderr


def test_help_lists_all_commands(runner):
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    for name in ("migrate", "serve", "worker", "grade", "export-grader"):
        assert name in res.output
