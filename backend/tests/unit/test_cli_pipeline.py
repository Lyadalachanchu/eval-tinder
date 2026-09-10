"""The CLI grades with the running renderer/parser; a bundle from another pipeline is never reported as audited."""
from __future__ import annotations

import json

from typer.testing import CliRunner

from eval_tinder.cli import app
from eval_tinder.domain.manifest import GraderManifest, pipeline_hash
from tests.unit.test_cli import make_bundle


def test_bundle_with_other_renderer_is_graded_as_a_different_pipeline(tmp_path):
    bundle_path, input_path, output_path = tmp_path / "b.json", tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    bundle = make_bundle(bundle_path)
    data = json.loads(bundle_path.read_text())
    # An internally consistent bundle exported by an older renderer/parser pipeline.
    old = GraderManifest.from_dict({**data["manifest"], "renderer_version": "r1", "parser_version": "p1"})
    data["manifest"] = old.to_dict()
    data["manifest_hash"] = old.manifest_hash
    data["pipeline_hash"] = pipeline_hash(old)
    data["audits"] = [{"audit_id": "a", "state": "COMPLETE", "gate_passed": True}]
    data["automation"] = {"state": "ENABLED"}
    bundle_path.write_text(json.dumps(data))
    bundle = data
    input_path.write_text(json.dumps({"external_id": "x", "input": "Cancel.", "output": "Your subscription has been cancelled.",
                                      "tool_calls": [{"name": "cancel_subscription", "result": {"status": "accepted"}}]}) + "\n")
    res = CliRunner().invoke(app, ["grade", "--bundle", str(bundle_path), "--input", str(input_path), "--output",
                                   str(output_path), "--provider", "fake"])
    assert res.exit_code == 0, res.output
    assert "different pipeline" in (res.stderr or res.output)
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows and all(r["pipeline_matches_bundle"] is False for r in rows)
    assert all(r["audit_status"] == "UNAUDITED" and r["automation_status"] == "DISABLED" for r in rows)
    assert all(r["effective_pipeline_hash"] != bundle["pipeline_hash"] for r in rows)
