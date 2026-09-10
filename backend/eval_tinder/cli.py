"""``eval-tinder`` command line: migrations, API server, worker, portable grading, grader export.

``grade`` reconstructs an exported grader from its JSON bundle (prompt text and
configuration only; anything that is not JSON is refused, so no pickle can ever
be loaded) and grades a new JSONL file with the configured provider. Its output
rows are MACHINE predictions: provisional, never human labels, and without any
per-case confidence figure. Credentials come from the environment, never from
the bundle.
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import typer

from eval_tinder.db.enums import AutomationState

app = typer.Typer(
    help="Human-aligned evaluation of production traces with GEPA-evolved graders.",
    no_args_is_help=True,
    add_completion=False,
)

EXIT_USAGE = 2
BUNDLE_KIND = "GRADER_BUNDLE"


def _err(message: str) -> None:
    typer.echo(message, err=True)


def _fail(message: str, code: int = EXIT_USAGE) -> None:
    _err(f"error: {message}")
    raise typer.Exit(code=code)


# ---------------------------------------------------------------- migrate / serve / worker


@app.command()
def migrate(
    database_url: Optional[str] = typer.Option(
        None, "--database-url", envvar="DATABASE_URL", help="SQLAlchemy URL (defaults to DATABASE_URL)."
    ),
    revision: str = typer.Option("head", "--revision", help="Alembic revision to upgrade to."),
) -> None:
    """Run database migrations (alembic upgrade head)."""
    from alembic import command
    from alembic.config import Config

    from eval_tinder.config import get_settings

    backend_dir = Path(__file__).resolve().parents[1]
    scripts = backend_dir / "alembic"
    ini = backend_dir / "alembic.ini"
    if not scripts.exists():
        _fail(f"alembic scripts not found at {scripts}; run migrations from the backend checkout")
    url = database_url or get_settings().database_url
    os.environ["ALEMBIC_DATABASE_URL"] = url
    cfg = Config(str(ini)) if ini.exists() else Config()
    cfg.set_main_option("script_location", str(scripts))
    command.upgrade(cfg, revision)
    typer.echo(f"migrated to {revision}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address (loopback unless API_TOKEN is set)."),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (development only)."),
) -> None:
    """Serve the HTTP API with uvicorn."""
    import uvicorn

    uvicorn.run("eval_tinder.api.app:app", host=host, port=port, reload=reload)


@app.command()
def worker(
    once: bool = typer.Option(False, "--once", help="Claim and run at most one job, then exit."),
    poll_interval: float = typer.Option(1.0, "--poll-interval", help="Seconds between idle polls."),
) -> None:
    """Run the job worker (imports, optimization, selection, bulk grading, audits, exports)."""
    from eval_tinder.worker.handlers import build_handlers
    from eval_tinder.worker.main import Worker

    w = Worker(build_handlers(), poll_interval=poll_interval)
    if once:
        job = w.run_once()
        typer.echo("no runnable job" if job is None else f"ran job {job.id} ({job.kind})")
        return
    w.run_forever()


# ---------------------------------------------------------------- grade


def load_bundle(path: Path):
    """Load a grader bundle. Anything that is not a JSON GRADER_BUNDLE is refused."""
    from eval_tinder.domain.manifest import GraderManifest, pipeline_hash

    try:
        raw = path.read_bytes()
    except OSError as e:
        _fail(f"cannot read bundle {path}: {e}")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        _fail(f"bundle {path} is not JSON ({e}); serialized programs and pickles are refused")
    if not isinstance(data, dict) or not isinstance(data.get("manifest"), dict):
        _fail(f"bundle {path} is not a grader bundle: expected a JSON object with a 'manifest'")
    if data.get("kind", BUNDLE_KIND) != BUNDLE_KIND:
        _fail(f"bundle {path} has kind {data.get('kind')!r}, expected {BUNDLE_KIND!r}")
    try:
        manifest = GraderManifest.from_dict(data["manifest"])
    except ValueError as e:
        _fail(f"bundle {path} manifest is invalid: {e}")
    declared = data.get("manifest_hash")
    if declared and declared != manifest.manifest_hash:
        _fail(
            f"bundle {path} declares manifest_hash {declared[:12]}... but its manifest hashes to "
            f"{manifest.manifest_hash[:12]}...; the bundle was modified after export"
        )
    data.setdefault("manifest_hash", manifest.manifest_hash)
    data.setdefault("pipeline_hash", pipeline_hash(manifest))
    return data, manifest


def bundle_audit_status(bundle: dict) -> str:
    """AUDITED only when a listed (COMPLETE/SPENT) audit passed its predeclared gate."""
    audits = bundle.get("audits") or []
    if any(a.get("gate_passed") is True for a in audits if isinstance(a, dict)):
        return "AUDITED"
    if any(a.get("gate_passed") is False for a in audits if isinstance(a, dict)):
        return "AUDIT_GATE_FAILED"
    if audits:
        return "AUDIT_COMPLETE"
    return "UNAUDITED"


def bundle_automation_status(bundle: dict) -> str:
    automation = bundle.get("automation") or {}
    state = automation.get("state") if isinstance(automation, dict) else None
    return state if state in {s.value for s in AutomationState} else AutomationState.DISABLED.value


def build_cli_lm(manifest, *, provider: Optional[str], model: Optional[str]):
    """Provider/model precedence: option > explicit environment variable > bundle manifest."""
    from eval_tinder.domain.manifest import ModelConfig
    from eval_tinder.llm.factory import ConfigurationError, build_grading_lm
    from eval_tinder.llm.fakes import ScriptedGradingLM, keyword_switch_policy

    base = manifest.model_config_.sanitized()
    provider = provider or os.environ.get("LLM_PROVIDER") or base.provider
    model = model or os.environ.get("GRADER_MODEL") or (base.model if provider == base.provider else None)
    if provider not in ("fake", "litellm"):
        _fail(f"unknown provider {provider!r}; expected 'fake' or 'litellm'")
    if provider == "fake":
        if base.provider == "fake" and (model or base.model) == base.model:
            effective = base  # exactly the bundle's configuration: the pipeline hash can match
        else:
            effective = ModelConfig(provider="fake", model=model or "fake-grader", temperature=0.0, max_tokens=base.max_tokens)
        return ScriptedGradingLM(keyword_switch_policy, model=effective.model), effective
    if not model:
        _fail("a model id is required for provider 'litellm' (use --model or GRADER_MODEL)")
    if base.provider == "litellm" and model == base.model:
        effective = base
    else:
        effective = ModelConfig(
            provider="litellm", model=model, temperature=base.temperature, max_tokens=base.max_tokens, extra=base.extra
        )
    try:
        return build_grading_lm(effective), effective
    except ConfigurationError as e:
        _fail(str(e))


@app.command()
def grade(
    bundle: Path = typer.Option(..., "--bundle", help="Grader bundle JSON (from export-grader or the API)."),
    input: Path = typer.Option(..., "--input", help="JSONL file of trace records to grade."),
    output: Path = typer.Option(..., "--output", help="Destination JSONL of MACHINE predictions ('-' for stdout)."),
    provider: Optional[str] = typer.Option(None, "--provider", help="fake | litellm (default: bundle, or LLM_PROVIDER)."),
    model: Optional[str] = typer.Option(None, "--model", help="Model id for litellm (default: bundle, or GRADER_MODEL)."),
    max_case_chars: Optional[int] = typer.Option(
        None, "--max-case-chars", help="Rendered-case budget; larger cases get REVIEW/CONTEXT_TOO_LARGE."
    ),
) -> None:
    """Reconstruct an exported grader and grade a JSONL file. Exit 0 even when some records error."""
    from eval_tinder.config import get_settings
    from eval_tinder.domain.manifest import pipeline_hash
    from eval_tinder.domain.rendering import render_case, render_project_context
    from eval_tinder.grader.runtime import grade as grade_case
    from eval_tinder.grader.signature import build_module
    from eval_tinder.services.imports import parse_jsonl

    data, manifest = load_bundle(bundle)
    try:
        raw = input.read_bytes()
    except OSError as e:
        _fail(f"cannot read input {input}: {e}")
    settings = get_settings()
    budget = max_case_chars if max_case_chars is not None else settings.max_case_chars
    lm, effective = build_cli_lm(manifest, provider=provider, model=model)
    # The pipeline that actually grades = bundle prompt + effective model + the RUNNING renderer/parser.
    from eval_tinder.domain.manifest import PARSER_VERSION
    from eval_tinder.domain.rendering import RENDERER_VERSION

    effective_manifest = manifest.model_copy(
        update={"model_config_": effective, "renderer_version": RENDERER_VERSION, "parser_version": PARSER_VERSION}
    )
    effective_pipeline = pipeline_hash(effective_manifest)
    same_pipeline = effective_pipeline == data.get("pipeline_hash")
    if not same_pipeline:
        _err(
            f"warning: effective pipeline {effective_pipeline[:12]} differs from the bundle's "
            f"{str(data.get('pipeline_hash'))[:12]} (model {effective.provider}/{effective.model} vs bundle "
            f"{manifest.model_config_.provider}/{manifest.model_config_.model}; renderer {RENDERER_VERSION} vs "
            f"{manifest.renderer_version}; parser {PARSER_VERSION} vs {manifest.parser_version}): this is a different "
            "pipeline. Audit evidence and automation status of the bundle do not apply; rows are UNAUDITED/DISABLED."
        )
    audit_status = bundle_audit_status(data) if same_pipeline else "UNAUDITED"
    automation_status = bundle_automation_status(data) if same_pipeline else AutomationState.DISABLED.value

    parsed = parse_jsonl(raw, max_bytes=max(len(raw), 1))
    for err in parsed.errors:
        _err(f"line {err.get('line')}: {err.get('error')}")

    module = build_module(manifest.instruction_text)
    project_context = render_project_context(
        str(data.get("project_description") or data.get("description") or ""), manifest.immutable_policy_context
    )
    counts: Counter[str] = Counter()
    verdicts: Counter[str] = Counter()
    sink = sys.stdout if str(output) == "-" else open(output, "w", encoding="utf-8")  # noqa: SIM115
    try:
        for rec in parsed.records:
            case = render_case(
                input_text=rec.input, output_text=rec.output, context=rec.context, tool_calls=rec.tool_calls,
                metadata=rec.metadata,
            )
            result = grade_case(module, lm, case=case, project_context=project_context, max_case_chars=budget)
            counts[result.status] += 1
            verdicts[result.verdict] += 1
            row = {
                "external_id": rec.external_id,
                "group_id": rec.group_id,
                "kind": "MACHINE",
                "grader_id": data.get("grader_id"),
                "manifest_hash": data["manifest_hash"],
                "pipeline_hash": data["pipeline_hash"],
                "effective_pipeline_hash": effective_pipeline,
                "pipeline_matches_bundle": same_pipeline,
                "status": result.status,
                "verdict": result.verdict,
                "evidence": result.evidence,
                "explanation": result.explanation,
                "error": result.error,
                "provisional": True,
                "audit_status": audit_status,
                "automation_status": automation_status,
                "case_hash": case.text_hash,
                "attempts": result.attempts,
                "latency_ms": result.latency_ms,
            }
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        sink.flush()
    finally:
        if sink is not sys.stdout:
            sink.close()
    by_status = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"
    by_verdict = ", ".join(f"{k}={v}" for k, v in sorted(verdicts.items())) or "none"
    _err(
        f"graded {len(parsed.records)} record(s) with grader {data.get('grader_id')} "
        f"({effective.provider}/{effective.model}); status: {by_status}; verdicts: {by_verdict}; "
        f"rejected lines: {len(parsed.errors)}; audit_status={audit_status}; automation_status={automation_status}; "
        "all rows are provisional MACHINE predictions, not human labels"
    )


# ---------------------------------------------------------------- export-grader


@app.command("export-grader")
def export_grader(
    grader_id: str = typer.Option(..., "--grader-id"),
    output: Path = typer.Option(..., "--output", help="Destination JSON file."),
) -> None:
    """Write a portable grader bundle (JSON, credential-free) for one grader version. Requires the database."""
    from eval_tinder.db.base import session_scope
    from eval_tinder.services.exports import ExportError, grader_bundle
    from eval_tinder.services.projects import NotFound, get_grader

    try:
        with session_scope() as s:
            grader = get_grader(s, grader_id)
            bundle = grader_bundle(s, grader)
    except NotFound as e:
        _fail(str(e))
    except ExportError as e:
        _fail(str(e))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    typer.echo(f"wrote grader bundle {grader_id} to {output} (manifest {bundle['manifest_hash'][:12]}...)")


if __name__ == "__main__":  # pragma: no cover
    app()


demo_app = typer.Typer(help="Seed and drive the synthetic cancellation demo.")
app.add_typer(demo_app, name="demo")


@demo_app.command("seed")
def demo_seed(
    name: str = typer.Option("Cancellation assistant (demo)", help="Project name."),
    simulate_expert: bool = typer.Option(False, "--simulate-expert", help="Answer the seed/DEV batches from the fixture truth table."),
    partition_seed: int = typer.Option(20260910, help="Seeded partition assignment."),
) -> None:
    """Create a demo project from backend/fixtures (SYNTHETIC data) with seed TRAIN and DEV review batches."""
    from eval_tinder.db.base import session_scope
    from eval_tinder.demo import seed_demo

    with session_scope() as session:
        result = seed_demo(session, name=name, partition_seed=partition_seed, simulate_expert=simulate_expert)
    typer.echo(json.dumps({k: v for k, v in result.items() if k != "line_errors"}, indent=1, default=str))
    if simulate_expert:
        _err("note: simulated judgments are stored with reviewer 'simulated-expert' and are not expert evidence")


@app.command()
def experiment(
    labeling_budget: int = typer.Option(32, help="Total human labels each strategy may use (TRAIN + DEV)."),
    batch_size: int = typer.Option(10, help="Labels per round."),
    max_metric_calls: int = typer.Option(60, help="GEPA metric-call budget per optimization round."),
    seed: int = typer.Option(1),
    output: Optional[Path] = typer.Option(None, help="Write the JSON report here (stdout otherwise)."),
) -> None:
    """Compare random versus committee selection under a fixed labeling budget (simulated expert; measured outcome)."""
    from eval_tinder.db.base import get_session_factory
    from eval_tinder.experiments.selection_experiment import run_experiment

    result = run_experiment(
        get_session_factory(), labeling_budget=labeling_budget, batch_size=batch_size,
        max_metric_calls=max_metric_calls, seed=seed,
    )
    text = json.dumps(result, indent=1, default=str)
    if output is not None:
        output.write_text(text)
        typer.echo(f"wrote {output}")
    else:
        typer.echo(text)
    typer.echo(json.dumps(result["comparison"], indent=1, default=str), err=True)
