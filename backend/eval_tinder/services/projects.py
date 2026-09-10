"""Projects, policy epochs, and grader versions."""
from __future__ import annotations

import random
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from eval_tinder.config import ProjectDefaults, Settings, get_settings
from eval_tinder.db.enums import GraderOrigin
from eval_tinder.db.models import GraderVersion, Project
from eval_tinder.domain.manifest import PARSER_VERSION, GraderManifest, ModelConfig
from eval_tinder.domain.rendering import RENDERER_VERSION, render_project_context
from eval_tinder.grader.signature import DEFAULT_SEED_INSTRUCTIONS
from eval_tinder.llm.factory import grader_model_config


class NotFound(LookupError):
    pass


def create_project(
    session: Session,
    *,
    name: str,
    description: str = "",
    partition_seed: int | None = None,
    configuration: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> Project:
    settings = settings or get_settings()
    config = settings.defaults.model_dump()
    config.update(configuration or {})
    ProjectDefaults.model_validate(config)  # validate shape
    project = Project(
        name=name,
        description=description or "",
        configuration=config,
        partition_seed=partition_seed if partition_seed is not None else random.SystemRandom().randrange(1, 2**31 - 1),
    )
    session.add(project)
    session.flush()
    create_seed_grader(session, project, settings=settings)
    return project


def get_project(session: Session, project_id: str) -> Project:
    project = session.get(Project, project_id)
    if project is None:
        raise NotFound(f"project {project_id} not found")
    return project


def project_config(project: Project) -> ProjectDefaults:
    return ProjectDefaults.model_validate(project.configuration or {})


def bump_policy_epoch(session: Session, project: Project, *, reason: str) -> Project:
    """A material change in what the expert wants starts a new epoch; old labels are revalidated, not overwritten."""
    project.policy_epoch += 1
    history = list((project.configuration or {}).get("policy_epoch_history", []))
    history.append({"epoch": project.policy_epoch, "reason": reason})
    project.configuration = {**(project.configuration or {}), "policy_epoch_history": history}
    session.flush()
    return project


def build_manifest(
    project: Project,
    *,
    instruction_text: str,
    model_config: ModelConfig,
    immutable_policy_context: str = "",
) -> GraderManifest:
    return GraderManifest(
        instruction_text=instruction_text,
        immutable_policy_context=immutable_policy_context,
        model_config=model_config,
        renderer_version=RENDERER_VERSION,
        parser_version=PARSER_VERSION,
        policy_epoch=project.policy_epoch,
    )


def create_grader_version(
    session: Session,
    project: Project,
    *,
    instruction_text: str,
    origin: str,
    parent_ids: list[str] | None = None,
    optimization_run_id: str | None = None,
    candidate_index: int | None = None,
    label: str = "",
    model_config: ModelConfig | None = None,
    immutable_policy_context: str | None = None,
    settings: Settings | None = None,
) -> GraderVersion:
    settings = settings or get_settings()
    model_config = model_config or grader_model_config(settings)
    policy_context = project.policy_notes if immutable_policy_context is None else immutable_policy_context
    manifest = build_manifest(
        project, instruction_text=instruction_text, model_config=model_config, immutable_policy_context=policy_context
    )
    grader = GraderVersion(
        project_id=project.id,
        label=label,
        origin=origin,
        parent_ids=parent_ids or [],
        optimization_run_id=optimization_run_id,
        candidate_index=candidate_index,
        instruction_text=instruction_text,
        immutable_policy_context=policy_context,
        model_config_=manifest.model_config_.sanitized().model_dump(),
        renderer_version=manifest.renderer_version,
        parser_version=manifest.parser_version,
        policy_epoch=project.policy_epoch,
        manifest=manifest.to_dict(),
        manifest_hash=manifest.manifest_hash,
    )
    session.add(grader)
    session.flush()
    return grader


def create_seed_grader(session: Session, project: Project, *, settings: Settings | None = None) -> GraderVersion:
    """The generic bootstrap grader. It is not a learned rubric and contains no task-specific rule."""
    return create_grader_version(
        session, project, instruction_text=DEFAULT_SEED_INSTRUCTIONS, origin=GraderOrigin.SEED,
        label="generic seed", settings=settings,
    )


def get_grader(session: Session, grader_id: str) -> GraderVersion:
    grader = session.get(GraderVersion, grader_id)
    if grader is None:
        raise NotFound(f"grader {grader_id} not found")
    return grader


def manifest_of(grader: GraderVersion) -> GraderManifest:
    return GraderManifest.from_dict(grader.manifest)


def seed_grader_for(session: Session, project: Project) -> GraderVersion:
    grader = session.scalar(
        select(GraderVersion)
        .where(GraderVersion.project_id == project.id, GraderVersion.origin == GraderOrigin.SEED)
        .order_by(GraderVersion.created_at)
    )
    if grader is None:
        grader = create_seed_grader(session, project)
    return grader


def project_context_for(project: Project, grader: GraderVersion) -> str:
    return render_project_context(project.description, grader.immutable_policy_context)


def list_graders(session: Session, project_id: str) -> list[GraderVersion]:
    return list(
        session.scalars(
            select(GraderVersion).where(GraderVersion.project_id == project_id).order_by(GraderVersion.created_at)
        )
    )
