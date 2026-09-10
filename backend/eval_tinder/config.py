"""Application configuration.

Everything operational (models, credentials, database, budgets) comes from the
environment or a ``.env`` file. Numeric defaults are engineering starting points,
not sample-size guarantees.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ReviewBatchDefaults(BaseModel):
    disagreement: int = 6
    coverage: int = 2
    random: int = 2


class ProjectDefaults(BaseModel):
    """Per-project configuration defaults (copied into ``Project.configuration``)."""

    bootstrap_train_labels: int = 12
    bootstrap_dev_labels: int = 8
    new_train_labels_per_round: int = 10
    gepa_max_metric_calls: int = 300
    gepa_reflection_minibatch_size: int = 3
    gepa_num_threads: int = 2
    committee_size: int = 4
    committee_max_shortlist: int = 12
    committee_quality_gap: float = 0.10
    committee_probe_size: int = 50
    selection_pool_size: int = 200
    review_batch: ReviewBatchDefaults = Field(default_factory=ReviewBatchDefaults)
    automatic_optimization: bool = False
    automation_enabled: bool = False
    partition_split: dict[str, float] = Field(
        default_factory=lambda: {"TRAIN": 0.70, "DEV": 0.15, "AUDIT_RESERVE": 0.15}
    )
    dev_topup_cap: int = 40


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+psycopg://postgres:postgres@127.0.0.1:5432/eval_tinder"
    test_database_url: str = "postgresql+psycopg://postgres:postgres@127.0.0.1:5432/eval_tinder_test"

    # Model identities are configuration, never hardcoded in the grader.
    grader_model: str | None = None
    reflection_model: str | None = None
    grader_temperature: float = 1.0
    grader_max_tokens: int = 4000
    reflection_max_tokens: int = 16000
    llm_provider: str = "fake"  # "fake" for deterministic tests/demo; "litellm" for a real provider

    artifact_path: Path = Path("./artifacts")
    max_upload_bytes: int = 25 * 1024 * 1024
    max_case_chars: int = 40_000  # rendered-case input budget before REVIEW/CONTEXT_TOO_LARGE

    # Application-level provider budgets (a metric-call budget is not a token budget).
    max_provider_calls_per_job: int = 2000
    max_tokens_per_call: int = 16000
    max_total_tokens_per_job: int = 20_000_000

    # Optional pricing table: {"openai/gpt-x": {"input_per_1k": 0.001, "output_per_1k": 0.002}}
    pricing_table: dict[str, dict[str, float]] = Field(default_factory=dict)

    lease_seconds: int = 900
    job_lease_seconds: int = 120
    api_token: str | None = None  # single-expert local deployments may leave this unset (loopback only)
    reviewer_id: str = "local-expert"

    defaults: ProjectDefaults = Field(default_factory=ProjectDefaults)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
