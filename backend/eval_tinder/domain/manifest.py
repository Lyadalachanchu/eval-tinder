"""Grader manifests and pipeline hashes.

A manifest is a JSON document that fully describes a grader version's behavior
configuration: instruction text, immutable policy context, model config,
renderer and parser versions. Credentials are never part of it.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from eval_tinder.domain.rendering import RENDERER_VERSION
from eval_tinder.ids import hash_value

PARSER_VERSION = "p2"
METRIC_VERSION = "agreement-v1"
MANIFEST_SCHEMA_VERSION = 1

FORBIDDEN_MODEL_CONFIG_KEYS = {"api_key", "api_base_key", "authorization", "token", "secret", "password"}
_FORBIDDEN_FRAGMENTS = ("api_key", "apikey", "authorization", "token", "secret", "password", "credential")


def _is_credential_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in FORBIDDEN_MODEL_CONFIG_KEYS or any(f in lowered for f in _FORBIDDEN_FRAGMENTS)


def _strip_credentials(value: Any) -> Any:
    """Recursively drop credential-like keys from nested provider settings."""
    if isinstance(value, dict):
        return {k: _strip_credentials(v) for k, v in value.items() if not _is_credential_key(str(k))}
    if isinstance(value, list):
        return [_strip_credentials(v) for v in value]
    return value


class ModelConfig(BaseModel):
    provider: str = "fake"  # fake | litellm
    model: str = "fake-grader"
    temperature: float = 1.0
    max_tokens: int = 4000
    extra: dict[str, Any] = Field(default_factory=dict)

    def sanitized(self) -> "ModelConfig":
        return self.model_copy(update={"extra": _strip_credentials(self.extra)})


class GraderManifest(BaseModel):
    schema_version: int = MANIFEST_SCHEMA_VERSION
    predictor_name: str = "judge"
    instruction_text: str
    immutable_policy_context: str = ""
    model_config_: ModelConfig = Field(alias="model_config", default_factory=ModelConfig)
    renderer_version: str = RENDERER_VERSION
    parser_version: str = PARSER_VERSION
    policy_epoch: int = 1

    model_config = {"populate_by_name": True, "protected_namespaces": ()}

    def to_dict(self) -> dict[str, Any]:
        d = self.model_dump(by_alias=True)
        d["model_config"] = self.model_config_.sanitized().model_dump()
        return d

    @property
    def manifest_hash(self) -> str:
        return hash_value(self.to_dict())

    @property
    def prompt_hash(self) -> str:
        """Hash of the editable prompt content only (instruction + policy context)."""
        return hash_value({"instruction_text": self.instruction_text, "policy": self.immutable_policy_context})

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GraderManifest":
        extra = (d.get("model_config") or {}).get("extra") or {}
        if _strip_credentials(extra) != extra:
            raise ValueError("Manifest may not carry credential-like keys in model_config.extra")
        return cls.model_validate(d)


def pipeline_hash(manifest: GraderManifest, *, routing_rules: dict[str, Any] | None = None) -> str:
    """Model + prompt + policy + renderer + parser + routing => one pipeline identity."""
    return hash_value({"manifest": manifest.to_dict(), "routing": routing_rules or {}})
