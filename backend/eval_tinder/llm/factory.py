"""Build language models from configuration.

Only one real provider adapter exists (LiteLLM via ``dspy.LM``); everything else
uses the scripted fakes. Model identities come from ``GRADER_MODEL`` and
``REFLECTION_MODEL``; nothing here hardcodes a model ID.
"""
from __future__ import annotations

from typing import Any

import dspy
from dspy.clients.base_lm import BaseLM

from eval_tinder.config import Settings, get_settings
from eval_tinder.domain.manifest import ModelConfig
from eval_tinder.llm.budget import BudgetGuard
from eval_tinder.llm.fakes import (
    ScriptedGradingLM,
    ScriptedReflectionLM,
    keyword_switch_policy,
    truthful_reflection_proposer,
)


class ConfigurationError(RuntimeError):
    pass


class MeteredLM(BaseLM):
    """Wrap any legacy-contract LM with a ``BudgetGuard`` and per-role usage accounting."""

    forward_contract = "legacy"

    def __init__(self, inner: BaseLM, guard: BudgetGuard, role: str):
        super().__init__(
            model=inner.model,
            model_type=getattr(inner, "model_type", "chat"),
            temperature=inner.kwargs.get("temperature", 0.0),
            max_tokens=inner.kwargs.get("max_tokens", 4000),
            cache=False,
        )
        self.kwargs = dict(inner.kwargs)
        self.inner = inner
        self.guard = guard
        self.role = role

    def forward(self, prompt=None, messages=None, **kwargs):
        self.guard.reserve(self.role)
        try:
            response = self.inner.forward(prompt=prompt, messages=messages, **kwargs)
        except Exception:
            self.guard.release(self.role, None, failed=True)
            raise
        self.guard.release(self.role, _usage_dict(getattr(response, "usage", None)))
        return response

    async def aforward(self, prompt=None, messages=None, **kwargs):
        return self.forward(prompt=prompt, messages=messages, **kwargs)


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    return {k: getattr(usage, k, None) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}


def grader_model_config(settings: Settings | None = None) -> ModelConfig:
    settings = settings or get_settings()
    if settings.llm_provider == "fake":
        return ModelConfig(provider="fake", model="fake-grader", temperature=0.0, max_tokens=4000)
    if not settings.grader_model:
        raise ConfigurationError("GRADER_MODEL must be configured when LLM_PROVIDER is not 'fake'")
    return ModelConfig(
        provider="litellm",
        model=settings.grader_model,
        temperature=settings.grader_temperature,
        max_tokens=settings.grader_max_tokens,
    )


def build_grading_lm(model_config: ModelConfig, *, settings: Settings | None = None) -> BaseLM:
    settings = settings or get_settings()
    if model_config.provider == "fake":
        return ScriptedGradingLM(keyword_switch_policy, model=model_config.model)
    if model_config.provider == "litellm":
        return dspy.LM(
            model_config.model,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            cache=False,
            num_retries=2,
        )
    raise ConfigurationError(f"Unknown provider {model_config.provider!r}")


def build_reflection_lm(*, settings: Settings | None = None) -> BaseLM:
    settings = settings or get_settings()
    if settings.llm_provider == "fake":
        return ScriptedReflectionLM(truthful_reflection_proposer)
    if not settings.reflection_model:
        raise ConfigurationError("REFLECTION_MODEL must be configured when LLM_PROVIDER is not 'fake'")
    return dspy.LM(
        settings.reflection_model,
        temperature=1.0,
        max_tokens=settings.reflection_max_tokens,
        cache=False,
        num_retries=2,
    )


def new_budget_guard(settings: Settings | None = None, **overrides: int) -> BudgetGuard:
    settings = settings or get_settings()
    return BudgetGuard(
        max_calls=overrides.get("max_calls", settings.max_provider_calls_per_job),
        max_total_tokens=overrides.get("max_total_tokens", settings.max_total_tokens_per_job),
        max_tokens_per_call=overrides.get("max_tokens_per_call", settings.max_tokens_per_call),
    )
