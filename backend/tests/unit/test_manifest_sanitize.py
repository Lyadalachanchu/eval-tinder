from __future__ import annotations

import pytest

from eval_tinder.domain.manifest import GraderManifest, ModelConfig


def test_nested_credentials_are_stripped_from_model_config_extra():
    cfg = ModelConfig(provider="litellm", model="m", extra={"extra_headers": {"Authorization": "Bearer abc"}, "timeout": 5})
    cleaned = cfg.sanitized()
    assert cleaned.extra == {"extra_headers": {}, "timeout": 5}
    manifest = GraderManifest(instruction_text="x", model_config=cfg)
    assert "Bearer" not in str(manifest.to_dict())
    with pytest.raises(ValueError, match="credential"):
        GraderManifest.from_dict({"instruction_text": "x", "model_config": {"provider": "litellm", "model": "m",
                                                                          "extra": {"nested": {"api_key": "sk-1"}}}})
