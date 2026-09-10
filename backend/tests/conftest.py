from __future__ import annotations

import logging
import os

import pytest

os.environ.setdefault("LLM_PROVIDER", "fake")
os.environ.setdefault("TQDM_DISABLE", "1")
logging.getLogger("dspy").setLevel(logging.WARNING)
logging.getLogger("gepa").setLevel(logging.WARNING)
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://postgres:postgres@127.0.0.1:5432/eval_tinder_test")


def pytest_collection_modifyitems(config, items):
    run_real_gepa = os.environ.get("REAL_GEPA") == "1"
    run_real_model = os.environ.get("REAL_MODEL") == "1"
    for item in items:
        if "real_gepa" in item.keywords and not run_real_gepa:
            item.add_marker(pytest.mark.skip(reason="set REAL_GEPA=1 to run the real-model GEPA integration test"))
        if "real_model" in item.keywords and not run_real_model:
            item.add_marker(pytest.mark.skip(reason="set REAL_MODEL=1 (and provider credentials) to run"))
