"""Database-backed test fixtures.

The schema is created by running the real Alembic migrations once per session
against the test database; tables are truncated between tests.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import text

from eval_tinder.config import get_settings, reset_settings_cache
from eval_tinder.db.base import Base, configure_engine, get_session_factory
from eval_tinder.db import models  # noqa: F401

BACKEND_DIR = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def test_database_url() -> str:
    return os.environ.get("TEST_DATABASE_URL") or get_settings().test_database_url


def _maintenance_url(url: str) -> str:
    base, _, _dbname = url.rpartition("/")
    return f"{base}/postgres"


@pytest.fixture(scope="session")
def migrated_engine(test_database_url: str, tmp_path_factory):
    """Create a private database for this test process so parallel runs never share tables."""
    from sqlalchemy import create_engine

    base, _, dbname = test_database_url.rpartition("/")
    private_name = f"{dbname}_{os.getpid()}"
    private_url = f"{base}/{private_name}"
    admin = create_engine(_maintenance_url(test_database_url), isolation_level="AUTOCOMMIT", future=True)
    with admin.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{private_name}"'))
        conn.execute(text(f'CREATE DATABASE "{private_name}"'))
    env = {**os.environ, "ALEMBIC_DATABASE_URL": private_url}
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=BACKEND_DIR, env=env, check=True,
                   capture_output=True)
    engine = configure_engine(private_url)
    artifact_dir = tmp_path_factory.mktemp("artifacts")
    os.environ["ARTIFACT_PATH"] = str(artifact_dir)
    os.environ["DATABASE_URL"] = private_url
    reset_settings_cache()
    yield engine
    engine.dispose()
    with admin.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{private_name}" WITH (FORCE)'))
    admin.dispose()


@pytest.fixture
def db_session(migrated_engine):
    """A session on a clean database. Each test starts with empty tables."""
    with migrated_engine.begin() as conn:
        tables = ", ".join(f'"{t.name}"' for t in Base.metadata.sorted_tables)
        conn.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.rollback()
    finally:
        session.close()


@pytest.fixture
def session_factory(migrated_engine):
    return get_session_factory()


@pytest.fixture
def settings(migrated_engine):
    reset_settings_cache()
    return get_settings()
