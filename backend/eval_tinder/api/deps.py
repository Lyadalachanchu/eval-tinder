"""FastAPI dependencies: database session, settings, and a simple bearer-token guard."""
from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from eval_tinder.config import Settings, get_settings
from eval_tinder.db.base import get_session_factory


def get_db() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def settings_dep() -> Settings:
    return get_settings()


def require_auth(
    request: Request,
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(settings_dep),
) -> str:
    """Single-expert local deployments may run without a token on loopback only.

    When API_TOKEN is configured every request must carry ``Authorization: Bearer <token>``.
    """
    if settings.api_token:
        if authorization != f"Bearer {settings.api_token}":
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")
        return settings.reviewer_id
    client = request.client.host if request.client else ""
    if client not in ("127.0.0.1", "::1", "testclient", ""):
        raise HTTPException(status_code=401, detail="API_TOKEN must be configured for non-loopback access")
    return settings.reviewer_id
