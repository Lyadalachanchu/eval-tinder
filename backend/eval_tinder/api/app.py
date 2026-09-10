"""FastAPI application factory."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from eval_tinder.services.jobs import JobError, LeaseLost
from eval_tinder.services.projects import NotFound
from eval_tinder.services.review import LeaseConflict, ReviewError, StaleSnapshot

log = logging.getLogger(__name__)


def create_app() -> FastAPI:
    from eval_tinder.config import get_settings

    app = FastAPI(title="eval-tinder", version="0.1.0")
    origins = get_settings().cors_origins
    if origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
        )

    @app.exception_handler(NotFound)
    async def _not_found(request: Request, exc: NotFound):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(LeaseConflict)
    async def _lease_conflict(request: Request, exc: LeaseConflict):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(StaleSnapshot)
    async def _stale(request: Request, exc: StaleSnapshot):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ReviewError)
    async def _review_error(request: Request, exc: ReviewError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(LeaseLost)
    async def _lease_lost(request: Request, exc: LeaseLost):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(JobError)
    async def _job_error(request: Request, exc: JobError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def _value_error(request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    from eval_tinder.api.routers import register_routers

    register_routers(app)

    @app.get("/health")
    def health():
        from eval_tinder.config import get_settings

        settings = get_settings()
        return {
            "status": "ok",
            "llm_provider": settings.llm_provider,
            "simulated": settings.llm_provider == "fake",
            "grader_model": None if settings.llm_provider == "fake" else settings.grader_model,
        }

    return app


app = create_app()
