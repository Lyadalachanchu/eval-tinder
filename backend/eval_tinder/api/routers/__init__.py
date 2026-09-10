"""Router registry. Each feature area owns one module exposing ``router``."""
from __future__ import annotations

import importlib

from fastapi import FastAPI

ROUTER_MODULES = [
    "eval_tinder.api.routers.projects",
    "eval_tinder.api.routers.imports",
    "eval_tinder.api.routers.review",
    "eval_tinder.api.routers.optimization",
    "eval_tinder.api.routers.graders",
    "eval_tinder.api.routers.jobs",
    "eval_tinder.api.routers.selection",
    "eval_tinder.api.routers.grading",
    "eval_tinder.api.routers.audits",
    "eval_tinder.api.routers.automation",
    "eval_tinder.api.routers.exports",
    "eval_tinder.api.routers.traces",
]


def register_routers(app: FastAPI) -> None:
    for name in ROUTER_MODULES:
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as e:  # a feature area not yet implemented
            if e.name == name:
                continue
            raise
        app.include_router(module.router)
