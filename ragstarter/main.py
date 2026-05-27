from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ragstarter.api.routes import router
from ragstarter.core.config import get_settings
from ragstarter.core.logging import configure_logging
from ragstarter.api.deps import get_service_container


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    services = get_service_container()
    app.state.services = services
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()
