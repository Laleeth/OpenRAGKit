from __future__ import annotations

from functools import lru_cache

from fastapi import Request

from ragstarter.core.config import Settings, get_settings
from ragstarter.services.container import Services, build_services


@lru_cache(maxsize=1)
def get_service_container() -> Services:
    return build_services(get_settings())


def get_services(request: Request | None = None) -> Services:
    if request is not None and hasattr(request.app.state, "services"):
        return request.app.state.services
    return get_service_container()
