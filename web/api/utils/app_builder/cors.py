from collections import defaultdict
from typing import Iterable, Optional

import aiohttp_cors
from aiohttp.abc import Application
from marshmallow import validate

__all__ = ('register_cors_routes',)


def origin_url_validator(origin_value: str) -> str:
    url_validator = validate.URL()
    if origin_value == '*':
        return '*'

    return url_validator(origin_value)


def register_cors_routes(app: Application,
                         origin_whitelist: Iterable[str],
                         allow_headers: Iterable[str],
                         expose_headers: Iterable[str],
                         allow_credentials: Optional[bool] = False,
                         max_age: Optional[int] = 0) -> None:
    if not origin_whitelist:
        return
    origin_whitelist = tuple(origin_url_validator(origin) for origin in origin_whitelist)

    cors = aiohttp_cors.setup(app)

    resource_methods_map = defaultdict(list)

    for route in app.router.routes():
        resource_methods_map[route.resource.canonical].append(route.method)

    if '*' in allow_headers:
        allow_headers = '*'

    if '*' in expose_headers:
        expose_headers = '*'

    for resource in app.router.resources():
        cors_options = aiohttp_cors.ResourceOptions(
            allow_credentials=allow_credentials,
            max_age=max_age,
            allow_methods=resource_methods_map[resource.canonical],
            allow_headers=allow_headers,
            expose_headers=expose_headers
        )
        cors_settings = dict.fromkeys(origin_whitelist, cors_options)
        cors.add(resource, cors_settings)
