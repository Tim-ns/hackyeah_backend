from typing import Iterable, Callable, List, Optional, Mapping

from aiohttp.web_app import Application
from aiohttp.web_routedef import RouteDef

from web.api.utils.app_builder import cors


class AppBuilder:
    routes: Iterable[RouteDef] = None
    on_start: Optional[List[Callable]] = None
    on_shutdown: Optional[List[Callable]] = None
    middlewares: Optional[List[Callable]] = None
    cors_settings: Optional[Mapping] = None
    doc_settings: Optional[Mapping] = None
    client_max_size: Optional[int] = None
    model_spec_settings: Optional[Mapping] = None
    base_url: str = '/'

    @classmethod
    def create(cls, **kwargs) -> Application:
        if cls.client_max_size and not kwargs.get('client_max_size'):
            kwargs['client_max_size'] = cls.client_max_size

        app = Application(**kwargs)
        routes = cls.routes or []
        app.add_routes(routes)

        if cls.on_start:
            app.on_startup.extend(cls.on_start)
        if cls.on_shutdown:
            app.on_shutdown.extend(cls.on_shutdown)
        if cls.middlewares:
            app.middlewares.extend(cls.middlewares)

        if cls.cors_settings:
            cors.register_cors_routes(app, **cls.cors_settings)

        return app
