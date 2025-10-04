from aiohttp import web
from settings import constants

from web.api.utils.app_builder.app_builder import AppBuilder
from web.initializers.initializer import app_init


async def startup_services(app: web.Application):
    pass


class App(AppBuilder):
    on_start = list(app_init.all) + [startup_services]
    cors_settings = constants.AppConfig.CORS
    base_url = constants.AppConfig.BASE_URL
