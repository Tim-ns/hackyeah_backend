from functools import cached_property

from settings.constants import RedisConfig
from web.initializers.http import HTTP
from web.initializers.redis_pool import Redis
from settings import constants as const


class InitApp:

    @cached_property
    def init_http_session(self):
        return HTTP(
            connection_settings={},
            app_attribute_name=const.HTTP_SESSION_NAME
        )

    @cached_property
    def redis_pool(self):
        return Redis(
            connection_settings=RedisConfig.CONNECTION_SETTINGS,
            app_attribute_name=const.REDIS_CONNECTION_NAME,
        )

    @property
    def all(self):
        return (
            self.init_http_session,
            self.redis_pool,
        )

app_init = InitApp()
