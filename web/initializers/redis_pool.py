from logging import getLogger
from redis import asyncio as aioredis

from web.initializers.base import BaseInitializer

logger = getLogger(__name__)


class Redis(BaseInitializer):
    def __init__(self, connection_settings, app_attribute_name: str = 'REDIS'):
        self.connection_settings = connection_settings
        self.app_attribute_name = app_attribute_name

    async def init(self, app):
        app[self.app_attribute_name] = aioredis.from_url(**self.connection_settings)
        logger.debug('Redis connection is established')

    async def close(self, app):
        await app[self.app_attribute_name].close()
        logger.debug('Redis connection is closed')

    async def check_status(self, app):
        try:
            redis_pool = app[self.app_attribute_name]
            return await redis_pool.ping()
        except Exception as exc:
            logger.error(exc)
            return False
