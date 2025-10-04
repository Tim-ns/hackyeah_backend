from logging import getLogger
from typing import Optional, Mapping

from aiohttp import ClientSession

from web.initializers.base import BaseInitializer

logger = getLogger(__name__)


class HTTP(BaseInitializer):
    def __init__(
        self,
        connection_settings: Mapping,
        app_attribute_name: str,
        service_name: Optional[str] = None,
    ):
        self.connection_settings = connection_settings
        self.app_attribute_name = app_attribute_name
        self.service_name = service_name or app_attribute_name

    async def init(self, app):
        app[self.app_attribute_name] = ClientSession(**self.connection_settings)
        logger.debug('%s connection is established', self.service_name)

    async def close(self, app):
        await app[self.app_attribute_name].close()
        logger.debug(f'%s connection is closed', self.service_name)
