from abc import ABC, abstractmethod
from logging import getLogger

logger = getLogger(__name__)


class BaseInitializer(ABC):
    @abstractmethod
    async def init(self, app):
        pass

    @abstractmethod
    async def close(self, app):
        pass

    async def check_status(self, app):
        return True

    async def __call__(self, app):
        await self.init(app)
        app.on_cleanup.insert(0, self.close)
