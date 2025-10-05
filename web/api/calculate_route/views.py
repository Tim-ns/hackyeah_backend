import json

from aiohttp import web

from web.api.calculate_route.schemas import GetBestRouteSchema
from web.api.utils.validate_schema import validate_query
from settings import constants as const


@validate_query(GetBestRouteSchema)
async def calculate_best_route(request, validated_data, *args, **kwargs):
    route = "some_calculations"
    redis = request.app[const.REDIS_CONNECTION_NAME]
    await redis.set(const.AI_REDIS_KEY, 'test_value')
    resp = {
        "data": route,
    }
    return web.json_response(data={"data": resp}, status=200)
