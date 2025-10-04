from functools import wraps
from marshmallow import Schema, ValidationError
from aiohttp import web


def validate_body(schema_cls: type[Schema]):
    def decorator(func):
        @wraps(func)
        async def wrapper(request: web.Request, *args, **kwargs):
            schema = schema_cls()
            try:
                data = await request.json()
                validated_data = schema.load(data)
            except ValidationError as err:
                raise web.HTTPBadRequest(text=str(err.messages))
            except Exception:
                raise web.HTTPBadRequest(text="Invalid JSON")
            return await func(request, validated_data, *args, **kwargs)
        return wrapper
    return decorator


def validate_query(schema_cls: type[Schema]):
    def decorator(func):
        @wraps(func)
        async def wrapper(request: web.Request, *args, **kwargs):
            schema = schema_cls()
            try:
                validated_data = schema.load(dict(request.query))
            except ValidationError as err:
                raise web.HTTPBadRequest(text=str(err.messages))
            return await func(request, validated_data, *args, **kwargs)
        return wrapper
    return decorator
