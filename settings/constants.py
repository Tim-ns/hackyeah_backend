from .base import env
import os


HTTP_SESSION_NAME='HTTP_SESSION_NAME'
REDIS_CONNECTION_NAME='REDIS_CONNECTION_NAME'

class RedisConfig:
    CONNECTION_SETTINGS = {
        'url': env.str('REDIS_URL', default='redis://100.74.106.106:6379'),
    }
    LOCK_TIMEOUT = env.int('REDIS_LOCK_TIMEOUT', default=60)
    LONG_LOCK_TIMEOUT = env.int('REDIS_LONG_LOCK_TIMEOUT', default=120)


class AppConfig:
    CLIENT_MAX_SIZE = 5
    FAILED_MESSAGE_MAX_RETRIES = 5
    FAILED_MESSAGE_DELAY_SEC = 1
    CORS = {
        'origin_whitelist': tuple(env.list('APP_ORIGIN_WHITELIST', default=['*'])),
        'allow_headers': tuple(env.list('APP_ALLOW_HEADERS', default=['*'])),
        'expose_headers': tuple(env.list('APP_EXPOSE_HEADERS', default=['*'])),
        'allow_credentials': env.bool('APP_ALLOW_CREDENTIALS', default=False),
        'max_age': env.int('APP_MAX_AGE', default=0),
    }
    BASE_URL = os.environ.get('APP_BASE_URL', default='/')
    PORT = 8015


AI_REDIS_KEY = 'AI_REDIS_KEY'
