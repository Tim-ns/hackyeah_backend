import platform
import click
from aiohttp import web

from settings import constants
from web.app_builder import App
from web.routes import setup_routes


@click.group()
def cli():
    if platform.system() != 'Windows':
        import uvloop
        uvloop.install()


@cli.command(short_help='start web')
def start():
    app = App.create(client_max_size=constants.AppConfig.CLIENT_MAX_SIZE)
    setup_routes(app)
    web.run_app(app, port=constants.AppConfig.PORT)


if __name__ == '__main__':
    cli()
