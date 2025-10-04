from web.api import calculate_route

def setup_routes(app):
    app.router.add_get('', calculate_route.get_best_route)