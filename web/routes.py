from web.api import calculate_route as route_calc


def setup_routes(app):
    app.router.add_post('/calculate_route', route_calc.calculate_best_route)
