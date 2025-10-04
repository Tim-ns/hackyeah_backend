from marshmallow import fields, Schema


class GetBestRouteSchema(Schema):
    suggested_routes = fields.String()
