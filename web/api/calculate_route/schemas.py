from marshmallow import fields, Schema


class GetBestRouteSchema(Schema):
    suggested_routes = fields.List(fields.String(), required=True, allow_none=False)
