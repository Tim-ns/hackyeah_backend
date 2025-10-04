import osmnx as ox
import pandas as pd
import json
from shapely.geometry import LineString

G = ox.graph_from_place("Kraków, Poland", network_type="drive")
osm_nodes, osm_edges = ox.graph_to_gdfs(G)

# osmid is the index, so we need to reset it to make it a column
osm_nodes = osm_nodes.reset_index()
osm_nodes = osm_nodes[['osmid', 'y', 'x']].copy()
osm_nodes.rename(columns={'y': 'lat', 'x': 'lon'}, inplace=True)
osm_nodes['label'] = 'RoadNode'

# Process edges to preserve geometry
osm_edges = osm_edges.reset_index()
osm_edges['length_m'] = osm_edges['length']
osm_edges['type'] = osm_edges['highway']
osm_edges['u'] = osm_edges['u']
osm_edges['v'] = osm_edges['v']

# Extract geometry as LineString and convert to GeoJSON format
def geometry_to_geojson(geom):
    """Convert Shapely geometry to GeoJSON format."""
    if geom is None:
        return None
    return json.loads(json.dumps(geom.__geo_interface__))

# Add geometry information
osm_edges['geometry'] = osm_edges['geometry'].apply(geometry_to_geojson)

# Create road segments with geometry
road_segments_df = osm_edges[['u', 'v', 'length_m', 'type', 'geometry']].copy()

NODES_FILE = 'krakow_road_nodes.csv'
RELATIONS_FILE = 'krakow_road_segments.csv'

osm_nodes.to_csv(NODES_FILE, index=False)
print(f"Saved road nodes to: {NODES_FILE}")

# Save road segments with geometry as JSON string
road_segments_df['geometry'] = road_segments_df['geometry'].apply(lambda x: json.dumps(x) if x is not None else None)
road_segments_df.to_csv(RELATIONS_FILE, index=False)
print(f"Saved road segments to: {RELATIONS_FILE}")

print(f"Total nodes: {len(osm_nodes)}")
print(f"Total road segments: {len(road_segments_df)}")
print(f"Segments with geometry: {road_segments_df['geometry'].notna().sum()}")