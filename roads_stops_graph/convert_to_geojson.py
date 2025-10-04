import json
import pandas as pd
from typing import Dict, List, Tuple

def load_node_coordinates(nodes_csv_path: str) -> Dict[int, Tuple[float, float]]:
    """Load node coordinates from CSV file."""
    print("Loading node coordinates...")
    df = pd.read_csv(nodes_csv_path)
    coordinates = {}
    
    for _, row in df.iterrows():
        coordinates[row['osmid']] = (row['lon'], row['lat'])  # GeoJSON uses [longitude, latitude]
    
    print(f"Loaded coordinates for {len(coordinates)} nodes")
    return coordinates

def convert_to_geojson(query_data: List[Dict], coordinates: Dict[int, Tuple[float, float]]) -> Dict:
    """Convert Neo4j query data to GeoJSON format."""
    
    # Remove duplicates (I notice there are duplicate entries in the data)
    unique_segments = []
    seen = set()
    
    for segment in query_data:
        key = (segment['start_node'], segment['connected_node'])
        if key not in seen:
            unique_segments.append(segment)
            seen.add(key)
    
    print(f"Processing {len(unique_segments)} unique road segments...")
    
    features = []
    missing_coords = 0
    
    for segment in unique_segments:
        start_node = segment['start_node']
        end_node = segment['connected_node']
        
        # Get coordinates for both nodes
        start_coords = coordinates.get(start_node)
        end_coords = coordinates.get(end_node)
        
        if start_coords and end_coords:
            # Create LineString feature
            feature = {
                "type": "Feature",
                "properties": {
                    "start_node": start_node,
                    "end_node": end_node,
                    "road_type": segment['road_type'],
                    "length_m": segment['length']
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [start_coords, end_coords]
                }
            }
            features.append(feature)
        else:
            missing_coords += 1
            if missing_coords <= 5:  # Only show first few missing coordinates
                print(f"Warning: Missing coordinates for nodes {start_node} or {end_node}")
    
    if missing_coords > 0:
        print(f"Total missing coordinates: {missing_coords}")
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    return geojson

def create_point_geojson(query_data: List[Dict], coordinates: Dict[int, Tuple[float, float]]) -> Dict:
    """Create a GeoJSON with individual points for each node."""
    
    # Get unique nodes
    unique_nodes = set()
    for segment in query_data:
        unique_nodes.add(segment['start_node'])
        unique_nodes.add(segment['connected_node'])
    
    print(f"Creating point features for {len(unique_nodes)} unique nodes...")
    
    features = []
    for node_id in unique_nodes:
        coords = coordinates.get(node_id)
        if coords:
            feature = {
                "type": "Feature",
                "properties": {
                    "osmid": node_id,
                    "node_type": "road_intersection"
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": coords
                }
            }
            features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    return geojson

def main():
    # File paths
    query_file = "roads_stops_graph/neo4j_query_table_data_2025-10-4.json"
    nodes_file = "roads_stops_graph/krakow_road_nodes.csv"
    
    print("Converting Neo4j query data to GeoJSON...")
    print("=" * 50)
    
    # Load query data
    print(f"Loading query data from {query_file}...")
    with open(query_file, 'r') as f:
        query_data = json.load(f)
    
    print(f"Loaded {len(query_data)} road segments")
    
    # Load node coordinates
    coordinates = load_node_coordinates(nodes_file)
    
    # Convert to GeoJSON (LineString features)
    print("\nCreating LineString GeoJSON...")
    line_geojson = convert_to_geojson(query_data, coordinates)
    
    # Save LineString GeoJSON
    line_output_file = "road_segments.geojson"
    with open(line_output_file, 'w') as f:
        json.dump(line_geojson, f, indent=2)
    
    print(f"Saved LineString GeoJSON to: {line_output_file}")
    print(f"Created {len(line_geojson['features'])} LineString features")
    
    # Convert to GeoJSON (Point features)
    print("\nCreating Point GeoJSON...")
    point_geojson = create_point_geojson(query_data, coordinates)
    
    # Save Point GeoJSON
    point_output_file = "road_nodes.geojson"
    with open(point_output_file, 'w') as f:
        json.dump(point_geojson, f, indent=2)
    
    print(f"Saved Point GeoJSON to: {point_output_file}")
    print(f"Created {len(point_geojson['features'])} Point features")
    
    # Display summary
    print("\n" + "=" * 50)
    print("CONVERSION SUMMARY")
    print("=" * 50)
    print(f"Input segments: {len(query_data)}")
    print(f"Unique segments: {len(line_geojson['features'])}")
    print(f"Unique nodes: {len(point_geojson['features'])}")
    print(f"LineString GeoJSON: {line_output_file}")
    print(f"Point GeoJSON: {point_output_file}")
    
    # Show sample features
    print("\nSample LineString feature:")
    if line_geojson['features']:
        sample = line_geojson['features'][0]
        print(json.dumps(sample, indent=2))
    
    print("\nSample Point feature:")
    if point_geojson['features']:
        sample = point_geojson['features'][0]
        print(json.dumps(sample, indent=2))

if __name__ == "__main__":
    main()
