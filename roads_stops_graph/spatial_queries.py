import json
import pandas as pd
from neo4j import GraphDatabase
from neo4j_config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from typing import Dict, List, Any, Tuple

class RoadSpatialQueries:
    def __init__(self, uri: str, user: str, password: str):
        """Initialize the Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()
    
    def find_roads_near_point(self, lat: float, lon: float, radius_m: float = 1000):
        """Find roads within a certain radius of a point."""
        with self.driver.session() as session:
            query = """
            MATCH (n:RoadNode)
            WHERE point.distance(n.location, point({latitude: $lat, longitude: $lon})) <= $radius
            MATCH (n)-[r:CONNECTS]->(m:RoadNode)
            WHERE r.geometry IS NOT NULL
            RETURN 
                n.osmid as start_node,
                m.osmid as end_node,
                r.length_m as length,
                r.type as road_type,
                r.geometry as geometry,
                point.distance(n.location, point({latitude: $lat, longitude: $lon})) as distance
            ORDER BY distance
            """
            
            result = session.run(query, lat=lat, lon=lon, radius=radius_m)
            roads = []
            
            for record in result:
                geometry = json.loads(record['geometry']) if isinstance(record['geometry'], str) else record['geometry']
                roads.append({
                    'start_node': record['start_node'],
                    'end_node': record['end_node'],
                    'length_m': record['length'],
                    'road_type': record['road_type'],
                    'geometry': geometry,
                    'distance_m': record['distance']
                })
            
            return roads
    
    def find_roads_by_type(self, road_type: str, limit: int = 100):
        """Find roads of a specific type."""
        with self.driver.session() as session:
            query = """
            MATCH (u:RoadNode)-[r:CONNECTS]->(v:RoadNode)
            WHERE r.type = $road_type AND r.geometry IS NOT NULL
            RETURN 
                u.osmid as start_node,
                v.osmid as end_node,
                r.length_m as length,
                r.type as road_type,
                r.geometry as geometry
            LIMIT $limit
            """
            
            result = session.run(query, road_type=road_type, limit=limit)
            roads = []
            
            for record in result:
                geometry = json.loads(record['geometry']) if isinstance(record['geometry'], str) else record['geometry']
                roads.append({
                    'start_node': record['start_node'],
                    'end_node': record['end_node'],
                    'length_m': record['length'],
                    'road_type': record['road_type'],
                    'geometry': geometry
                })
            
            return roads
    
    def find_roads_in_bounding_box(self, min_lat: float, min_lon: float, max_lat: float, max_lon: float):
        """Find roads within a bounding box."""
        with self.driver.session() as session:
            query = """
            MATCH (u:RoadNode)-[r:CONNECTS]->(v:RoadNode)
            WHERE r.geometry IS NOT NULL
            AND u.location.latitude >= $min_lat AND u.location.latitude <= $max_lat
            AND u.location.longitude >= $min_lon AND u.location.longitude <= $max_lon
            RETURN 
                u.osmid as start_node,
                v.osmid as end_node,
                r.length_m as length,
                r.type as road_type,
                r.geometry as geometry
            """
            
            result = session.run(query, 
                               min_lat=min_lat, min_lon=min_lon, 
                               max_lat=max_lat, max_lon=max_lon)
            roads = []
            
            for record in result:
                geometry = json.loads(record['geometry']) if isinstance(record['geometry'], str) else record['geometry']
                roads.append({
                    'start_node': record['start_node'],
                    'end_node': record['end_node'],
                    'length_m': record['length'],
                    'road_type': record['road_type'],
                    'geometry': geometry
                })
            
            return roads
    
    def export_roads_to_geojson(self, roads: List[Dict], output_file: str):
        """Export a list of roads to GeoJSON format."""
        features = []
        
        for road in roads:
            feature = {
                "type": "Feature",
                "properties": {
                    "start_node": road['start_node'],
                    "end_node": road['end_node'],
                    "length_m": road['length_m'],
                    "road_type": road['road_type']
                },
                "geometry": road['geometry']
            }
            
            # Add distance if available
            if 'distance_m' in road:
                feature['properties']['distance_m'] = road['distance_m']
            
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)
        
        print(f"Exported {len(features)} roads to {output_file}")
        return geojson

def main():
    print("Road Spatial Queries Demo")
    print("=" * 50)
    
    spatial_queries = RoadSpatialQueries(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Kraków center coordinates
        krakow_center_lat = 50.0647
        krakow_center_lon = 19.9450
        
        print(f"1. Finding roads near Kraków center ({krakow_center_lat}, {krakow_center_lon})")
        print("-" * 60)
        roads_near_center = spatial_queries.find_roads_near_point(
            krakow_center_lat, krakow_center_lon, radius_m=500
        )
        print(f"Found {len(roads_near_center)} roads within 500m of center")
        
        if roads_near_center:
            print("Sample roads:")
            for i, road in enumerate(roads_near_center[:3]):
                print(f"  {i+1}. {road['road_type']} road, {road['length_m']:.1f}m, distance: {road['distance_m']:.1f}m")
        
        # Export to GeoJSON
        spatial_queries.export_roads_to_geojson(roads_near_center, "roads_near_center.geojson")
        
        print(f"\n2. Finding highway roads")
        print("-" * 60)
        highway_roads = spatial_queries.find_roads_by_type("primary", limit=50)
        print(f"Found {len(highway_roads)} primary roads")
        
        if highway_roads:
            print("Sample primary roads:")
            for i, road in enumerate(highway_roads[:3]):
                print(f"  {i+1}. Length: {road['length_m']:.1f}m")
        
        # Export to GeoJSON
        spatial_queries.export_roads_to_geojson(highway_roads, "primary_roads.geojson")
        
        print(f"\n3. Finding roads in bounding box (Kraków area)")
        print("-" * 60)
        # Define a bounding box around Kraków
        bbox_roads = spatial_queries.find_roads_in_bounding_box(
            min_lat=50.0, min_lon=19.8, max_lat=50.1, max_lon=20.1
        )
        print(f"Found {len(bbox_roads)} roads in bounding box")
        
        # Export to GeoJSON
        spatial_queries.export_roads_to_geojson(bbox_roads, "krakow_bbox_roads.geojson")
        
        print(f"\n4. Road type distribution")
        print("-" * 60)
        with spatial_queries.driver.session() as session:
            result = session.run("""
                MATCH ()-[r:CONNECTS]->()
                WHERE r.geometry IS NOT NULL
                RETURN r.type as road_type, count(r) as count
                ORDER BY count DESC
                LIMIT 10
            """)
            
            for record in result:
                print(f"  {record['road_type']}: {record['count']}")
        
        print(f"\nSpatial queries completed successfully!")
        print(f"Generated files:")
        print(f"  - roads_near_center.geojson")
        print(f"  - primary_roads.geojson") 
        print(f"  - krakow_bbox_roads.geojson")
        
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        spatial_queries.close()

if __name__ == "__main__":
    main()
