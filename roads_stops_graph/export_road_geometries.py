import json
import pandas as pd
from neo4j import GraphDatabase
from neo4j_config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from typing import Dict, List, Any

class RoadGeometryExporter:
    def __init__(self, uri: str, user: str, password: str):
        """Initialize the Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()
    
    def export_road_geometries(self, output_file: str = "krakow_roads_with_geometry.geojson"):
        """Export road geometries from Neo4j to GeoJSON format."""
        with self.driver.session() as session:
            print("Exporting road geometries from Neo4j...")
            
            # Query to get all road segments with geometry
            query = """
            MATCH (u:RoadNode)-[r:CONNECTS]->(v:RoadNode)
            WHERE r.geometry IS NOT NULL
            RETURN 
                u.osmid as start_node,
                v.osmid as end_node,
                r.length_m as length,
                r.type as road_type,
                r.geometry as geometry
            """
            
            result = session.run(query)
            features = []
            
            for record in result:
                # Parse the geometry JSON string
                geometry = json.loads(record['geometry']) if isinstance(record['geometry'], str) else record['geometry']
                
                feature = {
                    "type": "Feature",
                    "properties": {
                        "start_node": record['start_node'],
                        "end_node": record['end_node'],
                        "length_m": record['length'],
                        "road_type": record['road_type']
                    },
                    "geometry": geometry
                }
                features.append(feature)
            
            # Create GeoJSON FeatureCollection
            geojson = {
                "type": "FeatureCollection",
                "features": features
            }
            
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(geojson, f, indent=2, ensure_ascii=False)
            
            print(f"Exported {len(features)} road segments with geometry to {output_file}")
            return geojson
    
    def export_road_network_summary(self):
        """Export a summary of the road network."""
        with self.driver.session() as session:
            print("\nRoad Network Summary:")
            print("=" * 50)
            
            # Count nodes
            result = session.run("MATCH (n:RoadNode) RETURN count(n) as node_count")
            node_count = result.single()["node_count"]
            print(f"Total road nodes: {node_count}")
            
            # Count relationships
            result = session.run("MATCH ()-[r:CONNECTS]->() RETURN count(r) as rel_count")
            rel_count = result.single()["rel_count"]
            print(f"Total road relationships: {rel_count}")
            
            # Count relationships with geometry
            result = session.run("""
                MATCH ()-[r:CONNECTS]->()
                RETURN count(r) as total, count(r.geometry) as with_geometry
            """)
            counts = result.single()
            print(f"Relationships with geometry: {counts['with_geometry']}")
            print(f"Geometry coverage: {(counts['with_geometry']/counts['total']*100):.1f}%")
            
            # Road type distribution
            result = session.run("""
                MATCH ()-[r:CONNECTS]->()
                RETURN r.type as road_type, count(r) as count
                ORDER BY count DESC
                LIMIT 10
            """)
            print("\nTop road types:")
            for record in result:
                print(f"  {record['road_type']}: {record['count']}")
            
            # Sample geometry
            result = session.run("""
                MATCH (u:RoadNode)-[r:CONNECTS]->(v:RoadNode)
                WHERE r.geometry IS NOT NULL
                RETURN r.geometry as geometry, r.type as road_type
                LIMIT 1
            """)
            sample = result.single()
            if sample:
                print(f"\nSample geometry (type: {sample['road_type']}):")
                geometry = json.loads(sample['geometry']) if isinstance(sample['geometry'], str) else sample['geometry']
                print(f"  Type: {geometry['type']}")
                print(f"  Coordinates: {len(geometry['coordinates'])} points")
                print(f"  First point: {geometry['coordinates'][0]}")
                print(f"  Last point: {geometry['coordinates'][-1]}")

def main():
    print("Road Geometry Exporter")
    print("=" * 50)
    
    exporter = RoadGeometryExporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Export road geometries
        geojson = exporter.export_road_geometries()
        
        # Show summary
        exporter.export_road_network_summary()
        
        print(f"\nSuccessfully exported road geometries!")
        print(f"Total features: {len(geojson['features'])}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        exporter.close()

if __name__ == "__main__":
    main()
