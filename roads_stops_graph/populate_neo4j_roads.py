import pandas as pd
from neo4j import GraphDatabase
import os
from typing import List, Dict, Any
from neo4j_config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NODES_FILE, SEGMENTS_FILE

class RoadNetworkPopulator:
    def __init__(self, uri: str, user: str, password: str):
        """Initialize the Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()
    
    def clear_database(self):
        """Clear all existing road network data."""
        with self.driver.session() as session:
            print("Clearing existing road network data...")
            session.run("MATCH (n:RoadNode) DETACH DELETE n")
            print("Database cleared.")
    
    def create_road_nodes(self, nodes_df: pd.DataFrame):
        """Create RoadNode nodes from the CSV data."""
        with self.driver.session() as session:
            print(f"Creating {len(nodes_df)} road nodes...")
            
            # Create nodes in batches for better performance
            batch_size = 1000
            for i in range(0, len(nodes_df), batch_size):
                batch = nodes_df.iloc[i:i+batch_size]
                
                query = """
                UNWIND $batch AS node
                CREATE (n:RoadNode {
                    osmid: node.osmid,
                    location: point({latitude: toFloat(node.lat), longitude: toFloat(node.lon)}),
                    label: node.label
                })
                """
                
                batch_data = batch.to_dict('records')
                session.run(query, batch=batch_data)
                
                if (i + batch_size) % 5000 == 0:
                    print(f"Created {min(i + batch_size, len(nodes_df))} nodes...")
            
            print(f"Successfully created {len(nodes_df)} road nodes.")
    
    def create_road_relationships(self, segments_df: pd.DataFrame):
        """Create CONNECTS relationships between road nodes with geometry."""
        with self.driver.session() as session:
            print(f"Creating {len(segments_df)} road relationships...")
            
            # Create relationships in batches
            batch_size = 1000
            for i in range(0, len(segments_df), batch_size):
                batch = segments_df.iloc[i:i+batch_size]
                
                query = """
                UNWIND $batch AS segment
                MATCH (u:RoadNode {osmid: segment.u})
                MATCH (v:RoadNode {osmid: segment.v})
                CREATE (u)-[r:CONNECTS {
                    length_m: toFloat(segment.length_m),
                    type: segment.type,
                    geometry: segment.geometry
                }]->(v)
                """
                
                batch_data = batch.to_dict('records')
                session.run(query, batch=batch_data)
                
                if (i + batch_size) % 5000 == 0:
                    print(f"Created {min(i + batch_size, len(segments_df))} relationships...")
            
            print(f"Successfully created {len(segments_df)} road relationships.")
    
    def create_indexes(self):
        """Create indexes for better query performance."""
        with self.driver.session() as session:
            print("Creating indexes...")
            
            # Create index on osmid for faster lookups
            session.run("CREATE INDEX road_node_osmid IF NOT EXISTS FOR (n:RoadNode) ON (n.osmid)")
            
            # Create spatial index on location for spatial queries
            session.run("CREATE POINT INDEX road_node_location IF NOT EXISTS FOR (n:RoadNode) ON (n.location)")
            
            # Create index on relationship type
            session.run("CREATE INDEX connects_type IF NOT EXISTS FOR ()-[r:CONNECTS]-() ON (r.type)")
            
            # Create index on geometry for spatial queries on road segments
            session.run("CREATE INDEX connects_geometry IF NOT EXISTS FOR ()-[r:CONNECTS]-() ON (r.geometry)")
            
            print("Indexes created successfully.")
    
    def verify_data(self):
        """Verify the data was loaded correctly."""
        with self.driver.session() as session:
            print("\nVerifying data...")
            
            # Count nodes
            result = session.run("MATCH (n:RoadNode) RETURN count(n) as node_count")
            node_count = result.single()["node_count"]
            print(f"Total road nodes: {node_count}")
            
            # Count relationships
            result = session.run("MATCH ()-[r:CONNECTS]->() RETURN count(r) as rel_count")
            rel_count = result.single()["rel_count"]
            print(f"Total road relationships: {rel_count}")
            
            # Sample some nodes
            result = session.run("""
                MATCH (n:RoadNode)
                RETURN n.osmid, n.location, n.label
                LIMIT 5
            """)
            print("\nSample road nodes:")
            for record in result:
                print(f"  OSM ID: {record['n.osmid']}, Location: {record['n.location']}")
            
            # Sample some relationships
            result = session.run("""
                MATCH (u:RoadNode)-[r:CONNECTS]->(v:RoadNode)
                RETURN u.osmid, v.osmid, r.length_m, r.type, r.geometry
                LIMIT 5
            """)
            print("\nSample road relationships:")
            for record in result:
                has_geometry = record['r.geometry'] is not None
                geom_info = f" (with geometry: {has_geometry})" if has_geometry else " (no geometry)"
                print(f"  {record['u.osmid']} -> {record['v.osmid']} ({record['r.type']}, {record['r.length_m']:.2f}m){geom_info}")
            
            # Count relationships with geometry
            result = session.run("""
                MATCH ()-[r:CONNECTS]->()
                RETURN count(r) as total, count(r.geometry) as with_geometry
            """)
            counts = result.single()
            print(f"\nGeometry statistics:")
            print(f"  Total relationships: {counts['total']}")
            print(f"  With geometry: {counts['with_geometry']}")
            print(f"  Geometry coverage: {(counts['with_geometry']/counts['total']*100):.1f}%")
            
            # Test spatial query
            print("\nTesting spatial query (nodes near Kraków center):")
            result = session.run("""
                MATCH (n:RoadNode)
                WHERE point.distance(n.location, point({latitude: 50.0647, longitude: 19.9450})) < 1000
                RETURN n.osmid, n.location
                ORDER BY point.distance(n.location, point({latitude: 50.0647, longitude: 19.9450}))
                LIMIT 3
            """)
            for record in result:
                print(f"  OSM ID: {record['n.osmid']}, Location: {record['n.location']}")

def main():
    
    print("Loading road network data into Neo4j...")
    print("=" * 50)
    
    # Load CSV files
    print("Loading CSV files...")
    nodes_df = pd.read_csv(NODES_FILE)
    segments_df = pd.read_csv(SEGMENTS_FILE)
    
    print(f"Loaded {len(nodes_df)} nodes and {len(segments_df)} segments")
    
    # Initialize populator
    populator = RoadNetworkPopulator(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Clear existing data
        populator.clear_database()
        
        # Create nodes
        populator.create_road_nodes(nodes_df)
        
        # Create relationships
        populator.create_road_relationships(segments_df)
        
        # Create indexes
        populator.create_indexes()
        
        # Verify data
        populator.verify_data()
        
        print("\n" + "=" * 50)
        print("Road network data successfully loaded into Neo4j!")
        
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        populator.close()

if __name__ == "__main__":
    main()
