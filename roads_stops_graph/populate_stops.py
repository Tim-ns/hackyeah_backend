#!/usr/bin/env python3
"""
Stop Population System for Neo4j - Bus and Tram Stops

This module provides a focused solution for populating Neo4j with public transport stops
from both bus (GTFS_KRK_A) and tram (GTFS_KRK_T) GTFS data. It includes data loading, 
validation, and database population capabilities.

Author: AI Assistant
Version: 1.0.0
"""

import csv
import os
import sys
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from neo4j import GraphDatabase
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('populate_stops.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    uri: str = 'neo4j://127.0.0.1:7687'
    user: str = 'neo4j'
    password: str = '12345678'
    max_retries: int = 3
    timeout: int = 30

@dataclass
class DataConfig:
    """Data file configuration"""
    base_dir: Path = Path(__file__).parent.parent
    gtfs_bus_dir: Path = base_dir / 'map' / 'GTFS_KRK_A'
    gtfs_tram_dir: Path = base_dir / 'map' / 'GTFS_KRK_T'
    
    # GTFS file paths
    bus_stops_file: Path = gtfs_bus_dir / 'stops.txt'
    tram_stops_file: Path = gtfs_tram_dir / 'stops.txt'

@dataclass
class ProcessingConfig:
    """Processing configuration settings"""
    batch_size: int = 1000
    spatial_index_enabled: bool = True
    progress_report_interval: int = 1000

class Neo4jManager:
    """Simplified Neo4j database manager with connection handling"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Establish database connection with retry logic"""
        for attempt in range(self.config.max_retries):
            try:
                self.driver = GraphDatabase.driver(
                    self.config.uri,
                    auth=(self.config.user, self.config.password),
                    connection_timeout=self.config.timeout
                )
                # Test connection
                with self.driver.session() as session:
                    session.run("RETURN 1")
                logger.info(f"Successfully connected to Neo4j database")
                return
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    raise ConnectionError(f"Failed to connect to Neo4j after {self.config.max_retries} attempts")
    
    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
            logger.info("Database connection closed")
    
    def execute_query(self, query: str, parameters: Optional[Dict] = None) -> Any:
        """Execute a Cypher query with proper error handling"""
        with self.driver.session() as session:
            try:
                result = session.run(query, parameters or {})
                return result
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                logger.error(f"Query: {query}")
                raise
    
    def execute_write_query(self, query: str, parameters: Optional[Dict] = None) -> int:
        """Execute a write query and return the number of affected records"""
        with self.driver.session() as session:
            try:
                result = session.run(query, parameters or {})
                summary = result.consume()
                return summary.counters.nodes_created + summary.counters.relationships_created
            except Exception as e:
                logger.error(f"Write query execution failed: {e}")
                logger.error(f"Query: {query}")
                raise

class DataLoader:
    """Simplified data loader focused on stops data"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self._validate_files()
    
    def _validate_files(self):
        """Validate that required data files exist"""
        required_files = [self.config.bus_stops_file, self.config.tram_stops_file]
        
        missing_files = [f for f in required_files if not f.exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {missing_files}")
        
        logger.info("All required data files validated")
    
    def load_stops_data(self, limit: Optional[int] = None) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Load stops data from both bus and tram GTFS files"""
        logger.info("Loading stops data from both bus and tram GTFS files")
        
        # Load bus stops
        bus_data = []
        try:
            with open(self.config.bus_stops_file, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if limit and i >= limit:
                        break
                    row['transport_type'] = 'bus'  # Add transport type
                    bus_data.append(row)
            
            logger.info(f"Successfully loaded {len(bus_data)} bus stop records")
        except Exception as e:
            logger.error(f"Failed to load bus stops data: {e}")
            raise
        
        # Load tram stops
        tram_data = []
        try:
            with open(self.config.tram_stops_file, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if limit and i >= limit:
                        break
                    row['transport_type'] = 'tram'  # Add transport type
                    tram_data.append(row)
            
            logger.info(f"Successfully loaded {len(tram_data)} tram stop records")
        except Exception as e:
            logger.error(f"Failed to load tram stops data: {e}")
            raise
        
        return bus_data, tram_data

class TransportGraphBuilder:
    """Simplified transport graph builder focused on stops"""
    
    def __init__(self, db_manager: Neo4jManager, data_loader: DataLoader, config: ProcessingConfig):
        self.db = db_manager
        self.loader = data_loader
        self.config = config
    
    def create_constraints(self):
        """Create database constraints for stops"""
        logger.info("Creating database constraints...")
        
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Stop) REQUIRE s.stop_id IS UNIQUE"
        ]
        
        for constraint in constraints:
            try:
                self.db.execute_query(constraint)
            except Exception as e:
                logger.warning(f"Constraint creation warning: {e}")
        
        logger.info("Database constraints created")
    
    def create_spatial_indexes(self):
        """Create spatial indexes for stops"""
        if not self.config.spatial_index_enabled:
            return
        
        logger.info("Creating spatial indexes...")
        
        indexes = [
            "CREATE INDEX stop_location_index IF NOT EXISTS FOR (s:Stop) ON (s.location)",
            "CREATE INDEX stop_id_index IF NOT EXISTS FOR (s:Stop) ON (s.stop_id)"
        ]
        
        for index in indexes:
            try:
                self.db.execute_query(index)
                logger.info(f"Created index: {index.split('FOR')[1].split('ON')[0].strip()}")
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")
        
        logger.info("Spatial indexes created")
    
    def load_stops(self, limit: Optional[int] = None) -> int:
        """Load stop nodes from both bus and tram GTFS data"""
        logger.info("Loading stop nodes from bus and tram data...")
        
        bus_data, tram_data = self.loader.load_stops_data(limit)
        
        query = """
        UNWIND $batch AS row
        MERGE (s:Stop {stop_id: row.stop_id})
        SET s.name = row.stop_name,
            s.location = point({latitude: toFloat(row.stop_lat), longitude: toFloat(row.stop_lon)}),
            s.code = row.stop_code,
            s.description = row.stop_desc,
            s.transport_type = row.transport_type
        """
        
        total_loaded = 0
        
        # Load bus stops
        logger.info(f"Loading {len(bus_data)} bus stops...")
        for i in range(0, len(bus_data), self.config.batch_size):
            batch = bus_data[i:i + self.config.batch_size]
            self.db.execute_write_query(query, {'batch': batch})
            total_loaded += len(batch)
            
            if total_loaded % self.config.progress_report_interval == 0:
                logger.info(f"Loaded {total_loaded} / {len(bus_data) + len(tram_data)} stops")
        
        # Load tram stops
        logger.info(f"Loading {len(tram_data)} tram stops...")
        for i in range(0, len(tram_data), self.config.batch_size):
            batch = tram_data[i:i + self.config.batch_size]
            self.db.execute_write_query(query, {'batch': batch})
            total_loaded += len(batch)
            
            if total_loaded % self.config.progress_report_interval == 0:
                logger.info(f"Loaded {total_loaded} / {len(bus_data) + len(tram_data)} stops")
        
        logger.info(f"Successfully loaded {total_loaded} stop nodes ({len(bus_data)} bus + {len(tram_data)} tram)")
        return total_loaded
    
    def verify_data(self):
        """Verify the stops data was loaded correctly"""
        with self.db.driver.session() as session:
            logger.info("Verifying stops data...")
            
            # Count total stops
            result = session.run("MATCH (s:Stop) RETURN count(s) as stop_count")
            stop_count = result.single()["stop_count"]
            logger.info(f"Total stops: {stop_count}")
            
            # Count by transport type
            result = session.run("""
                MATCH (s:Stop)
                RETURN s.transport_type as transport_type, count(s) as count
                ORDER BY transport_type
            """)
            logger.info("Stops by transport type:")
            for record in result:
                logger.info(f"  {record['transport_type']}: {record['count']}")
            
            # Sample some stops
            result = session.run("""
                MATCH (s:Stop)
                RETURN s.stop_id, s.name, s.transport_type, s.location
                LIMIT 5
            """)
            logger.info("Sample stops:")
            for record in result:
                logger.info(f"  ID: {record['s.stop_id']}, Name: {record['s.name']}, Type: {record['s.transport_type']}, Location: {record['s.location']}")
            
            # Test spatial query
            logger.info("Testing spatial query (stops near Kraków center):")
            result = session.run("""
                MATCH (s:Stop)
                WHERE point.distance(s.location, point({latitude: 50.0647, longitude: 19.9450})) < 2000
                RETURN s.stop_id, s.name, s.transport_type, s.location
                ORDER BY point.distance(s.location, point({latitude: 50.0647, longitude: 19.9450}))
                LIMIT 5
            """)
            for record in result:
                logger.info(f"  ID: {record['s.stop_id']}, Name: {record['s.name']}, Type: {record['s.transport_type']}, Location: {record['s.location']}")

class StopPopulationSystem:
    """Main system class for stop population"""
    
    def __init__(self, 
                 db_config: DatabaseConfig = None,
                 data_config: DataConfig = None,
                 processing_config: ProcessingConfig = None):
        
        self.db_config = db_config or DatabaseConfig()
        self.data_config = data_config or DataConfig()
        self.processing_config = processing_config or ProcessingConfig()
        
        # Initialize components
        self.db_manager = Neo4jManager(self.db_config)
        self.data_loader = DataLoader(self.data_config)
        self.transport_builder = TransportGraphBuilder(self.db_manager, self.data_loader, self.processing_config)
    
    def populate_stops(self, limit: Optional[int] = None) -> int:
        """Populate the database with stops"""
        logger.info("=== POPULATING STOPS ===")
        
        # Create constraints and indexes
        logger.info("Creating constraints and indexes...")
        self.transport_builder.create_constraints()
        self.transport_builder.create_spatial_indexes()
        
        # Load stops
        logger.info("Loading stops...")
        start_time = time.time()
        stops_count = self.transport_builder.load_stops(limit)
        load_time = time.time() - start_time
        
        # Verify data
        logger.info("Verifying data...")
        self.transport_builder.verify_data()
        
        logger.info("=== STOP POPULATION COMPLETE ===")
        logger.info(f"🚀 Loaded {stops_count} stops in {load_time:.1f} seconds")
        
        return stops_count
    
    def clear_stops(self):
        """Clear all stops from the database"""
        logger.info("Clearing existing stops...")
        query = "MATCH (s:Stop) DETACH DELETE s"
        self.db_manager.execute_query(query)
        logger.info("Stops cleared")
    
    def close(self):
        """Clean up resources"""
        self.db_manager.close()

def main():
    """Main entry point with command line interface"""
    parser = argparse.ArgumentParser(
        description='Stop Population System for Neo4j - Loads Bus and Tram Stops from GTFS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python populate_stops.py --populate
  python populate_stops.py --populate --limit 100
  python populate_stops.py --clear
  python populate_stops.py --populate --clear
        """
    )
    
    parser.add_argument('--populate', action='store_true',
                       help='Populate the database with stops')
    parser.add_argument('--clear', action='store_true',
                       help='Clear existing stops before populating')
    parser.add_argument('--limit', type=int,
                       help='Limit number of stops to load (for testing)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for data processing (default: 1000)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize system
    processing_config = ProcessingConfig(batch_size=args.batch_size)
    system = StopPopulationSystem(processing_config=processing_config)
    
    try:
        if args.clear:
            logger.info("Clearing existing stops...")
            system.clear_stops()
        
        if args.populate:
            logger.info("Populating stops...")
            result = system.populate_stops(limit=args.limit)
            logger.info(f"Successfully populated {result} stops")
        
        if not args.populate and not args.clear:
            parser.print_help()
            
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)
    finally:
        system.close()

if __name__ == "__main__":
    main()