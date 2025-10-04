#!/usr/bin/env python3
"""
Professional Traffic Incident and Public Transport Integration System

This module provides a comprehensive solution for integrating traffic incident data
with public transport networks in Neo4j. It handles data loading, relationship
creation, and provides analysis capabilities.

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
        logging.FileHandler('transport_integration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Performance monitoring and timing utilities"""
    
    def __init__(self):
        self.start_times = {}
        self.total_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
        logger.info(f"⏱️  Starting {operation}...")
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration"""
        if operation not in self.start_times:
            logger.warning(f"No start time found for operation: {operation}")
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        self.total_times[operation] = duration
        
        # Format duration nicely
        if duration < 60:
            time_str = f"{duration:.1f}s"
        elif duration < 3600:
            time_str = f"{duration/60:.1f}m"
        else:
            time_str = f"{duration/3600:.1f}h"
        
        logger.info(f"✅ {operation} completed in {time_str}")
        return duration
    
    def get_summary(self) -> Dict[str, float]:
        """Get timing summary for all operations"""
        return self.total_times.copy()
    
    def estimate_remaining_time(self, current_progress: float, operation: str) -> str:
        """Estimate remaining time based on current progress"""
        if operation not in self.start_times:
            return "Unknown"
        
        elapsed = time.time() - self.start_times[operation]
        if current_progress <= 0:
            return "Unknown"
        
        total_estimated = elapsed / current_progress
        remaining = total_estimated - elapsed
        
        if remaining < 60:
            return f"{remaining:.0f}s"
        elif remaining < 3600:
            return f"{remaining/60:.1f}m"
        else:
            return f"{remaining/3600:.1f}h"

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
    base_dir: Path = Path(__file__).parent.parent.parent
    gtfs_dir: Path = base_dir / 'map' / 'GTFS_KRK_A'
    incidents_file: Path = base_dir / 'map' / 'krakow_incidents_processed.csv'
    
    # GTFS file paths
    stops_file: Path = gtfs_dir / 'stops.txt'
    routes_file: Path = gtfs_dir / 'routes.txt'
    trips_file: Path = gtfs_dir / 'trips.txt'
    stop_times_file: Path = gtfs_dir / 'stop_times.txt'

@dataclass
class ProcessingConfig:
    """Processing configuration settings"""
    batch_size: int = 5000  # Increased for better performance
    max_distance_stops: int = 200  # meters
    max_distance_paths: int = 100  # meters
    spatial_index_enabled: bool = True
    progress_report_interval: int = 5000  # More frequent updates
    use_parallel_processing: bool = True  # Enable parallel processing
    memory_limit: int = 2000  # MB for APOC operations
    apoc_batch_size: int = 1000  # APOC specific batch size

class Neo4jManager:
    """Professional Neo4j database manager with connection pooling and error handling"""
    
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
    """Professional data loader with validation and error handling"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self._validate_files()
    
    def _validate_files(self):
        """Validate that all required data files exist"""
        required_files = [
            self.config.stops_file,
            self.config.routes_file,
            self.config.trips_file,
            self.config.stop_times_file,
            self.config.incidents_file
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {missing_files}")
        
        logger.info("All required data files validated")
    
    def load_csv_data(self, file_path: Path, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Load CSV data with proper encoding and error handling"""
        logger.info(f"Loading data from {file_path}")
        
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if limit and i >= limit:
                        break
                    data.append(row)
            
            logger.info(f"Successfully loaded {len(data)} records from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            raise

class TransportGraphBuilder:
    """Professional transport graph builder"""
    
    def __init__(self, db_manager: Neo4jManager, data_loader: DataLoader, config: ProcessingConfig):
        self.db = db_manager
        self.loader = data_loader
        self.config = config
    
    def create_constraints(self):
        """Create database constraints for performance"""
        logger.info("Creating database constraints...")
        
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Stop) REQUIRE s.stop_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Trip) REQUIRE t.trip_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Route) REQUIRE r.route_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Incident) REQUIRE i.incident_id IS UNIQUE"
        ]
        
        for constraint in constraints:
            try:
                self.db.execute_query(constraint)
            except Exception as e:
                logger.warning(f"Constraint creation warning: {e}")
        
        logger.info("Database constraints created")
    
    def create_spatial_indexes(self):
        """Create spatial indexes for optimal performance"""
        if not self.config.spatial_index_enabled:
            return
        
        logger.info("Creating spatial indexes...")
        
        # Create comprehensive spatial indexes
        indexes = [
            "CREATE INDEX stop_location_index IF NOT EXISTS FOR (s:Stop) ON (s.location)",
            "CREATE INDEX incident_location_index IF NOT EXISTS FOR (i:Incident) ON (i.location)",
            "CREATE INDEX stop_id_index IF NOT EXISTS FOR (s:Stop) ON (s.stop_id)",
            "CREATE INDEX trip_id_index IF NOT EXISTS FOR (t:Trip) ON (t.trip_id)",
            "CREATE INDEX route_id_index IF NOT EXISTS FOR (r:Route) ON (r.route_id)",
            "CREATE INDEX incident_id_index IF NOT EXISTS FOR (i:Incident) ON (i.incident_id)"
        ]
        
        for index in indexes:
            try:
                self.db.execute_query(index)
                logger.info(f"Created index: {index.split('FOR')[1].split('ON')[0].strip()}")
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")
        
        # Create composite indexes for better performance
        composite_indexes = [
            "CREATE INDEX stop_times_sequence_index IF NOT EXISTS FOR ()-[r:STOPS_AT]-() ON (r.sequence)",
            "CREATE INDEX next_stop_trip_count_index IF NOT EXISTS FOR ()-[r:NEXT_STOP]-() ON (r.trip_count)"
        ]
        
        for index in composite_indexes:
            try:
                self.db.execute_query(index)
                logger.info(f"Created composite index: {index.split('ON')[1].strip()}")
            except Exception as e:
                logger.warning(f"Composite index creation warning: {e}")
        
        logger.info("All spatial and composite indexes created")
    
    def load_stops(self) -> int:
        """Load stop nodes from GTFS data"""
        logger.info("Loading stop nodes...")
        
        stops_data = self.loader.load_csv_data(self.loader.config.stops_file)
        
        query = """
        UNWIND $batch AS row
        MERGE (s:Stop {stop_id: row.stop_id})
        SET s.name = row.stop_name,
            s.location = point({latitude: toFloat(row.stop_lat), longitude: toFloat(row.stop_lon)}),
            s.code = row.stop_code,
            s.description = row.stop_desc
        """
        
        total_loaded = 0
        for i in range(0, len(stops_data), self.config.batch_size):
            batch = stops_data[i:i + self.config.batch_size]
            self.db.execute_write_query(query, {'batch': batch})
            total_loaded += len(batch)
            
            if total_loaded % self.config.progress_report_interval == 0:
                logger.info(f"Loaded {total_loaded} / {len(stops_data)} stops")
        
        logger.info(f"Successfully loaded {total_loaded} stop nodes")
        return total_loaded
    
    def load_routes_and_trips(self) -> Tuple[int, int]:
        """Load route and trip nodes"""
        logger.info("Loading route and trip nodes...")
        
        # Load routes
        routes_data = self.loader.load_csv_data(self.loader.config.routes_file)
        route_query = """
        UNWIND $batch AS row
        MERGE (r:Route {route_id: row.route_id})
        SET r.short_name = row.route_short_name,
            r.long_name = row.route_long_name,
            r.type = toInteger(row.route_type),
            r.color = row.route_color
        """
        
        self.db.execute_write_query(route_query, {'batch': routes_data})
        logger.info(f"Loaded {len(routes_data)} route nodes")
        
        # Load trips
        trips_data = self.loader.load_csv_data(self.loader.config.trips_file)
        trip_query = """
        UNWIND $batch AS row
        MATCH (r:Route {route_id: row.route_id})
        MERGE (t:Trip {trip_id: row.trip_id})
        SET t.headsign = row.trip_headsign,
            t.direction = toInteger(row.direction_id),
            t.service_id = row.service_id
        MERGE (t)-[:RUNS_ON]->(r)
        """
        
        self.db.execute_write_query(trip_query, {'batch': trips_data})
        logger.info(f"Loaded {len(trips_data)} trip nodes and connected them to routes")
        
        return len(routes_data), len(trips_data)
    
    def load_stop_times(self) -> int:
        """Load stop_times relationships"""
        logger.info("Loading stop_times relationships...")
        
        stop_times_data = self.loader.load_csv_data(self.loader.config.stop_times_file)
        
        query = """
        UNWIND $batch AS row
        MATCH (t:Trip {trip_id: row.trip_id})
        MATCH (s:Stop {stop_id: row.stop_id})
        MERGE (t)-[:STOPS_AT {
            arrival_time: row.arrival_time,
            departure_time: row.departure_time,
            sequence: toInteger(row.stop_sequence),
            pickup_type: toInteger(row.pickup_type),
            drop_off_type: toInteger(row.drop_off_type)
        }]->(s)
        """
        
        total_loaded = 0
        for i in range(0, len(stop_times_data), self.config.batch_size):
            batch = stop_times_data[i:i + self.config.batch_size]
            self.db.execute_write_query(query, {'batch': batch})
            total_loaded += len(batch)
            
            if total_loaded % self.config.progress_report_interval == 0:
                logger.info(f"Loaded {total_loaded} / {len(stop_times_data)} stop_times relationships")
        
        logger.info(f"Successfully loaded {total_loaded} stop_times relationships")
        return total_loaded
    
    def create_next_stop_relationships(self) -> int:
        """Create NEXT_STOP relationships between consecutive stops"""
        logger.info("Creating NEXT_STOP relationships...")
        
        query = """
        MATCH (t:Trip)
        MATCH (t)-[st:STOPS_AT]->(s:Stop)
        WITH t, s, st
        ORDER BY st.sequence
        WITH t, collect(s) AS stops
        UNWIND range(0, size(stops) - 2) AS i
        WITH t, stops[i] AS currentStop, stops[i+1] AS nextStop
        MERGE (currentStop)-[n:NEXT_STOP]->(nextStop)
        ON CREATE SET n.trip_count = 1
        ON MATCH SET n.trip_count = coalesce(n.trip_count, 0) + 1
        """
        
        self.db.execute_write_query(query)
        logger.info("NEXT_STOP relationships created")
        
        # Count relationships with proper session management
        count_query = "MATCH (s1:Stop)-[r:NEXT_STOP]->(s2:Stop) RETURN count(r) as count"
        with self.db.driver.session() as session:
            result = session.run(count_query)
            count = result.single()['count']
            logger.info(f"Created {count} NEXT_STOP relationships")
            return count

class IncidentIntegrator:
    """Professional incident integration system"""
    
    def __init__(self, db_manager: Neo4jManager, data_loader: DataLoader, config: ProcessingConfig):
        self.db = db_manager
        self.loader = data_loader
        self.config = config
    
    def load_incidents(self, limit: Optional[int] = None) -> int:
        """Load incident nodes from CSV data"""
        logger.info("Loading incident nodes...")
        
        incidents_data = self.loader.load_csv_data(self.loader.config.incidents_file, limit)
        
        query = """
        UNWIND $batch AS row
        MERGE (i:Incident {incident_id: row.incident_date_only + '_' + row.incident_time + '_' + toString(row.lat) + '_' + toString(row.lon)})
        SET i.date = date(row.incident_date_only),
            i.time = row.incident_time,
            i.location = point({latitude: toFloat(row.lat), longitude: toFloat(row.lon)}),
            i.speed_limit = toInteger(row.speed_limit),
            i.vehicles_count = toInteger(row.vehicles_count),
            i.participants_count = toInteger(row.participants_count),
            i.road_type_code = row.road_type_code,
            i.road_condition_code = row.road_condition_code,
            i.public_road = row.public_road,
            i.lanes_count = row.lanes_count
        """
        
        total_loaded = 0
        for i in range(0, len(incidents_data), self.config.batch_size):
            batch = incidents_data[i:i + self.config.batch_size]
            self.db.execute_write_query(query, {'batch': batch})
            total_loaded += len(batch)
            
            if total_loaded % self.config.progress_report_interval == 0:
                logger.info(f"Loaded {total_loaded} / {len(incidents_data)} incidents")
        
        logger.info(f"Successfully loaded {total_loaded} incident nodes")
        return total_loaded
    
    def create_near_stop_relationships(self) -> int:
        """Create NEAR_STOP relationships between incidents and stops - OPTIMIZED VERSION"""
        logger.info(f"Creating NEAR_STOP relationships (max distance: {self.config.max_distance_stops}m)...")
        
        # Simple APOC approach - test if APOC works first
        test_query = "CALL apoc.help('periodic') YIELD name RETURN count(name) as apoc_count"
        
        try:
            with self.db.driver.session() as session:
                # Test APOC availability
                result = session.run(test_query)
                apoc_count = result.single()['apoc_count']
                logger.info(f"APOC procedures available: {apoc_count}")
                
                if apoc_count > 0:
                    # Use APOC with hardcoded parameters to avoid syntax issues
                    apoc_query = f"""
                    CALL apoc.periodic.iterate(
                        "MATCH (i:Incident) RETURN i",
                        "MATCH (s:Stop)
                         WHERE point.distance(i.location, s.location) <= {self.config.max_distance_stops}
                         MERGE (i)-[r:NEAR_STOP {{
                             distance_meters: point.distance(i.location, s.location)
                         }}]->(s)",
                        {{
                            batchSize: {self.config.apoc_batch_size}, 
                            iterateList: true, 
                            parallel: {str(self.config.use_parallel_processing).lower()},
                            concurrency: 4,
                            retries: 3
                        }}
                    )
                    YIELD batches, total, timeTaken, committedOperations, failedOperations, errorMessages
                    RETURN batches, total, timeTaken, committedOperations, failedOperations, errorMessages
                    """
                    
                    result = session.run(apoc_query)
                    record = result.single()
                    if record:
                        logger.info(f"APOC NEAR_STOP creation completed:")
                        logger.info(f"  Batches: {record['batches']}, Total: {record['total']}")
                        logger.info(f"  Time taken: {record['timeTaken']}ms")
                        logger.info(f"  Committed: {record['committedOperations']}, Failed: {record['failedOperations']}")
                        if record['errorMessages']:
                            logger.warning(f"  Errors: {record['errorMessages']}")
                else:
                    raise Exception("APOC procedures not found")
                    
        except Exception as e:
            logger.warning(f"APOC not working properly, using optimized fallback: {e}")
            # Optimized fallback with batching
            self._create_near_stop_relationships_fallback()
        
        # Count relationships
        count_query = "MATCH (i:Incident)-[r:NEAR_STOP]->(s:Stop) RETURN count(r) as count"
        try:
            with self.db.driver.session() as session:
                result = session.run(count_query)
                count = result.single()['count']
                logger.info(f"Total NEAR_STOP relationships: {count}")
                return count
        except Exception as e:
            logger.error(f"Failed to count NEAR_STOP relationships: {e}")
            return 0
    
    def _create_near_stop_relationships_fallback(self):
        """Ultra-optimized fallback for NEAR_STOP relationships without APOC"""
        logger.info("Using ultra-optimized fallback for NEAR_STOP relationships...")
        
        # Get counts for progress tracking
        with self.db.driver.session() as session:
            incident_count = session.run("MATCH (i:Incident) RETURN count(i) as total").single()['total']
            stop_count = session.run("MATCH (s:Stop) RETURN count(s) as total").single()['total']
        
        logger.info(f"Processing {incident_count} incidents against {stop_count} stops...")
        
        # Use larger batches for better performance
        batch_size = 500  # Increased batch size
        processed = 0
        start_time = time.time()
        
        for offset in range(0, incident_count, batch_size):
            # Optimized query with better performance
            query = """
            MATCH (i:Incident)
            WITH i SKIP $offset LIMIT $batch_size
            MATCH (s:Stop)
            WHERE point.distance(i.location, s.location) <= $max_distance
            MERGE (i)-[r:NEAR_STOP {
                distance_meters: point.distance(i.location, s.location)
            }]->(s)
            RETURN count(r) as relationships_created
            """
            
            with self.db.driver.session() as session:
                result = session.run(query, {
                    'offset': offset,
                    'batch_size': batch_size,
                    'max_distance': self.config.max_distance_stops
                })
                relationships_created = result.single()['relationships_created']
                processed += batch_size
                
                # Calculate progress and ETA
                progress = min(processed, incident_count) / incident_count
                elapsed = time.time() - start_time
                if progress > 0:
                    eta = (elapsed / progress) - elapsed
                    eta_str = f"{eta/60:.1f}m" if eta > 60 else f"{eta:.0f}s"
                else:
                    eta_str = "Unknown"
                
                logger.info(f"Processed {min(processed, incident_count)}/{incident_count} incidents "
                          f"({progress*100:.1f}%) - ETA: {eta_str} - Created {relationships_created} relationships")
    
    def create_affects_route_relationships(self) -> int:
        """Create AFFECTS_ROUTE relationships between incidents and routes - OPTIMIZED"""
        logger.info("Creating AFFECTS_ROUTE relationships...")
        
        try:
            with self.db.driver.session() as session:
                # Test APOC availability first
                test_result = session.run("CALL apoc.help('periodic') YIELD name RETURN count(name) as apoc_count")
                apoc_count = test_result.single()['apoc_count']
                
                if apoc_count > 0:
                    # Use APOC with hardcoded parameters
                    apoc_query = f"""
                    CALL apoc.periodic.iterate(
                        "MATCH (i:Incident)-[r:NEAR_STOP]->(s:Stop) RETURN i, r, s",
                        "MATCH (t:Trip)-[:STOPS_AT]->(s)
                         MATCH (t)-[:RUNS_ON]->(route:Route)
                         WITH i, route, min(r.distance_meters) as min_distance, count(DISTINCT s) as affected_stops
                         MERGE (i)-[ar:AFFECTS_ROUTE {{
                             min_distance_to_route: min_distance,
                             affected_stops_count: affected_stops
                         }}]->(route)",
                        {{
                            batchSize: {self.config.apoc_batch_size},
                            iterateList: true,
                            parallel: {str(self.config.use_parallel_processing).lower()},
                            concurrency: 2,
                            retries: 3
                        }}
                    )
                    YIELD batches, total, timeTaken, committedOperations, failedOperations, errorMessages
                    RETURN batches, total, timeTaken, committedOperations, failedOperations, errorMessages
                    """
                    
                    result = session.run(apoc_query)
                    record = result.single()
                    if record:
                        logger.info(f"APOC AFFECTS_ROUTE creation completed:")
                        logger.info(f"  Batches: {record['batches']}, Total: {record['total']}")
                        logger.info(f"  Time taken: {record['timeTaken']}ms")
                        logger.info(f"  Committed: {record['committedOperations']}, Failed: {record['failedOperations']}")
                else:
                    raise Exception("APOC procedures not found")
                    
        except Exception as e:
            logger.warning(f"APOC not working properly, using standard query: {e}")
            # Standard optimized query
            standard_query = """
            MATCH (i:Incident)-[r:NEAR_STOP]->(s:Stop)
            MATCH (t:Trip)-[:STOPS_AT]->(s)
            MATCH (t)-[:RUNS_ON]->(route:Route)
            WITH i, route, min(r.distance_meters) as min_distance, count(DISTINCT s) as affected_stops
            MERGE (i)-[ar:AFFECTS_ROUTE {
                min_distance_to_route: min_distance,
                affected_stops_count: affected_stops
            }]->(route)
            """
            with self.db.driver.session() as session:
                result = session.run(standard_query)
                summary = result.consume()
                logger.info(f"AFFECTS_ROUTE relationships created: {summary.counters.relationships_created}")
        
        # Count relationships
        count_query = "MATCH (i:Incident)-[r:AFFECTS_ROUTE]->(route:Route) RETURN count(r) as count"
        try:
            with self.db.driver.session() as session:
                result = session.run(count_query)
                count = result.single()['count']
                logger.info(f"Total AFFECTS_ROUTE relationships: {count}")
                return count
        except Exception as e:
            logger.error(f"Failed to count AFFECTS_ROUTE relationships: {e}")
            return 0
    
    def create_on_transport_path_relationships(self) -> int:
        """Create ON_TRANSPORT_PATH relationships for incidents on actual transport paths - OPTIMIZED"""
        logger.info(f"Creating ON_TRANSPORT_PATH relationships (max distance: {self.config.max_distance_paths}m)...")
        
        try:
            with self.db.driver.session() as session:
                # Test APOC availability first
                test_result = session.run("CALL apoc.help('periodic') YIELD name RETURN count(name) as apoc_count")
                apoc_count = test_result.single()['apoc_count']
                
                if apoc_count > 0:
                    # Use APOC with hardcoded parameters
                    apoc_query = f"""
                    CALL apoc.periodic.iterate(
                        "MATCH (s1:Stop)-[:NEXT_STOP]->(s2:Stop) RETURN s1, s2",
                        "MATCH (i:Incident)
                         WHERE point.distance(i.location, s1.location) <= {self.config.max_distance_paths} 
                            OR point.distance(i.location, s2.location) <= {self.config.max_distance_paths}
                            OR (
                                point.distance(i.location, s1.location) + point.distance(i.location, s2.location) 
                                <= point.distance(s1.location, s2.location) * 1.2
                            )
                         MATCH (t:Trip)-[:STOPS_AT]->(s1)
                         MATCH (t)-[:RUNS_ON]->(route:Route)
                         WITH i, route, min(point.distance(i.location, s1.location)) as min_distance
                         MERGE (i)-[r:ON_TRANSPORT_PATH {{
                             distance_to_path: min_distance,
                             path_type: 'route_segment'
                         }}]->(route)",
                        {{
                            batchSize: 50,
                            iterateList: true,
                            parallel: {str(self.config.use_parallel_processing).lower()},
                            concurrency: 2,
                            retries: 3
                        }}
                    )
                    YIELD batches, total, timeTaken, committedOperations, failedOperations, errorMessages
                    RETURN batches, total, timeTaken, committedOperations, failedOperations, errorMessages
                    """
                    
                    result = session.run(apoc_query)
                    record = result.single()
                    if record:
                        logger.info(f"APOC ON_TRANSPORT_PATH creation completed:")
                        logger.info(f"  Batches: {record['batches']}, Total: {record['total']}")
                        logger.info(f"  Time taken: {record['timeTaken']}ms")
                        logger.info(f"  Committed: {record['committedOperations']}, Failed: {record['failedOperations']}")
                else:
                    raise Exception("APOC procedures not found")
                    
        except Exception as e:
            logger.warning(f"APOC not working properly, using optimized fallback: {e}")
            # Optimized fallback with batching
            self._create_on_transport_path_fallback()
        
        # Count relationships
        count_query = "MATCH (i:Incident)-[r:ON_TRANSPORT_PATH]->(route:Route) RETURN count(r) as count"
        try:
            with self.db.driver.session() as session:
                result = session.run(count_query)
                count = result.single()['count']
                logger.info(f"Total ON_TRANSPORT_PATH relationships: {count}")
                return count
        except Exception as e:
            logger.error(f"Failed to count ON_TRANSPORT_PATH relationships: {e}")
            return 0
    
    def _create_on_transport_path_fallback(self):
        """Ultra-optimized fallback for ON_TRANSPORT_PATH relationships without APOC"""
        logger.info("Using ultra-optimized fallback for ON_TRANSPORT_PATH relationships...")
        
        # Get counts for progress tracking
        with self.db.driver.session() as session:
            stop_pairs_count = session.run("MATCH (s1:Stop)-[:NEXT_STOP]->(s2:Stop) RETURN count(*) as total").single()['total']
            incident_count = session.run("MATCH (i:Incident) RETURN count(i) as total").single()['total']
        
        logger.info(f"Processing {stop_pairs_count} stop pairs against {incident_count} incidents...")
        
        # Use larger batches for better performance
        batch_size = 100  # Increased batch size
        processed = 0
        start_time = time.time()
        
        for offset in range(0, stop_pairs_count, batch_size):
            # Optimized query with better performance
            query = """
            MATCH (s1:Stop)-[:NEXT_STOP]->(s2:Stop)
            WITH s1, s2 SKIP $offset LIMIT $batch_size
            MATCH (i:Incident)
            WHERE point.distance(i.location, s1.location) <= $max_distance 
               OR point.distance(i.location, s2.location) <= $max_distance
               OR (
                   point.distance(i.location, s1.location) + point.distance(i.location, s2.location) 
                   <= point.distance(s1.location, s2.location) * 1.2
               )
            MATCH (t:Trip)-[:STOPS_AT]->(s1)
            MATCH (t)-[:RUNS_ON]->(route:Route)
            WITH i, route, min(point.distance(i.location, s1.location)) as min_distance
            MERGE (i)-[r:ON_TRANSPORT_PATH {
                distance_to_path: min_distance,
                path_type: 'route_segment'
            }]->(route)
            RETURN count(r) as relationships_created
            """
            
            with self.db.driver.session() as session:
                result = session.run(query, {
                    'offset': offset,
                    'batch_size': batch_size,
                    'max_distance': self.config.max_distance_paths
                })
                relationships_created = result.single()['relationships_created']
                processed += batch_size
                
                # Calculate progress and ETA
                progress = min(processed, stop_pairs_count) / stop_pairs_count
                elapsed = time.time() - start_time
                if progress > 0:
                    eta = (elapsed / progress) - elapsed
                    eta_str = f"{eta/60:.1f}m" if eta > 60 else f"{eta:.0f}s"
                else:
                    eta_str = "Unknown"
                
                logger.info(f"Processed {min(processed, stop_pairs_count)}/{stop_pairs_count} stop pairs "
                          f"({progress*100:.1f}%) - ETA: {eta_str} - Created {relationships_created} relationships")

class AnalyticsEngine:
    """Professional analytics engine for the integrated data"""
    
    def __init__(self, db_manager: Neo4jManager):
        self.db = db_manager
    
    def get_statistics(self) -> Dict[str, int]:
        """Get comprehensive statistics about the integrated data"""
        logger.info("Generating comprehensive statistics...")
        
        stats_queries = {
            'total_incidents': "MATCH (i:Incident) RETURN count(i) as count",
            'incidents_near_stops': "MATCH (i:Incident)-[:NEAR_STOP]->(s:Stop) RETURN count(DISTINCT i) as count",
            'incidents_affecting_routes': "MATCH (i:Incident)-[:AFFECTS_ROUTE]->(r:Route) RETURN count(DISTINCT i) as count",
            'incidents_on_paths': "MATCH (i:Incident)-[:ON_TRANSPORT_PATH]->(r:Route) RETURN count(DISTINCT i) as count",
            'routes_affected': "MATCH (i:Incident)-[:AFFECTS_ROUTE]->(r:Route) RETURN count(DISTINCT r) as count",
            'stops_near_incidents': "MATCH (i:Incident)-[:NEAR_STOP]->(s:Stop) RETURN count(DISTINCT s) as count",
            'total_stops': "MATCH (s:Stop) RETURN count(s) as count",
            'total_routes': "MATCH (r:Route) RETURN count(r) as count",
            'total_trips': "MATCH (t:Trip) RETURN count(t) as count",
            'stops_at_relationships': "MATCH (t:Trip)-[:STOPS_AT]->(s:Stop) RETURN count(*) as count",
            'next_stop_relationships': "MATCH (s1:Stop)-[:NEXT_STOP]->(s2:Stop) RETURN count(*) as count"
        }
        
        statistics = {}
        with self.db.driver.session() as session:
            for stat_name, query in stats_queries.items():
                try:
                    result = session.run(query)
                    count = result.single()['count']
                    statistics[stat_name] = count
                except Exception as e:
                    logger.warning(f"Failed to get statistic {stat_name}: {e}")
                    statistics[stat_name] = 0
        
        return statistics
    
    def get_high_risk_routes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get routes with the most incidents"""
        query = """
        MATCH (r:Route)<-[:AFFECTS_ROUTE]-(i:Incident)
        RETURN r.route_short_name as route_name, 
               r.route_long_name as route_description,
               count(i) as incident_count
        ORDER BY incident_count DESC
        LIMIT $limit
        """
        
        with self.db.driver.session() as session:
            result = session.run(query, {'limit': limit})
            records = [dict(record) for record in result]
            return records
    
    def get_incidents_by_time_pattern(self) -> List[Dict[str, Any]]:
        """Analyze incident patterns by time of day"""
        query = """
        MATCH (i:Incident)
        WITH i, 
             datetime(i.date + duration('P' + substring(i.time, 0, 2) + 'H' + substring(i.time, 3, 2) + 'M')) as incident_datetime
        WITH i, incident_datetime,
             datetime(incident_datetime).hour as hour_of_day,
             datetime(incident_datetime).dayOfWeek as day_of_week
        RETURN hour_of_day, day_of_week, count(i) as incident_count
        ORDER BY day_of_week, hour_of_day
        """
        
        with self.db.driver.session() as session:
            result = session.run(query)
            records = [dict(record) for record in result]
            return records

class TransportIntegrationSystem:
    """Main system class that orchestrates the entire integration process"""
    
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
        self.incident_integrator = IncidentIntegrator(self.db_manager, self.data_loader, self.processing_config)
        self.analytics = AnalyticsEngine(self.db_manager)
        self.performance_monitor = PerformanceMonitor()
    
    def build_transport_network(self) -> Dict[str, int]:
        """Build the complete transport network with performance monitoring"""
        logger.info("=== BUILDING TRANSPORT NETWORK ===")
        
        # Create constraints and indexes
        self.performance_monitor.start_timer("Creating constraints and indexes")
        self.transport_builder.create_constraints()
        self.transport_builder.create_spatial_indexes()
        self.performance_monitor.end_timer("Creating constraints and indexes")
        
        # Load transport data with timing
        self.performance_monitor.start_timer("Loading stops")
        stops_count = self.transport_builder.load_stops()
        self.performance_monitor.end_timer("Loading stops")
        
        self.performance_monitor.start_timer("Loading routes and trips")
        routes_count, trips_count = self.transport_builder.load_routes_and_trips()
        self.performance_monitor.end_timer("Loading routes and trips")
        
        self.performance_monitor.start_timer("Loading stop times")
        stop_times_count = self.transport_builder.load_stop_times()
        self.performance_monitor.end_timer("Loading stop times")
        
        self.performance_monitor.start_timer("Creating next stop relationships")
        next_stop_count = self.transport_builder.create_next_stop_relationships()
        self.performance_monitor.end_timer("Creating next stop relationships")
        
        logger.info("=== TRANSPORT NETWORK BUILD COMPLETE ===")
        
        # Print performance summary
        summary = self.performance_monitor.get_summary()
        total_time = sum(summary.values())
        logger.info(f"🚀 Total transport network build time: {total_time/60:.1f} minutes")
        
        return {
            'stops': stops_count,
            'routes': routes_count,
            'trips': trips_count,
            'stop_times': stop_times_count,
            'next_stops': next_stop_count
        }
    
    def integrate_incidents(self, limit: Optional[int] = None) -> Dict[str, int]:
        """Integrate incident data with transport network with performance monitoring"""
        logger.info("=== INTEGRATING INCIDENT DATA ===")
        
        # Load incidents with timing
        self.performance_monitor.start_timer("Loading incidents")
        incidents_count = self.incident_integrator.load_incidents(limit)
        self.performance_monitor.end_timer("Loading incidents")
        
        # Create relationships with timing
        self.performance_monitor.start_timer("Creating near stop relationships")
        near_stop_count = self.incident_integrator.create_near_stop_relationships()
        self.performance_monitor.end_timer("Creating near stop relationships")
        
        self.performance_monitor.start_timer("Creating affects route relationships")
        affects_route_count = self.incident_integrator.create_affects_route_relationships()
        self.performance_monitor.end_timer("Creating affects route relationships")
        
        self.performance_monitor.start_timer("Creating on transport path relationships")
        on_path_count = self.incident_integrator.create_on_transport_path_relationships()
        self.performance_monitor.end_timer("Creating on transport path relationships")
        
        logger.info("=== INCIDENT INTEGRATION COMPLETE ===")
        
        # Print performance summary
        summary = self.performance_monitor.get_summary()
        total_time = sum(summary.values())
        logger.info(f"🚀 Total incident integration time: {total_time/60:.1f} minutes")
        
        return {
            'incidents': incidents_count,
            'near_stops': near_stop_count,
            'affects_routes': affects_route_count,
            'on_paths': on_path_count
        }
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report of the integrated system"""
        logger.info("=== GENERATING COMPREHENSIVE REPORT ===")
        
        statistics = self.analytics.get_statistics()
        high_risk_routes = self.analytics.get_high_risk_routes()
        time_patterns = self.analytics.get_incidents_by_time_pattern()
        
        return {
            'statistics': statistics,
            'high_risk_routes': high_risk_routes,
            'time_patterns': time_patterns,
            'generated_at': datetime.now().isoformat()
        }
    
    def close(self):
        """Clean up resources"""
        self.db_manager.close()

def main():
    """Main entry point with command line interface"""
    parser = argparse.ArgumentParser(
        description='Professional Traffic Incident and Public Transport Integration System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transport_integration.py --build-transport
  python transport_integration.py --integrate-incidents
  python transport_integration.py --full-integration
  python transport_integration.py --report-only
        """
    )
    
    parser.add_argument('--build-transport', action='store_true',
                       help='Build the transport network only')
    parser.add_argument('--integrate-incidents', action='store_true',
                       help='Integrate incidents with existing transport network')
    parser.add_argument('--full-integration', action='store_true',
                       help='Perform complete integration (transport + incidents)')
    parser.add_argument('--report-only', action='store_true',
                       help='Generate report only (assumes data already loaded)')
    parser.add_argument('--limit', type=int,
                       help='Limit number of incidents to load (for testing)')
    parser.add_argument('--stop-distance', type=int, default=200,
                       help='Maximum distance to connect incidents to stops (meters)')
    parser.add_argument('--path-distance', type=int, default=100,
                       help='Maximum distance to find incidents on transport paths (meters)')
    parser.add_argument('--batch-size', type=int, default=5000,
                       help='Batch size for data processing (default: 5000)')
    parser.add_argument('--apoc-batch-size', type=int, default=1000,
                       help='APOC batch size for parallel processing (default: 1000)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing (slower but more stable)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize system with performance optimizations
    processing_config = ProcessingConfig(
        batch_size=args.batch_size,
        max_distance_stops=args.stop_distance,
        max_distance_paths=args.path_distance,
        use_parallel_processing=not args.no_parallel,
        apoc_batch_size=args.apoc_batch_size
    )
    
    system = TransportIntegrationSystem(processing_config=processing_config)
    
    try:
        if args.build_transport:
            logger.info("Building transport network...")
            result = system.build_transport_network()
            logger.info(f"Transport network built: {result}")
            
        elif args.integrate_incidents:
            logger.info("Integrating incidents...")
            result = system.integrate_incidents(limit=args.limit)
            logger.info(f"Incidents integrated: {result}")
            
        elif args.full_integration:
            logger.info("Performing full integration...")
            transport_result = system.build_transport_network()
            incident_result = system.integrate_incidents(limit=args.limit)
            logger.info(f"Full integration complete: Transport={transport_result}, Incidents={incident_result}")
            
        elif args.report_only:
            logger.info("Generating report...")
            report = system.get_comprehensive_report()
            
            print("\n" + "="*80)
            print("COMPREHENSIVE INTEGRATION REPORT")
            print("="*80)
            
            print("\n STATISTICS:")
            for key, value in report['statistics'].items():
                print(f"  {key.replace('_', ' ').title()}: {value:,}")
            
            print("\nHIGH-RISK ROUTES:")
            for route in report['high_risk_routes'][:5]:
                print(f"  {route['route_name']}: {route['incident_count']} incidents")
            
            print(f"\nReport generated at: {report['generated_at']}")
            
        else:
            parser.print_help()
            
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)
    finally:
        system.close()

if __name__ == "__main__":
    main()
