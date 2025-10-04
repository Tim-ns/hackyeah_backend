#!/usr/bin/env python3
"""
Test script for the refactored transport integration system
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from transport_integration import (
    TransportIntegrationSystem,
    DatabaseConfig,
    DataConfig,
    ProcessingConfig
)

def test_system_initialization():
    """Test that the system can be initialized properly"""
    print("Testing system initialization...")
    
    try:
        # Test with default configuration
        system = TransportIntegrationSystem()
        print("✅ System initialized successfully with default config")
        
        # Test database connection
        system.db_manager._connect()
        print("✅ Database connection successful")
        
        # Test data file validation
        system.data_loader._validate_files()
        print("✅ Data files validated successfully")
        
        system.close()
        return True
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False

def test_configuration():
    """Test configuration system"""
    print("\nTesting configuration system...")
    
    try:
        # Test custom database config
        db_config = DatabaseConfig(
            uri='neo4j://127.0.0.1:7687',
            user='neo4j',
            password='12345678'
        )
        print("✅ Database configuration created")
        
        # Test custom processing config
        processing_config = ProcessingConfig(
            batch_size=500,
            max_distance_stops=150,
            max_distance_paths=75
        )
        print("✅ Processing configuration created")
        
        # Test custom data config
        data_config = DataConfig()
        print("✅ Data configuration created")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_data_loading():
    """Test data loading capabilities"""
    print("\nTesting data loading...")
    
    try:
        system = TransportIntegrationSystem()
        
        # Test CSV data loading (small sample)
        stops_data = system.data_loader.load_csv_data(
            system.data_loader.config.stops_file, 
            limit=5
        )
        print(f"✅ Loaded {len(stops_data)} stop records")
        
        # Test incidents data loading (small sample)
        incidents_data = system.data_loader.load_csv_data(
            system.data_loader.config.incidents_file,
            limit=5
        )
        print(f"✅ Loaded {len(incidents_data)} incident records")
        
        system.close()
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def test_database_operations():
    """Test basic database operations"""
    print("\nTesting database operations...")
    
    try:
        system = TransportIntegrationSystem()
        
        # Test query execution
        result = system.db_manager.execute_query("RETURN 1 as test")
        test_value = result.single()['test']
        print(f"✅ Database query executed successfully (result: {test_value})")
        
        # Test constraint creation
        system.transport_builder.create_constraints()
        print("✅ Database constraints created")
        
        # Test spatial indexes
        system.transport_builder.create_spatial_indexes()
        print("✅ Spatial indexes created")
        
        system.close()
        return True
        
    except Exception as e:
        print(f"❌ Database operations test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("TRANSPORT INTEGRATION SYSTEM - TEST SUITE")
    print("="*60)
    
    tests = [
        ("System Initialization", test_system_initialization),
        ("Configuration System", test_configuration),
        ("Data Loading", test_data_loading),
        ("Database Operations", test_database_operations)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("🎉 All tests passed! System is ready for use.")
        print("\nNext steps:")
        print("1. Run: python transport_integration.py --build-transport")
        print("2. Run: python transport_integration.py --integrate-incidents")
        print("3. Run: python transport_integration.py --report-only")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
