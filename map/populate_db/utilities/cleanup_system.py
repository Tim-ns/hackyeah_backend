#!/usr/bin/env python3
"""
Cleanup script to remove old files and keep only the professional refactored system
"""

import os
import shutil
from pathlib import Path

def cleanup_old_files():
    """Remove old files and keep only the professional system"""
    
    # Files to keep (the professional refactored system)
    keep_files = {
        'transport_integration.py',
        'README_TRANSPORT_INTEGRATION.md',
        'test_transport_system.py'
    }
    
    # Files to remove (old versions)
    remove_files = {
        'map/load_incidents.py',
        'map/incident_analysis.py', 
        'map/incident_transport_integration.py',
        'check_db_status.py',
        'debug_affects_route.py',
        'load_stop_times_only.py',
        'load_all_stop_times.py',
        'create_affects_route.py',
        'create_next_stop_relationships.py',
        'complete_incident_integration.py',
        'verify_relationships.py',
        'check_transport_graph.py'
    }
    
    print("🧹 CLEANUP: Removing old files...")
    
    removed_count = 0
    for file_path in remove_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"  ✅ Removed: {file_path}")
                removed_count += 1
            except Exception as e:
                print(f"  ❌ Failed to remove {file_path}: {e}")
        else:
            print(f"  ⚠️ File not found: {file_path}")
    
    print(f"\n📊 Cleanup Summary:")
    print(f"  Files removed: {removed_count}")
    print(f"  Files kept: {len(keep_files)}")
    
    print(f"\n🎯 PROFESSIONAL SYSTEM FILES:")
    for file_name in keep_files:
        if os.path.exists(file_name):
            print(f"  ✅ {file_name}")
        else:
            print(f"  ❌ Missing: {file_name}")

def create_usage_examples():
    """Create usage examples file"""
    
    examples_content = """#!/usr/bin/env python3
'''
Usage Examples for Transport Integration System
'''

# Example 1: Complete Integration
# python transport_integration.py --full-integration

# Example 2: Step-by-Step Integration
# python transport_integration.py --build-transport
# python transport_integration.py --integrate-incidents

# Example 3: Generate Report
# python transport_integration.py --report-only

# Example 4: Custom Configuration
# python transport_integration.py --full-integration --stop-distance 150 --path-distance 75 --batch-size 2000

# Example 5: Testing with Limited Data
# python transport_integration.py --integrate-incidents --limit 1000 --verbose

# Example 6: Test System
# python test_transport_system.py

print("See README_TRANSPORT_INTEGRATION.md for detailed usage instructions")
"""
    
    with open('usage_examples.py', 'w') as f:
        f.write(examples_content)
    
    print("✅ Created usage_examples.py")

def main():
    """Main cleanup function"""
    print("="*60)
    print("PROFESSIONAL TRANSPORT INTEGRATION SYSTEM - CLEANUP")
    print("="*60)
    
    cleanup_old_files()
    create_usage_examples()
    
    print("\n" + "="*60)
    print("🎉 CLEANUP COMPLETE!")
    print("="*60)
    
    print("\n📁 PROFESSIONAL SYSTEM STRUCTURE:")
    print("├── transport_integration.py          # Main system")
    print("├── README_TRANSPORT_INTEGRATION.md   # Documentation")
    print("├── test_transport_system.py          # Test suite")
    print("├── usage_examples.py                # Usage examples")
    print("└── transport_integration.log       # Log file (created on first run)")
    
    print("\n🚀 QUICK START:")
    print("1. Test system: python test_transport_system.py")
    print("2. Full integration: python transport_integration.py --full-integration")
    print("3. Generate report: python transport_integration.py --report-only")
    
    print("\n📚 For detailed instructions, see README_TRANSPORT_INTEGRATION.md")

if __name__ == "__main__":
    main()
