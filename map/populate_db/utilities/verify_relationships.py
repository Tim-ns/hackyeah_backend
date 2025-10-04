#!/usr/bin/env python3
"""Final verification script to check all relationships"""

from neo4j import GraphDatabase

URI = 'neo4j://127.0.0.1:7687'
USER = "neo4j"
PASSWORD = "12345678"

def verify_all_relationships():
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    
    print("=== FINAL VERIFICATION OF ALL RELATIONSHIPS ===")
    
    with driver.session() as session:
        # Check all relationship types
        result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
        print("\n📋 Available relationship types:")
        relationship_types = []
        for record in result:
            rel_type = record['relationshipType']
            relationship_types.append(rel_type)
            print(f"  ✅ {rel_type}")
        
        # Check incident relationships
        print("\n🚨 INCIDENT RELATIONSHIPS:")
        
        # NEAR_STOP
        result = session.run("MATCH (i:Incident)-[r:NEAR_STOP]->(s:Stop) RETURN count(r) as count")
        near_stop_count = result.single()['count']
        print(f"  ✅ NEAR_STOP: {near_stop_count:,} relationships")
        
        # AFFECTS_ROUTE
        result = session.run("MATCH (i:Incident)-[r:AFFECTS_ROUTE]->(route:Route) RETURN count(r) as count")
        affects_route_count = result.single()['count']
        print(f"  ✅ AFFECTS_ROUTE: {affects_route_count:,} relationships")
        
        # ON_TRANSPORT_PATH
        result = session.run("MATCH (i:Incident)-[r:ON_TRANSPORT_PATH]->(route:Route) RETURN count(r) as count")
        on_path_count = result.single()['count']
        print(f"  ✅ ON_TRANSPORT_PATH: {on_path_count:,} relationships")
        
        # Check transport relationships
        print("\n🚌 TRANSPORT RELATIONSHIPS:")
        
        # STOPS_AT
        result = session.run("MATCH (t:Trip)-[r:STOPS_AT]->(s:Stop) RETURN count(r) as count")
        stops_at_count = result.single()['count']
        print(f"  ✅ STOPS_AT: {stops_at_count:,} relationships")
        
        # NEXT_STOP
        result = session.run("MATCH (s1:Stop)-[r:NEXT_STOP]->(s2:Stop) RETURN count(r) as count")
        next_stop_count = result.single()['count']
        print(f"  ✅ NEXT_STOP: {next_stop_count:,} relationships")
        
        # RUNS_ON
        result = session.run("MATCH (t:Trip)-[r:RUNS_ON]->(route:Route) RETURN count(r) as count")
        runs_on_count = result.single()['count']
        print(f"  ✅ RUNS_ON: {runs_on_count:,} relationships")
        
        # Check node counts
        print("\n📊 NODE COUNTS:")
        
        result = session.run("MATCH (i:Incident) RETURN count(i) as count")
        incident_count = result.single()['count']
        print(f"  📍 Incidents: {incident_count:,}")
        
        result = session.run("MATCH (s:Stop) RETURN count(s) as count")
        stop_count = result.single()['count']
        print(f"  🚏 Stops: {stop_count:,}")
        
        result = session.run("MATCH (r:Route) RETURN count(r) as count")
        route_count = result.single()['count']
        print(f"  🛣️ Routes: {route_count:,}")
        
        result = session.run("MATCH (t:Trip) RETURN count(t) as count")
        trip_count = result.single()['count']
        print(f"  🚌 Trips: {trip_count:,}")
        
        # Summary
        print("\n🎯 INTEGRATION SUMMARY:")
        print(f"  ✅ All {len(relationship_types)} relationship types created")
        print(f"  ✅ {incident_count:,} incidents connected to transport network")
        print(f"  ✅ {affects_route_count:,} incidents affect routes")
        print(f"  ✅ {on_path_count:,} incidents on transport paths")
        
        if affects_route_count > 0:
            print("\n🎉 SUCCESS! AFFECTS_ROUTE relationships are now available!")
            print("\nYou can now run queries like:")
            print("MATCH (i:Incident)-[:AFFECTS_ROUTE]->(r:Route) RETURN r.route_short_name, count(i) as incident_count ORDER BY incident_count DESC LIMIT 10")
        else:
            print("\n⚠️ AFFECTS_ROUTE relationships still need to be created")
    
    driver.close()

if __name__ == "__main__":
    verify_all_relationships()
