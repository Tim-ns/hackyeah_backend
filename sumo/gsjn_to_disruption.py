import json
import xml.etree.ElementTree as ET
import sumolib
from datetime import datetime
from shapely.geometry import LineString, shape
from shapely.wkt import loads as loads_wkt

GEOJSON_FILE = 'sumo/tom_tom_incidents.geojson'
NET_FILE = 'sumo/krakow_traffic.net.xml'
OUTPUT_FILE = 'sumo/krakow_incidents.add.xml'
MAX_MAP_DISTANCE = 50.0
DEFAULT_SPEED_FACTOR = 0.2
DEFAULT_DURATION = 3600

def map_coordinates_to_edge(net, lon, lat):
    """Finds the closest SUMO edge to the given lon/lat coordinates."""
    # sumolib requires coordinates in the network's projection (typically UTM).
    # net.convertLonLat2XY transforms lon/lat to the internal projection.
    x, y = net.convertLonLat2XY(lon, lat)
    
    # getEdgesNearby returns (edge, distance) tuples
    edges = net.getNeighboringEdges(x, y, MAX_MAP_DISTANCE)
    
    if edges:
        # Return the ID of the closest edge
        return edges[0][0].getID()
    return None

def calculate_time_duration(start_time_str, end_time_str):
    """Calculates start and end simulation time steps (in seconds)."""
    
    # Define a simulation start time (e.g., 2024-01-01 00:00:00 UTC)
    # SUMO typically works with seconds from the simulation start (time=0).
    SIM_START_EPOCH = datetime(2024, 1, 1, 0, 0, 0).timestamp()

    start_dt = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
    start_sim_time = int(start_dt.timestamp() - SIM_START_EPOCH)
    
    if end_time_str:
        end_dt = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
        end_sim_time = int(end_dt.timestamp() - SIM_START_EPOCH)
    else:
        # If no end time, use a default duration
        end_sim_time = start_sim_time + DEFAULT_DURATION
    
    return max(0, start_sim_time), max(start_sim_time + 1, end_sim_time)

# --- Main Logic ---

def generate_incident_xml():
    """Reads GeoJSON, maps incidents to network, and generates SUMO XML."""
    
    print(f"Loading SUMO network: {NET_FILE}...")
    try:
        net = sumolib.net.readNet(NET_FILE)
    except Exception as e:
        print(f"Error loading network: {e}")
        print("Please ensure 'my_city.net.xml' exists and is a valid SUMO network file.")
        return

    print(f"Loading GeoJSON incidents: {GEOJSON_FILE}...")
    with open(GEOJSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Setup XML root
    root = ET.Element("additional")

    incident_count = 0
    
    for feature in data.get('features', []):
        incident_count += 1
        
        properties = feature.get('properties', {})
        geom = feature.get('geometry')
        
        # We focus only on features with LineString geometry
        if geom and geom['type'] == 'LineString':
            
            # --- Extract Incident Metadata ---
            
            # Use 'magnitudeOfDelay' or 'iconCategory' to determine severity (speed factor)
            delay_magnitude = properties.get('magnitudeOfDelay', 0)
            
            # A simplified speed factor logic: higher delay magnitude means lower speed factor
            speed_factor = DEFAULT_SPEED_FACTOR if delay_magnitude > 2 else 0.5
            
            start_time_str = properties.get('startTime')
            end_time_str = properties.get('endTime')
            
            if not start_time_str:
                print(f"Warning: Incident {incident_count} has no 'startTime', skipping.")
                continue

            start_time, end_time = calculate_time_duration(start_time_str, end_time_str)
            
            # --- Map-Matching (Simplified) ---
            
            # Get start/end coordinates (Lon/Lat)
            coordinates = geom['coordinates']
            start_lon, start_lat = coordinates[0]
            end_lon, end_lat = coordinates[-1]
            
            # Map start and end points to SUMO edges
            start_edge_id = map_coordinates_to_edge(net, start_lon, start_lat)
            end_edge_id = map_coordinates_to_edge(net, end_lon, end_lat)

            if not start_edge_id or not end_edge_id:
                # print(f"Warning: Could not map start/end coordinates for incident {incident_count} to the network. Skipping.")
                continue

            # Find all edges along the path using shortest path routing
            # This is a robust way to map a sequence of edges between two points
            try:
                # We need the full edge objects, not just IDs
                start_edge = net.getEdge(start_edge_id)
                end_edge = net.getEdge(end_edge_id)

                # Use SUMO's routing engine to get the full list of intermediate edges
                # (Assuming travel time cost)
                route = net.getShortestPath(start_edge, end_edge)
                edge_ids = [e.getID() for e in route[0]]
                
            except Exception as e:
                # print(f"Warning: Failed to find route between mapped edges for incident {incident_count}. Skipping.")
                continue

            # --- Generate XML Data ---
            
            # Create a variable speed sign to apply speed reduction
            vss = ET.SubElement(root, "variableSpeedSign", 
                              id=f"incident_{incident_count}",
                              lanes=" ".join([f"{edge_id}_0" for edge_id in edge_ids]))
            
            # Apply speed factor during the incident period
            ET.SubElement(vss, "step", 
                         time=str(start_time),
                         speed=str(speed_factor))
            ET.SubElement(vss, "step", 
                         time=str(end_time),
                         speed="1.0")

    # --- Write Output ---
    
    tree = ET.ElementTree(root)
    # Beautify the XML output
    xml_string = ET.tostring(root, encoding='unicode')
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">\n')
        
        # Simple indentation for readability (cannot use xml.dom.minidom reliably here)
        for line in xml_string.split('\n'):
            f.write(line.replace('><', '>\n<') + '\n')
            
        f.write('</additional>')

    print(f"\nSuccessfully processed {incident_count} incidents.")
    print(f"Output saved to {OUTPUT_FILE}.")
    print("This file should be included in your .sumocfg file as an <additional-files> entry.")

if __name__ == "__main__":
    generate_incident_xml()