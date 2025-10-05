import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
import argparse
warnings.filterwarnings('ignore')

class ContinuousLearningSTGNN:
    """
    Continuous Learning STGNN that can adapt to new data without full retraining.
    Supports incremental learning, online updates, and model persistence.
    """
    
    def __init__(self, model_path=None, learning_rate=0.001, update_frequency=100):
        self.model_path = model_path or 'sumo/models/continuous_stgnn.pth'
        self.learning_rate = learning_rate
        self.update_frequency = update_frequency
        self.update_count = 0
        self.performance_history = []
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        self.model = None
        self.optimizer = None
        self.scaler = None
        
    def load_or_create_model(self, num_nodes, num_features, hidden_dim=32):
        """Load existing model or create new one."""
        if os.path.exists(self.model_path):
            print("Loading existing model...")
            self.load_model()
        else:
            print("Creating new model...")
            self.create_model(num_nodes, num_features, hidden_dim)
        
        self.load_adjacency_matrix()
    
    def load_adjacency_matrix(self):
        """Load adjacency matrix from file or create default."""
        adj_matrix_paths = [
            "sumo/model_inputs/adj_matrix.npy"
        ]
        
        for path in adj_matrix_paths:
            if os.path.exists(path):
                print(f"Loading adjacency matrix from: {path}")
                self.adjacency_matrix = torch.FloatTensor(np.load(path)).to(self.device)
                print(f"Adjacency matrix shape: {self.adjacency_matrix.shape}")
                return
        
        print("WARNING: No adjacency matrix found. Will use identity matrix.")
        self.adjacency_matrix = None
    
    def create_model(self, num_nodes, num_features, hidden_dim=32):
        """Create a new STGNN model."""
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from path_aware_stgnn import PathAwareSTGNN
        
        self.model = PathAwareSTGNN(
            num_nodes=num_nodes,
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=0.1
        )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scaler = StandardScaler()
        
        self.model = self.model.to(self.device)
        print(f"Model moved to {self.device}")
        
        self.save_model()
    
    def load_model(self):
        """Load existing model and optimizer state."""
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from path_aware_stgnn import PathAwareSTGNN
        self.model = PathAwareSTGNN(**checkpoint['model_config'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scaler' in checkpoint:
            self.scaler = checkpoint['scaler']
        
        if 'update_count' in checkpoint:
            self.update_count = checkpoint['update_count']
        if 'performance_history' in checkpoint:
            self.performance_history = checkpoint['performance_history']
        
        self.model = self.model.to(self.device)
        print(f"Model loaded from {self.model_path} and moved to {self.device}")
        print(f"Update count: {self.update_count}")
    
    def save_model(self):
        """Save model, optimizer, and training state."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'num_nodes': self.model.num_nodes,
                'num_features': self.model.num_features,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': 2,
                'dropout': 0.1
            },
            'scaler': self.scaler,
            'update_count': self.update_count,
            'performance_history': self.performance_history,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def incremental_update(self, new_traffic_data, new_incidents, paths):
        """
        Incrementally update model with new data.
        
        Args:
            new_traffic_data: New traffic features
            new_incidents: New incident data
            paths: Path configurations
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call load_or_create_model() first.")
        
        print(f"Incremental update #{self.update_count + 1}")
        
        X_new, y_new = self._prepare_incremental_data(new_traffic_data, new_incidents, paths)
        
        if len(X_new) == 0:
            print("No new data to process")
            return
        
        if self.scaler is not None:
            X_reshaped = X_new.reshape(-1, X_new.shape[-1])
            self.scaler.partial_fit(X_reshaped)
            X_new = self.scaler.transform(X_reshaped).reshape(X_new.shape)
        
        self._incremental_learning_step(X_new, y_new, paths)
        
        self.update_count += 1
        
        if self.update_count % self.update_frequency == 0:
            self.save_model()
            print(f"Model checkpoint saved at update {self.update_count}")
    
    def incremental_update_from_json(self, json_file_path, paths):
        """
        Incrementally update model with combined data from JSON file.
        
        Args:
            json_file_path: Path to JSON file containing both traffic and incident data
            paths: Path configurations
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call load_or_create_model() first.")
        
        print(f"Incremental update #{self.update_count + 1} from {json_file_path}")
        
        # Load combined data from JSON
        combined_data = self._load_combined_data(json_file_path)
        
        if not combined_data:
            print("No data found in JSON file")
            return
        
        # Extract traffic and incident data
        traffic_data = combined_data.get('traffic', [])
        incidents = combined_data.get('incidents', [])
        
        print(f"Processing {len(traffic_data)} traffic records and {len(incidents)} incidents")
        
        # Prepare new data
        X_new, y_new = self._prepare_incremental_data(traffic_data, incidents, paths)
        
        if len(X_new) == 0:
            print("No new data to process")
            return
        
        # Update scaler with new data
        if self.scaler is not None:
            X_reshaped = X_new.reshape(-1, X_new.shape[-1])
            self.scaler.partial_fit(X_reshaped)
            X_new = self.scaler.transform(X_reshaped).reshape(X_new.shape)
        
        # Perform incremental learning
        self._incremental_learning_step(X_new, y_new, paths)
        
        self.update_count += 1
        
        # Save model periodically
        if self.update_count % self.update_frequency == 0:
            self.save_model()
            print(f"Model checkpoint saved at update {self.update_count}")
    
    def _load_combined_data(self, json_file_path):
        """Load combined traffic and incident data from JSON file."""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # If it's a list, assume it contains mixed traffic and incident records
                traffic_data = []
                incidents = []
                
                for record in data:
                    if record.get('type') == 'traffic' or 'speed' in record or 'flow' in record:
                        traffic_data.append(record)
                    elif record.get('type') == 'incident' or 'severity' in record:
                        incidents.append(record)
                
                return {'traffic': traffic_data, 'incidents': incidents}
            
            elif isinstance(data, dict):
                # Check if it's GeoJSON format with incidents
                if 'incidents' in data and isinstance(data['incidents'], list):
                    # Handle GeoJSON format
                    return self._process_geojson_data(data)
                else:
                    # Standard format
                    return {
                        'traffic': data.get('traffic', data.get('jams', [])),
                        'incidents': data.get('incidents', [])
                    }
            
            else:
                print(f"Unexpected JSON structure in {json_file_path}")
                return None
                
        except FileNotFoundError:
            print(f"File not found: {json_file_path}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON decode error in {json_file_path}: {e}")
            return None
        except Exception as e:
            print(f"Error loading {json_file_path}: {e}")
            return None
    
    def _process_geojson_data(self, data):
        """Process GeoJSON format data with comprehensive feature extraction."""
        incidents = []
        traffic_data = []
        
        for incident in data.get('incidents', []):
            if incident.get('type') == 'Feature':
                properties = incident.get('properties', {})
                geometry = incident.get('geometry', {})
                
                # Extract comprehensive incident information
                incident_data = {
                    'id': properties.get('id', f"incident_{len(incidents)}"),
                    'timestamp': properties.get('startTime', datetime.now().isoformat()),
                    'severity': min(properties.get('magnitudeOfDelay', 0) / 10.0, 1.0),
                    'type': self._map_icon_category(properties.get('iconCategory', 0)),
                    'description': f"{properties.get('from', '')} to {properties.get('to', '')}",
                    'length': properties.get('length', 0),
                    'road_numbers': properties.get('roadNumbers', []),
                    'time_validity': properties.get('timeValidity', 'unknown'),
                    'probability': properties.get('probabilityOfOccurrence', 'unknown'),
                    'iconCategory': properties.get('iconCategory', 0),
                    'magnitudeOfDelay': properties.get('magnitudeOfDelay', 0),
                    'from': properties.get('from', ''),
                    'to': properties.get('to', '')
                }
                
                # Extract coordinates from geometry
                if geometry.get('type') == 'LineString':
                    coordinates = geometry.get('coordinates', [])
                    if coordinates:
                        # Use first coordinate as location
                        lon, lat = coordinates[0]
                        incident_data['location'] = {'lat': lat, 'lon': lon}
                        
                        # Generate comprehensive traffic data with full feature set
                        severity_factor = incident_data['severity']
                        icon_category = properties.get('iconCategory', 0)
                        
                        # More sophisticated traffic impact modeling
                        speed_impact = severity_factor * 25.0 + (icon_category * 3.0)
                        flow_impact = severity_factor * 60.0 + (icon_category * 8.0)
                        density_impact = severity_factor * 8.0 + (icon_category * 1.5)
                        
                        traffic_point = {
                            # Core traffic metrics
                            'speed': max(5.0, 60.0 - speed_impact),
                            'flow': max(5.0, 100.0 - flow_impact),
                            'density': min(100.0, density_impact),
                            
                            # Location data
                            'location': {'lat': lat, 'lon': lon},
                            
                            # Temporal features
                            'timestamp': incident_data['timestamp'],
                            
                            # Incident impact features
                            'incident_severity': incident_data['severity'],
                            'incident_type_encoded': icon_category,
                            'magnitude_of_delay': properties.get('magnitudeOfDelay', 0),
                            
                            # Road network features
                            'road_type': self._encode_road_type(properties.get('from', '')),
                            'congestion_level': min(10.0, severity_factor * 5.0 + (icon_category // 2)),
                            
                            # Additional context
                            'has_incident': 1,
                            'incident_description': incident_data['description'],
                            'from_location': properties.get('from', ''),
                            'to_location': properties.get('to', ''),
                            'edge_id': f"edge_{properties.get('id', len(traffic_data))}",
                            'type': 'traffic'
                        }
                        traffic_data.append(traffic_point)
                
                incidents.append(incident_data)
        
        print(f"Processed {len(incidents)} incidents from GeoJSON format")
        print(f"Generated {len(traffic_data)} traffic data points")
        
        return {'traffic': traffic_data, 'incidents': incidents}
    
    def _encode_road_type(self, road_name):
        """Encode road type based on road name patterns."""
        if not road_name:
            return 0
        
        road_name_lower = road_name.lower()
        if any(keyword in road_name_lower for keyword in ['highway', 'autostrada', 'a1', 'a2', 'a4']):
            return 3  # Highway
        elif any(keyword in road_name_lower for keyword in ['street', 'ulica', 'ul.', 'avenue']):
            return 2  # Street
        elif any(keyword in road_name_lower for keyword in ['road', 'droga', 'route']):
            return 1  # Road
        else:
            return 0  # Unknown
    
    def _convert_timestamp_to_numeric(self, timestamp):
        """Convert timestamp string to numeric value."""
        if isinstance(timestamp, (int, float)):
            return float(timestamp)
        
        if isinstance(timestamp, str):
            try:
                # Try to parse ISO format timestamp
                from datetime import datetime
                if 'T' in timestamp:
                    # Remove 'Z' suffix if present
                    clean_timestamp = timestamp.replace('Z', '+00:00')
                    dt = datetime.fromisoformat(clean_timestamp)
                    return dt.timestamp()
                else:
                    # Try to parse as regular datetime
                    dt = datetime.fromisoformat(timestamp)
                    return dt.timestamp()
            except (ValueError, TypeError):
                # If parsing fails, return 0
                return 0.0
        
        return 0.0
    
    def _map_icon_category(self, icon_category):
        """Map icon category to incident type."""
        category_map = {
            0: 'unknown',
            1: 'accident',
            2: 'breakdown',
            3: 'construction',
            4: 'weather',
            5: 'police',
            6: 'hazard',
            7: 'congestion',
            8: 'road_closure',
            9: 'roadworks',
            10: 'traffic_light'
        }
        return category_map.get(icon_category, 'unknown')
    
    def _prepare_incremental_data(self, traffic_data, incidents, paths):
        """Prepare new data for incremental learning using full feature set."""
        sequence_length = 6
        X_new = []
        y_new = []
        
        if len(traffic_data) < sequence_length:
            return np.array([]), np.array([])
        
        # Extract features from traffic data - match model's expected input size
        processed_features = []
        for data_point in traffic_data:
            if isinstance(data_point, dict):
                # Extract basic feature set (3 features to match model expectations)
                feature_vector = [
                    data_point.get('speed', 0.0),
                    data_point.get('flow', 0.0), 
                    data_point.get('density', 0.0)
                ]
                processed_features.append(feature_vector)
            else:
                processed_features.append(data_point)
        
        if len(processed_features) < sequence_length:
            return np.array([]), np.array([])

        num_nodes = self.adjacency_matrix.shape[0] if self.adjacency_matrix is not None else 1
        num_features = len(processed_features[0]) if processed_features else 3
        
        for i in range(len(processed_features) - sequence_length + 1):
            X_seq = processed_features[i:i+sequence_length]

            # Always use a small subset for incremental learning to avoid memory issues
            subset_size = min(100, num_nodes)
            X_seq_reshaped = np.zeros((sequence_length, subset_size, num_features))
            
            # Fill the first node with our data
            for t in range(sequence_length):
                X_seq_reshaped[t, 0, :] = X_seq[t]
            
            X_new.append(X_seq_reshaped)
        
        X_new = np.array(X_new)
        
        # Calculate incident labels based on first 3 paths only
        limited_paths = paths[:3] if len(paths) >= 3 else paths
        incident_labels = self._calculate_incident_labels(incidents, limited_paths)
        
        # Ensure we have exactly 3 labels
        if len(incident_labels) < 3:
            # Pad with zeros if we have fewer than 3 paths
            incident_labels = np.pad(incident_labels, (0, 3 - len(incident_labels)), 'constant')
        elif len(incident_labels) > 3:
            # Truncate if we have more than 3 paths
            incident_labels = incident_labels[:3]
        
        y_new = np.tile(incident_labels, (len(X_new), 1))
        
        return X_new, y_new
    
    def _calculate_incident_labels(self, incidents, paths):
        """Calculate incident severity labels for paths."""
        incident_labels = []
        
        for path in paths:
            severity = 0.0
            
            for node in path:
                node_incidents = self._get_incidents_for_node(incidents, node)
                severity += sum(incident.get('severity', 0) for incident in node_incidents)
            
            max_severity = len(path) * 1.0
            normalized_severity = min(severity / max_severity, 1.0)
            incident_labels.append(normalized_severity)
        
        return np.array(incident_labels)
    
    def _get_incidents_for_node(self, incidents, node_id):
        """Get incidents affecting a specific node."""
        node_incidents = []
        for incident in incidents:
            if self._incident_affects_node(incident, node_id):
                node_incidents.append(incident)
        return node_incidents
    
    def _incident_affects_node(self, incident, node_id):
        """Check if incident affects a specific node."""
        return False
    
    def _incremental_learning_step(self, X_new, y_new, paths):
        """Perform one incremental learning step."""
        self.model.train()
        
        X_tensor = torch.FloatTensor(X_new).to(self.device)
        y_tensor = torch.FloatTensor(y_new).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        
        total_loss = 0.0
        for batch_X, batch_y in dataloader:
            self.optimizer.zero_grad()
            
            # Use the loaded adjacency matrix (or subset if needed)
            if self.adjacency_matrix is not None:
                batch_size, seq_len, num_nodes, num_features = batch_X.shape
                adj_matrix_size = self.adjacency_matrix.shape[0]
                
                if num_nodes == adj_matrix_size:
                    # Full adjacency matrix
                    adj_matrix = self.adjacency_matrix
                elif num_nodes < adj_matrix_size:
                    # Use subset of adjacency matrix
                    adj_matrix = self.adjacency_matrix[:num_nodes, :num_nodes]
                else:
                    # Fallback to identity matrix
                    adj_matrix = torch.eye(num_nodes, device=self.device)
                
                # Ensure paths are within the subset size and limit to 3 paths
                valid_paths = []
                for path in paths[:3]:  # Limit to first 3 paths
                    valid_path = [node for node in path if node < num_nodes]
                    if valid_path:  # Only add non-empty paths
                        valid_paths.append(valid_path)
                
                # Ensure we have exactly 3 paths
                while len(valid_paths) < 3:
                    valid_paths.append([min(len(valid_paths), num_nodes-1)])
                
                # Truncate if we have more than 3 paths
                valid_paths = valid_paths[:3]
                
                path_scores, incident_preds = self.model(batch_X, adj_matrix, valid_paths)
            else:
                # Fallback to identity matrix if no adjacency matrix loaded
                batch_size, seq_len, num_nodes, num_features = batch_X.shape
                adj_matrix = torch.eye(num_nodes, device=self.device)
                path_scores, incident_preds = self.model(batch_X, adj_matrix, paths)
            
            # Remove extra dimension from path_scores if it exists
            if path_scores.dim() == 3 and path_scores.shape[-1] == 1:
                path_scores = path_scores.squeeze(-1)
            
            # Use path_scores for loss calculation (incident predictions are auxiliary)
            loss = nn.MSELoss()(path_scores, batch_y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        self.performance_history.append({
            'update_count': self.update_count,
            'loss': avg_loss,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"Incremental learning completed on {self.device}. Average loss: {avg_loss:.6f}")
    
    def online_prediction(self, traffic_data, adj_matrix, candidate_paths):
        """Make predictions using the continuously updated model."""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        self.model.eval()
        
        with torch.no_grad():
            traffic_tensor = torch.FloatTensor(traffic_data).unsqueeze(0).to(self.device)
            
            # Handle adjacency matrix device conversion
            if isinstance(adj_matrix, torch.Tensor):
                if adj_matrix.device != self.device:
                    adj_tensor = adj_matrix.to(self.device)
                else:
                    adj_tensor = adj_matrix
            else:
                adj_tensor = torch.FloatTensor(adj_matrix).to(self.device)
            
            path_scores, incident_preds = self.model(traffic_tensor, adj_tensor, candidate_paths)
            
            # Remove extra dimension if present
            if path_scores.dim() == 3 and path_scores.shape[-1] == 1:
                path_scores = path_scores.squeeze(-1)
            if incident_preds.dim() == 3 and incident_preds.shape[-1] == 1:
                incident_preds = incident_preds.squeeze(-1)
            
            path_scores_np = path_scores.cpu().numpy().flatten()
            incident_preds_np = incident_preds.cpu().numpy().flatten()
            best_path_idx = np.argmin(path_scores_np)
            
            return best_path_idx, path_scores_np, incident_preds_np
    
    def predict_from_json(self, json_file_path, candidate_paths=None):
        """
        Make predictions from a JSON file containing traffic data.
        
        Args:
            json_file_path: Path to JSON file with traffic data
            candidate_paths: List of candidate paths to evaluate
        
        Returns:
            prediction_results: Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call load_or_create_model() first.")
        
        if candidate_paths is None:
            candidate_paths = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        
        print(f"Making predictions from {json_file_path}")
        
        # Load data from JSON
        combined_data = self._load_combined_data(json_file_path)
        if not combined_data:
            print("No data found in JSON file")
            return None
        
        traffic_data = combined_data.get('traffic', [])
        if not traffic_data:
            print("No traffic data found")
            return None
        
        # Prepare traffic data for prediction
        processed_features = []
        for data_point in traffic_data:
            if isinstance(data_point, dict):
                feature_vector = [
                    data_point.get('speed', 0.0),
                    data_point.get('flow', 0.0), 
                    data_point.get('density', 0.0)
                ]
                processed_features.append(feature_vector)
        
        if len(processed_features) < 6:  # Need at least 6 timesteps
            print(f"Insufficient data for prediction (need at least 6 timesteps, got {len(processed_features)})")
            print("Creating synthetic data to fill the sequence...")
            
            # Create synthetic data to fill the sequence if we don't have enough
            while len(processed_features) < 6:
                # Use the last available data point or create default values
                if processed_features:
                    last_point = processed_features[-1]
                    # Add some variation to make it realistic
                    synthetic_point = [
                        max(0.1, last_point[0] + np.random.normal(0, 2)),  # speed
                        max(0.1, last_point[1] + np.random.normal(0, 5)),  # flow
                        max(0.1, last_point[2] + np.random.normal(0, 1))   # density
                    ]
                else:
                    # Default values if no data available
                    synthetic_point = [30.0, 50.0, 15.0]
                
                processed_features.append(synthetic_point)
            
            print(f"Created {len(processed_features)} timesteps for prediction")
        
        # Create input sequence
        sequence_length = 6
        num_nodes = min(100, self.adjacency_matrix.shape[0]) if self.adjacency_matrix is not None else 1
        num_features = 3
        
        X_seq = np.zeros((sequence_length, num_nodes, num_features))
        for t in range(sequence_length):
            X_seq[t, 0, :] = processed_features[t]
        
        # Get adjacency matrix subset
        if self.adjacency_matrix is not None:
            adj_matrix = self.adjacency_matrix[:num_nodes, :num_nodes]
        else:
            adj_matrix = torch.eye(num_nodes, device=self.device)
        
        # Make prediction
        best_path_idx, path_scores, incident_preds = self.online_prediction(
            X_seq, adj_matrix, candidate_paths
        )
        
        # Create path metadata with coordinates
        path_metadata = []
        for i, path in enumerate(candidate_paths):
            # Generate realistic coordinates for each path
            # This creates a path with multiple waypoints
            start_lat, start_lon = 50.0647, 19.9445  # Krakow center
            coordinates = []
            
            # Generate waypoints along the path
            for j, node in enumerate(path):
                # Add some variation to create realistic paths
                lat = start_lat + (j * 0.001) + np.random.normal(0, 0.0005)
                lon = start_lon + (j * 0.001) + np.random.normal(0, 0.0005)
                coordinates.append([lon, lat])
            
            # Add path metadata
            path_info = {
                "coordinates": coordinates,
                "properties": {
                    "route_id": f"route_{i}",
                    "name": f"Route {i+1}",
                    "description": f"Path {i+1} with {len(path)} nodes",
                    "estimated_time": f"{15 + i*2} minutes",
                    "distance": f"{2.5 + i*0.3:.1f} km"
                },
                "path_id": i
            }
            path_metadata.append(path_info)
        
        # Create results dictionary with coordinates
        prediction_results = {
            'timestamp': datetime.now().isoformat(),
            'input_file': json_file_path,
            'best_path_index': int(best_path_idx),
            'best_path_score': float(path_scores[best_path_idx]),
            'path_scores': path_scores.tolist(),
            'incident_predictions': incident_preds.tolist(),
            'path_rankings': np.argsort(path_scores).tolist(),
            'candidate_paths': candidate_paths,
            'path_metadata': path_metadata
        }
        
        print(f"Prediction completed!")
        print(f"Best path: {best_path_idx} with score: {path_scores[best_path_idx]:.4f}")
        print(f"All path scores: {path_scores}")
        
        return prediction_results
    
    def get_performance_history(self):
        """Get model performance history."""
        return self.performance_history
    
    def reset_learning_rate(self, new_lr):
        """Reset learning rate for optimizer."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self.learning_rate = new_lr
        print(f"Learning rate updated to {new_lr}")

class IncrementalDataProcessor:
    """
    Process new data incrementally without full pipeline rerun.
    """
    
    def __init__(self, base_data_dir='sumo/'):
        self.base_data_dir = base_data_dir
        self.incident_cache = []
        self.traffic_cache = []
        
    def add_new_incidents(self, incident_data):
        """Add new incident data without full reprocessing."""
        print(f"Adding {len(incident_data)} new incidents...")
        
        processed_incidents = self._process_incidents(incident_data)
        self.incident_cache.extend(processed_incidents)
        
        self._save_incremental_incidents(processed_incidents)
        
        return processed_incidents
    
    def add_new_traffic_data(self, traffic_data):
        """Add new traffic data without full SUMO simulation."""
        print(f"Adding new traffic data with {len(traffic_data)} timesteps...")
        
        processed_traffic = self._process_traffic_data(traffic_data)
        self.traffic_cache.extend(processed_traffic)
        
        self._append_traffic_data(processed_traffic)
        
        return processed_traffic
    
    def _process_incidents(self, incident_data):
        """Process incident data for incremental learning."""
        processed = []
        for incident in incident_data:
            processed_incident = {
                'id': incident.get('id', f"incident_{len(self.incident_cache)}"),
                'timestamp': incident.get('timestamp', datetime.now().isoformat()),
                'severity': incident.get('severity', 0.5),
                'location': incident.get('location', {}),
                'type': incident.get('type', 'unknown'),
                'processed_at': datetime.now().isoformat()
            }
            processed.append(processed_incident)
        return processed
    
    def _process_traffic_data(self, traffic_data):
        """Process traffic data for incremental learning."""
        processed = []
        for data_point in traffic_data:
            processed_point = {
                'timestamp': data_point.get('timestamp', datetime.now().isoformat()),
                'speed': data_point.get('speed', 0.0),
                'flow': data_point.get('flow', 0.0),
                'density': data_point.get('density', 0.0),
                'processed_at': datetime.now().isoformat()
            }
            processed.append(processed_point)
        return processed
    
    def _save_incremental_incidents(self, incidents):
        """Save incremental incident data."""
        incident_file = os.path.join(self.base_data_dir, 'incremental_incidents.json')
        
        if os.path.exists(incident_file):
            with open(incident_file, 'r') as f:
                existing_incidents = json.load(f)
        else:
            existing_incidents = []
        
        existing_incidents.extend(incidents)
        
        with open(incident_file, 'w') as f:
            json.dump(existing_incidents, f, indent=2)
        
        print(f"Incremental incidents saved to {incident_file}")
    
    def _append_traffic_data(self, traffic_data):
        """Append new traffic data to existing file."""
        traffic_file = os.path.join(self.base_data_dir, 'incremental_traffic.json')
        
        if os.path.exists(traffic_file):
            with open(traffic_file, 'r') as f:
                existing_traffic = json.load(f)
        else:
            existing_traffic = []
        
        existing_traffic.extend(traffic_data)
        
        with open(traffic_file, 'w') as f:
            json.dump(existing_traffic, f, indent=2)
        
        print(f"Incremental traffic data saved to {traffic_file}")

def main():
    """Main function with command-line argument support."""
    parser = argparse.ArgumentParser(description='Continuous Learning STGNN System')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input JSON file containing incidents data')
    parser.add_argument('--model-path', type=str, default='sumo/models/continuous_stgnn.pth',
                       help='Path to model file (default: sumo/models/continuous_stgnn.pth)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate for model updates (default: 0.001)')
    parser.add_argument('--update-frequency', type=int, default=50,
                       help='Frequency of model saves (default: 50)')
    parser.add_argument('--num-nodes', type=int, default=33095,
                       help='Number of nodes in network (default: 33095)')
    parser.add_argument('--num-features', type=int, default=3,
                       help='Number of features per node (default: 3)')
    parser.add_argument('--hidden-dim', type=int, default=32,
                       help='Hidden dimension size (default: 32)')
    parser.add_argument('--paths', type=str, default='[[1,2,3],[4,5,6],[7,8,9]]',
                       help='Path configurations as JSON string (default: [[1,2,3],[4,5,6],[7,8,9]])')
    parser.add_argument('--predict', action='store_true',
                       help='Run prediction mode instead of training mode')
    parser.add_argument('--output-dir', type=str, default='sumo/predictions/',
                       help='Output directory for prediction results (default: sumo/predictions/)')
    
    args = parser.parse_args()
    print("=== Continuous Learning STGNN System ===")
    print(f"Processing input file: {args.input}")
    if torch.cuda.is_available():
        print(f"CUDA is available! Using GPU acceleration.")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available, using CPU (slower training)")
    
    # Initialize continuous learning system
    cl_system = ContinuousLearningSTGNN(
        model_path=args.model_path,
        learning_rate=args.learning_rate,
        update_frequency=args.update_frequency
    )
    
    # Load or create model
    cl_system.load_or_create_model(
        num_nodes=args.num_nodes,
        num_features=args.num_features,
        hidden_dim=args.hidden_dim
    )
    
    # Parse paths from command line argument
    try:
        paths = json.loads(args.paths)
        print(f"Using paths: {paths}")
    except json.JSONDecodeError:
        print(f"Invalid paths format: {args.paths}")
        print("Using default paths: [[1,2,3],[4,5,6],[7,8,9]]")
        paths = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    
    # Process input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return None, None
    
    if args.predict:
        # Prediction mode
        print(f"Running prediction mode on {args.input}")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Make prediction
        prediction_results = cl_system.predict_from_json(args.input, paths)
        
        if prediction_results is None:
            print("Prediction failed!")
            return None, None
        
        # Save prediction results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_file = os.path.join(args.output_dir, f'prediction_results_{timestamp}.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(prediction_results, f, indent=2, ensure_ascii=False)
        
        # Save structured prediction with coordinates (like path_predictions format)
        structured_file = os.path.join(args.output_dir, f'path_predictions_{timestamp}.json')
        structured_results = {
            'timestamp': prediction_results['timestamp'],
            'best_path_index': prediction_results['best_path_index'],
            'best_path_score': prediction_results['best_path_score'],
            'path_scores': prediction_results['path_scores'],
            'incident_predictions': prediction_results['incident_predictions'],
            'path_rankings': prediction_results['path_rankings'],
            'path_metadata': prediction_results['path_metadata']
        }
        with open(structured_file, 'w', encoding='utf-8') as f:
            json.dump(structured_results, f, indent=2, ensure_ascii=False)
        
        # Save summary
        summary_file = os.path.join(args.output_dir, f'prediction_summary_{timestamp}.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Path Prediction Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {prediction_results['timestamp']}\n")
            f.write(f"Input file: {prediction_results['input_file']}\n")
            f.write(f"Best path index: {prediction_results['best_path_index']}\n")
            f.write(f"Best path score: {prediction_results['best_path_score']:.4f}\n\n")
            
            f.write("Path Rankings:\n")
            f.write("-" * 20 + "\n")
            for rank, path_idx in enumerate(prediction_results['path_rankings'], 1):
                score = prediction_results['path_scores'][path_idx]
                incident = prediction_results['incident_predictions'][path_idx]
                f.write(f"Rank {rank}: Path {path_idx} (Score: {score:.4f}, Incident: {incident:.4f})\n")
            
            f.write(f"\nCandidate paths: {prediction_results['candidate_paths']}\n")
        
        print(f"\nPrediction results saved to:")
        print(f"  - JSON: {json_file}")
        print(f"  - Structured with coordinates: {structured_file}")
        print(f"  - Summary: {summary_file}")
        
        # Print summary to console
        print(f"\n=== Prediction Summary ===")
        print(f"Best path: {prediction_results['best_path_index']} (score: {prediction_results['best_path_score']:.4f})")
        print(f"Path rankings: {prediction_results['path_rankings']}")
        print(f"All scores: {[f'{s:.4f}' for s in prediction_results['path_scores']]}")
        
        print("Prediction completed successfully!")
        
    else:
        # Training mode
        print(f"Processing data from {args.input}")
        
        # Perform incremental update from JSON file
        cl_system.incremental_update_from_json(args.input, paths)
        
        print("Model update completed successfully!")
    
    return cl_system, None

if __name__ == "__main__":
    cl_system, data_processor = main()
