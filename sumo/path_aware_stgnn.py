import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

class PathAwareSTGNN(nn.Module):
    """
    Path-Aware Spatio-Temporal Graph Neural Network for incident-aware path prediction.
    Takes multiple paths as input and predicts the best path based on incident severity.
    """
    
    def __init__(self, num_nodes, num_features, hidden_dim=64, num_layers=2, dropout=0.1):
        super(PathAwareSTGNN, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        
        # Graph Convolutional Layers for spatial feature extraction
        self.gcn_layers = nn.ModuleList([
            nn.Linear(num_features if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Path encoder for multiple paths
        self.path_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Path comparison network
        self.path_comparator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Output: incident severity score
        )
        
        # Path ranking network
        self.path_ranker = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Output: path quality score
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj_matrix, paths):
        """
        Forward pass for path-aware prediction.
        
        Args:
            x: Traffic features (batch_size, seq_len, num_nodes, num_features)
            adj_matrix: Adjacency matrix (num_nodes, num_nodes)
            paths: List of paths, each path is a list of node indices
        
        Returns:
            path_scores: Scores for each path (lower = better)
            incident_predictions: Predicted incident severity for each path
        """
        batch_size, seq_len, num_nodes, num_features = x.shape
        
        # 1. Extract spatial-temporal features using GCN + LSTM
        x_reshaped = x.view(batch_size * seq_len, num_nodes, num_features)
        
        # Apply GCN layers
        for i, gcn_layer in enumerate(self.gcn_layers):
            x_reshaped = torch.matmul(adj_matrix, x_reshaped)
            x_reshaped = gcn_layer(x_reshaped)
            if i < len(self.gcn_layers) - 1:
                x_reshaped = F.relu(x_reshaped)
                x_reshaped = self.dropout(x_reshaped)
        
        # Reshape back for LSTM
        x_reshaped = x_reshaped.view(batch_size, seq_len, num_nodes, self.hidden_dim)
        
        # Apply LSTM to each node
        node_features = []
        for node in range(num_nodes):
            node_data = x_reshaped[:, :, node, :]
            lstm_out, _ = self.lstm(node_data)
            node_features.append(lstm_out[:, -1, :])  # Last timestep
        
        # Stack node features: (batch_size, num_nodes, hidden_dim)
        node_features = torch.stack(node_features, dim=1)
        
        # 2. Process each path
        path_scores = []
        incident_predictions = []
        
        for path in paths:
            # Extract features for nodes in this path
            path_features = node_features[:, path, :]  # (batch_size, path_length, hidden_dim)
            
            # Encode path using LSTM
            path_encoded, _ = self.path_encoder(path_features)
            path_summary = path_encoded[:, -1, :]  # Last timestep of path
            
            # Predict incident severity for this path
            # Use average of path features and path summary
            path_avg = torch.mean(path_features, dim=1)  # Average over path nodes
            combined_features = torch.cat([path_summary, path_avg], dim=-1)
            
            incident_score = self.path_comparator(combined_features)
            path_quality = self.path_ranker(path_summary)
            
            # Lower score = better path (less incidents)
            path_score = incident_score + (1.0 - path_quality)  # Combine incident and quality
            
            path_scores.append(path_score)
            incident_predictions.append(incident_score)
        
        return torch.stack(path_scores, dim=1), torch.stack(incident_predictions, dim=1)

class PathDataProcessor:
    """Data processor for path-aware STGNN."""
    
    def __init__(self, adj_matrix_path, features_path, incidents_path):
        self.adj_matrix_path = adj_matrix_path
        self.features_path = features_path
        self.incidents_path = incidents_path
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and preprocess the data."""
        print("Loading adjacency matrix...")
        self.adj_matrix = np.load(self.adj_matrix_path)
        
        print("Loading time series features...")
        self.features_df = pd.read_csv(self.features_path, index_col=0)
        
        print("Loading incidents data...")
        with open(self.incidents_path, 'r', encoding='utf-8') as f:
            self.incidents_data = json.load(f)
        
        print(f"Adjacency matrix shape: {self.adj_matrix.shape}")
        print(f"Features shape: {self.features_df.shape}")
        print(f"Incidents loaded: {len(self.incidents_data.get('features', []))}")
        
        return self.adj_matrix, self.features_df, self.incidents_data
    
    def parse_paths_from_geojson(self, geojson_paths):
        """
        Parse multiple paths from GeoJSON files.
        
        Args:
            geojson_paths: List of GeoJSON file paths containing route geometries
        
        Returns:
            paths: List of paths, each path is a list of node indices
            path_metadata: Metadata for each path
        """
        paths = []
        path_metadata = []
        
        for geojson_path in geojson_paths:
            with open(geojson_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for feature in data.get('features', []):
                if feature['geometry']['type'] == 'LineString':
                    coordinates = feature['geometry']['coordinates']
                    
                    # Convert coordinates to node indices
                    # This is a simplified mapping - you'll need to implement
                    # proper coordinate-to-node mapping based on your network
                    path_nodes = self._coordinates_to_nodes(coordinates)
                    
                    if len(path_nodes) > 1:  # Valid path
                        paths.append(path_nodes)
                        path_metadata.append({
                            'coordinates': coordinates,
                            'properties': feature.get('properties', {}),
                            'path_id': len(paths) - 1
                        })
        
        return paths, path_metadata
    
    def _coordinates_to_nodes(self, coordinates):
        """
        Convert GeoJSON coordinates to network node indices.
        This is a placeholder - implement proper coordinate mapping.
        """
        # Simplified mapping - replace with actual coordinate-to-node logic
        # For now, return random node indices
        num_nodes = self.adj_matrix.shape[0]
        path_length = min(len(coordinates), 10)  # Limit path length
        return np.random.choice(num_nodes, path_length, replace=False).tolist()
    
    def create_incident_labels(self, paths, path_metadata):
        """
        Create incident severity labels for each path.
        
        Args:
            paths: List of paths
            path_metadata: Metadata for each path
        
        Returns:
            incident_labels: Incident severity scores for each path
        """
        incident_labels = []
        
        for i, path in enumerate(paths):
            # Calculate incident severity for this path
            # This is a simplified calculation - implement based on your incident data
            severity = 0.0
            
            # Check for incidents along the path
            for node in path:
                # Look for incidents affecting this node
                # This is where you'd check your incident database
                node_incidents = self._get_incidents_for_node(node)
                severity += sum(incident.get('severity', 0) for incident in node_incidents)
            
            # Normalize severity (0 = no incidents, 1 = maximum incidents)
            max_severity = len(path) * 1.0  # Maximum possible severity
            normalized_severity = min(severity / max_severity, 1.0)
            
            incident_labels.append(normalized_severity)
        
        return np.array(incident_labels)
    
    def _get_incidents_for_node(self, node_id):
        """Get incidents affecting a specific node."""
        # Placeholder - implement based on your incident data structure
        return []
    
    def prepare_training_data(self, paths, sequence_length=6):
        """
        Prepare training data for path-aware model.
        
        Args:
            paths: List of paths
            sequence_length: Number of timesteps for input
        
        Returns:
            X: Input features (batch_size, seq_len, num_nodes, num_features)
            y: Target incident scores for each path
        """
        # Extract speed and flow features
        speed_cols = [col for col in self.features_df.columns if '_meanSpeed' in col]
        flow_cols = [col for col in self.features_df.columns if '_flow' in col]
        
        num_timesteps, num_nodes = len(self.features_df), len(speed_cols)
        
        # Create feature matrix
        X = np.zeros((num_timesteps, num_nodes, 2))
        
        for i, (speed_col, flow_col) in enumerate(zip(speed_cols, flow_cols)):
            X[:, i, 0] = self.features_df[speed_col].values
            X[:, i, 1] = self.features_df[flow_col].values
        
        # Normalize features
        X_reshaped = X.reshape(-1, 2)
        X_normalized = self.scaler.fit_transform(X_reshaped)
        X_normalized = X_normalized.reshape(X.shape)
        
        # Create sequences
        X_sequences = []
        for i in range(len(X_normalized) - sequence_length + 1):
            X_seq = X_normalized[i:i+sequence_length]
            X_sequences.append(X_seq)
        
        X_sequences = np.array(X_sequences)
        
        # Create incident labels for paths
        incident_labels = self.create_incident_labels(paths, [])
        
        return X_sequences, incident_labels

def train_path_aware_model(model, train_loader, test_loader, adj_matrix, num_epochs=100):
    """Train the path-aware STGNN model."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    adj_matrix = torch.FloatTensor(adj_matrix).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print(f"Training on device: {device}")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_paths, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            path_scores, incident_preds = model(batch_X, adj_matrix, batch_paths)
            
            # Calculate loss (lower scores should be better paths)
            loss = criterion(path_scores, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d}: Train Loss: {train_loss/len(train_loader):.6f}')
    
    return model

def predict_best_path(model, traffic_data, adj_matrix, candidate_paths):
    """
    Predict the best path among candidates.
    
    Args:
        model: Trained PathAwareSTGNN model
        traffic_data: Current traffic conditions
        adj_matrix: Road network adjacency matrix
        candidate_paths: List of candidate paths
    
    Returns:
        best_path_idx: Index of the best path
        path_scores: Scores for all paths (lower = better)
    """
    model.eval()
    
    with torch.no_grad():
        # Prepare input data
        traffic_tensor = torch.FloatTensor(traffic_data).unsqueeze(0)  # Add batch dimension
        adj_tensor = torch.FloatTensor(adj_matrix)
        
        # Get predictions
        path_scores, incident_preds = model(traffic_tensor, adj_tensor, candidate_paths)
        
        # Find best path (lowest score)
        path_scores_np = path_scores.cpu().numpy().flatten()
        best_path_idx = np.argmin(path_scores_np)
        
        return best_path_idx, path_scores_np

def save_prediction_results(prediction_results, candidate_paths, output_dir='sumo/predictions'):
    """
    Save prediction results to various file formats.
    
    Args:
        prediction_results: Dictionary with prediction results
        candidate_paths: List of candidate paths
        output_dir: Directory to save results
    """
    import os
    import json
    from datetime import datetime
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save detailed JSON results
    json_file = os.path.join(output_dir, f'path_predictions_{timestamp}.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(prediction_results, f, indent=2, ensure_ascii=False)
    
    # 2. Save CSV summary
    csv_file = os.path.join(output_dir, f'path_summary_{timestamp}.csv')
    summary_data = []
    for i, (score, incident) in enumerate(zip(prediction_results['path_scores'], 
                                            prediction_results['incident_predictions'])):
        summary_data.append({
            'path_index': i,
            'path_score': score,
            'incident_severity': incident,
            'is_best': i == prediction_results['best_path_index'],
            'rank': prediction_results['path_rankings'].index(i) + 1
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(csv_file, index=False)
    
    # 3. Save best path as GeoJSON
    if prediction_results['best_path_index'] < len(candidate_paths):
        best_path_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": candidate_paths[prediction_results['best_path_index']]
                    },
                    "properties": {
                        "path_id": f"best_path_{timestamp}",
                        "score": prediction_results['best_path_score'],
                        "rank": 1,
                        "prediction_time": prediction_results['timestamp']
                    }
                }
            ]
        }
        
        geojson_file = os.path.join(output_dir, f'best_path_{timestamp}.geojson')
        with open(geojson_file, 'w', encoding='utf-8') as f:
            json.dump(best_path_geojson, f, indent=2, ensure_ascii=False)
    
    # 4. Save model performance log
    log_file = os.path.join(output_dir, f'prediction_log_{timestamp}.txt')
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Path Prediction Results - {prediction_results['timestamp']}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Best Path Index: {prediction_results['best_path_index']}\n")
        f.write(f"Best Path Score: {prediction_results['best_path_score']:.4f}\n")
        f.write(f"Total Paths Analyzed: {len(prediction_results['path_scores'])}\n\n")
        
        f.write("Path Rankings:\n")
        f.write("-" * 20 + "\n")
        for rank, path_idx in enumerate(prediction_results['path_rankings'], 1):
            score = prediction_results['path_scores'][path_idx]
            incident = prediction_results['incident_predictions'][path_idx]
            f.write(f"Rank {rank}: Path {path_idx} (Score: {score:.4f}, Incidents: {incident:.4f})\n")
    
    print(f"Predictions saved to {output_dir}/")
    print(f"  - JSON: {json_file}")
    print(f"  - CSV: {csv_file}")
    print(f"  - GeoJSON: {geojson_file}")
    print(f"  - Log: {log_file}")

def predict_and_save(model, traffic_data, adj_matrix, candidate_paths, path_metadata=None):
    """
    Predict best path and save results to files.
    
    Args:
        model: Trained PathAwareSTGNN model
        traffic_data: Current traffic conditions
        adj_matrix: Road network adjacency matrix
        candidate_paths: List of candidate paths
        path_metadata: Metadata for each path
    
    Returns:
        best_path_idx: Index of the best path
        path_scores: Scores for all paths
        prediction_results: Detailed results dictionary
    """
    model.eval()
    
    with torch.no_grad():
        # Prepare input data
        traffic_tensor = torch.FloatTensor(traffic_data).unsqueeze(0)
        adj_tensor = torch.FloatTensor(adj_matrix)
        
        # Get predictions
        path_scores, incident_preds = model(traffic_tensor, adj_tensor, candidate_paths)
        
        # Convert to numpy
        path_scores_np = path_scores.cpu().numpy().flatten()
        incident_preds_np = incident_preds.cpu().numpy().flatten()
        best_path_idx = np.argmin(path_scores_np)
        
        # Create detailed results
        prediction_results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'best_path_index': int(best_path_idx),
            'best_path_score': float(path_scores_np[best_path_idx]),
            'path_scores': path_scores_np.tolist(),
            'incident_predictions': incident_preds_np.tolist(),
            'path_rankings': np.argsort(path_scores_np).tolist(),
            'path_metadata': path_metadata or []
        }
        
        # Save results
        save_prediction_results(prediction_results, candidate_paths)
        
        return best_path_idx, path_scores_np, prediction_results

def main():
    """Main training pipeline for path-aware STGNN."""
    print("=== Path-Aware STGNN for Incident-Aware Routing ===")
    
    # Initialize data processor
    processor = PathDataProcessor(
        adj_matrix_path='sumo/adj_matrix.npy',
        features_path='sumo/time_series_features.csv',
        incidents_path='sumo/tom_tom_incidents.geojson'
    )
    
    # Load data
    adj_matrix, features_df, incidents_data = processor.load_data()
    
    # Parse paths from GeoJSON files
    geojson_paths = [
        'sumo/route_main_street.geojson',
        'sumo/route_highway.geojson', 
        'sumo/route_residential.geojson'
    ]
    
    paths, path_metadata = processor.parse_paths_from_geojson(geojson_paths)
    print(f"Parsed {len(paths)} paths")
    
    # Prepare training data
    X, y = processor.prepare_training_data(paths)
    
    # Initialize model
    num_nodes = adj_matrix.shape[0]
    num_features = 2
    
    model = PathAwareSTGNN(
        num_nodes=num_nodes,
        num_features=num_features,
        hidden_dim=32,
        num_layers=2,
        dropout=0.1
    )
    
    print(f"Model initialized with {num_nodes} nodes and {num_features} features")
    print(f"Processing {len(paths)} candidate paths")
    
    # Example usage with prediction saving
    print("\n=== Example Path Prediction ===")
    print("Given multiple paths, the model will predict the best one based on incident severity.")
    print("Lower scores indicate better paths (fewer incidents).")
    
    # Example prediction with saving
    if len(paths) > 0:
        print("\nRunning prediction and saving results...")
        best_idx, scores, results = predict_and_save(
            model, X[0], adj_matrix, paths, path_metadata
        )
        print(f"Best path: {best_idx} with score: {scores[best_idx]:.4f}")
    
    return model, adj_matrix, paths

if __name__ == "__main__":
    model, adj_matrix, paths = main()
