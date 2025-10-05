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
        
        # Setup device (CUDA if available, otherwise CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Create model directory
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Initialize model (will be loaded or created)
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
    
    def create_model(self, num_nodes, num_features, hidden_dim=32):
        """Create a new STGNN model."""
        # Import PathAwareSTGNN from the same directory
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
        
        # Move model to device
        self.model = self.model.to(self.device)
        print(f"Model moved to {self.device}")
        
        # Save initial model
        self.save_model()
    
    def load_model(self):
        """Load existing model and optimizer state."""
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Load model architecture
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from path_aware_stgnn import PathAwareSTGNN
        self.model = PathAwareSTGNN(**checkpoint['model_config'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scaler
        if 'scaler' in checkpoint:
            self.scaler = checkpoint['scaler']
        
        # Load training history
        if 'update_count' in checkpoint:
            self.update_count = checkpoint['update_count']
        if 'performance_history' in checkpoint:
            self.performance_history = checkpoint['performance_history']
        
        # Move model to device
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
                'num_layers': 2,  # Fixed value
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
        
        # Prepare new data
        X_new, y_new = self._prepare_incremental_data(new_traffic_data, new_incidents, paths)
        
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
    
    def _prepare_incremental_data(self, traffic_data, incidents, paths):
        """Prepare new data for incremental learning."""
        # Convert traffic data to sequences
        sequence_length = 6
        X_new = []
        y_new = []
        
        if len(traffic_data) < sequence_length:
            return np.array([]), np.array([])
        
        for i in range(len(traffic_data) - sequence_length + 1):
            X_seq = traffic_data[i:i+sequence_length]
            X_new.append(X_seq)
        
        X_new = np.array(X_new)
        
        # Create incident labels for paths
        incident_labels = self._calculate_incident_labels(incidents, paths)
        y_new = np.tile(incident_labels, (len(X_new), 1))
        
        return X_new, y_new
    
    def _calculate_incident_labels(self, incidents, paths):
        """Calculate incident severity labels for paths."""
        incident_labels = []
        
        for path in paths:
            severity = 0.0
            
            # Check for incidents along the path
            for node in path:
                # Look for incidents affecting this node
                node_incidents = self._get_incidents_for_node(incidents, node)
                severity += sum(incident.get('severity', 0) for incident in node_incidents)
            
            # Normalize severity
            max_severity = len(path) * 1.0
            normalized_severity = min(severity / max_severity, 1.0)
            incident_labels.append(normalized_severity)
        
        return np.array(incident_labels)
    
    def _get_incidents_for_node(self, incidents, node_id):
        """Get incidents affecting a specific node."""
        # Simplified incident matching - implement based on your data structure
        node_incidents = []
        for incident in incidents:
            if self._incident_affects_node(incident, node_id):
                node_incidents.append(incident)
        return node_incidents
    
    def _incident_affects_node(self, incident, node_id):
        """Check if incident affects a specific node."""
        # Placeholder - implement based on your incident data structure
        # This could be based on spatial proximity, road segments, etc.
        return False  # Simplified for now
    
    def _incremental_learning_step(self, X_new, y_new, paths):
        """Perform one incremental learning step."""
        self.model.train()
        
        # Convert to tensors and move to device
        X_tensor = torch.FloatTensor(X_new).to(self.device)
        y_tensor = torch.FloatTensor(y_new).to(self.device)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Training loop
        total_loss = 0.0
        for batch_X, batch_y in dataloader:
            self.optimizer.zero_grad()
            
            # Forward pass (simplified - you'll need to adapt based on your model)
            # This is a placeholder - implement based on your actual model forward pass
            outputs = self.model(batch_X, None, paths)  # You'll need to provide adj_matrix
            
            # Calculate loss
            loss = nn.MSELoss()(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Record performance
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
            # Prepare input data and move to device
            traffic_tensor = torch.FloatTensor(traffic_data).unsqueeze(0).to(self.device)
            adj_tensor = torch.FloatTensor(adj_matrix).to(self.device)
            
            # Get predictions
            path_scores, incident_preds = self.model(traffic_tensor, adj_tensor, candidate_paths)
            
            # Convert to numpy
            path_scores_np = path_scores.cpu().numpy().flatten()
            incident_preds_np = incident_preds.cpu().numpy().flatten()
            best_path_idx = np.argmin(path_scores_np)
            
            return best_path_idx, path_scores_np, incident_preds_np
    
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
        
        # Process new incidents
        processed_incidents = self._process_incidents(incident_data)
        self.incident_cache.extend(processed_incidents)
        
        # Save to incremental incident file
        self._save_incremental_incidents(processed_incidents)
        
        return processed_incidents
    
    def add_new_traffic_data(self, traffic_data):
        """Add new traffic data without full SUMO simulation."""
        print(f"Adding new traffic data with {len(traffic_data)} timesteps...")
        
        # Process new traffic data
        processed_traffic = self._process_traffic_data(traffic_data)
        self.traffic_cache.extend(processed_traffic)
        
        # Append to existing traffic file
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
        # Convert to standardized format
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
        
        # Load existing incidents
        if os.path.exists(incident_file):
            with open(incident_file, 'r') as f:
                existing_incidents = json.load(f)
        else:
            existing_incidents = []
        
        # Add new incidents
        existing_incidents.extend(incidents)
        
        # Save updated incidents
        with open(incident_file, 'w') as f:
            json.dump(existing_incidents, f, indent=2)
        
        print(f"Incremental incidents saved to {incident_file}")
    
    def _append_traffic_data(self, traffic_data):
        """Append new traffic data to existing file."""
        traffic_file = os.path.join(self.base_data_dir, 'incremental_traffic.json')
        
        # Load existing traffic data
        if os.path.exists(traffic_file):
            with open(traffic_file, 'r') as f:
                existing_traffic = json.load(f)
        else:
            existing_traffic = []
        
        # Add new traffic data
        existing_traffic.extend(traffic_data)
        
        # Save updated traffic data
        with open(traffic_file, 'w') as f:
            json.dump(existing_traffic, f, indent=2)
        
        print(f"Incremental traffic data saved to {traffic_file}")

def main():
    """Example usage of continuous learning system."""
    print("=== Continuous Learning STGNN System ===")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA is available! Using GPU acceleration.")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available, using CPU (slower training)")
    
    # Initialize continuous learning system
    cl_system = ContinuousLearningSTGNN(
        model_path='sumo/models/continuous_stgnn.pth',
        learning_rate=0.001,
        update_frequency=50
    )
    
    # Load or create model
    cl_system.load_or_create_model(
        num_nodes=33095,  # Your network size
        num_features=2,   # Speed and flow
        hidden_dim=32
    )
    
    # Initialize data processor
    data_processor = IncrementalDataProcessor()
    
    # Example: Add new incidents
    new_incidents = [
        {
            'id': 'incident_001',
            'timestamp': datetime.now().isoformat(),
            'severity': 0.8,
            'location': {'lat': 50.0647, 'lon': 19.9445},
            'type': 'accident'
        },
        {
            'id': 'incident_002', 
            'timestamp': datetime.now().isoformat(),
            'severity': 0.3,
            'location': {'lat': 50.0650, 'lon': 19.9450},
            'type': 'construction'
        }
    ]
    
    # Process new incidents
    processed_incidents = data_processor.add_new_incidents(new_incidents)
    
    # Example: Add new traffic data
    new_traffic = [
        {
            'timestamp': datetime.now().isoformat(),
            'speed': 45.2,
            'flow': 120.5,
            'density': 2.7
        }
    ]
    
    # Process new traffic data
    processed_traffic = data_processor.add_new_traffic_data(new_traffic)
    
    # Perform incremental update
    paths = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # Example paths
    cl_system.incremental_update(
        new_traffic_data=np.array([processed_traffic]),
        new_incidents=processed_incidents,
        paths=paths
    )
    
    print("Continuous learning system ready!")
    print("You can now add new data without rerunning the full SUMO simulation.")
    
    return cl_system, data_processor

if __name__ == "__main__":
    cl_system, data_processor = main()
