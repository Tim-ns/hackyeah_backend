import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

class STGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network for traffic prediction.
    Combines Graph Convolutional Networks (GCN) with LSTM for spatio-temporal modeling.
    """
    
    def __init__(self, num_nodes, num_features, hidden_dim=64, num_layers=2, dropout=0.1):
        super(STGNN, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Graph Convolutional Layers
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
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, num_features)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj_matrix):
        """
        Forward pass through the STGNN.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_nodes, num_features)
            adj_matrix: Adjacency matrix of shape (num_nodes, num_nodes)
        
        Returns:
            Output tensor of shape (batch_size, num_nodes, num_features)
        """
        batch_size, seq_len, num_nodes, num_features = x.shape
        
        # Reshape for GCN processing: (batch_size * seq_len, num_nodes, num_features)
        x_reshaped = x.view(batch_size * seq_len, num_nodes, num_features)
        
        # Apply GCN layers
        for i, gcn_layer in enumerate(self.gcn_layers):
            # Graph convolution: X' = AXW
            x_reshaped = torch.matmul(adj_matrix, x_reshaped)  # (batch*seq, nodes, features)
            x_reshaped = gcn_layer(x_reshaped)  # Linear transformation
            if i < len(self.gcn_layers) - 1:
                x_reshaped = F.relu(x_reshaped)
                x_reshaped = self.dropout(x_reshaped)
        
        # Reshape back for LSTM: (batch_size, seq_len, num_nodes, hidden_dim)
        x_reshaped = x_reshaped.view(batch_size, seq_len, num_nodes, self.hidden_dim)
        
        # Apply LSTM to each node independently
        outputs = []
        for node in range(num_nodes):
            node_data = x_reshaped[:, :, node, :]  # (batch_size, seq_len, hidden_dim)
            lstm_out, _ = self.lstm(node_data)
            outputs.append(lstm_out[:, -1, :])  # Take last timestep
        
        # Stack outputs: (batch_size, num_nodes, hidden_dim)
        lstm_output = torch.stack(outputs, dim=1)
        
        # Final output layer
        output = self.output_layer(lstm_output)
        
        return output

class STGNNDataProcessor:
    """Data processor for STGNN training."""
    
    def __init__(self, adj_matrix_path, features_path):
        self.adj_matrix_path = adj_matrix_path
        self.features_path = features_path
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and preprocess the data."""
        print("Loading adjacency matrix...")
        self.adj_matrix = np.load(self.adj_matrix_path)
        
        print("Loading time series features...")
        self.features_df = pd.read_csv(self.features_path, index_col=0)
        
        print(f"Adjacency matrix shape: {self.adj_matrix.shape}")
        print(f"Features shape: {self.features_df.shape}")
        
        return self.adj_matrix, self.features_df
    
    def prepare_features(self, sequence_length=6, prediction_horizon=1):
        """
        Prepare features for STGNN training.
        
        Args:
            sequence_length: Number of timesteps to use as input
            prediction_horizon: Number of timesteps to predict ahead
        """
        # Extract speed and flow features
        speed_cols = [col for col in self.features_df.columns if '_meanSpeed' in col]
        flow_cols = [col for col in self.features_df.columns if '_flow' in col]
        
        # Create feature matrix (time, nodes, features)
        num_timesteps, num_nodes = len(self.features_df), len(speed_cols)
        
        # Reshape features: (timesteps, nodes, 2) where 2 = [speed, flow]
        X = np.zeros((num_timesteps, num_nodes, 2))
        
        for i, (speed_col, flow_col) in enumerate(zip(speed_cols, flow_cols)):
            X[:, i, 0] = self.features_df[speed_col].values  # Speed
            X[:, i, 1] = self.features_df[flow_col].values   # Flow
        
        # Normalize features
        X_reshaped = X.reshape(-1, 2)
        X_normalized = self.scaler.fit_transform(X_reshaped)
        X_normalized = X_normalized.reshape(X.shape)
        
        # Create sequences for training
        X_sequences, y_sequences = [], []
        
        for i in range(len(X_normalized) - sequence_length - prediction_horizon + 1):
            X_seq = X_normalized[i:i+sequence_length]
            y_seq = X_normalized[i+sequence_length:i+sequence_length+prediction_horizon]
            
            X_sequences.append(X_seq)
            y_sequences.append(y_seq)
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        print(f"Created sequences: {X_sequences.shape}")
        print(f"Target sequences: {y_sequences.shape}")
        
        return X_sequences, y_sequences
    
    def create_train_test_split(self, X, y, test_size=0.2, random_state=42):
        """Create train/test split."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test

def train_stgnn(model, train_loader, test_loader, adj_matrix, num_epochs=100, learning_rate=0.001):
    """Train the STGNN model."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    adj_matrix = torch.FloatTensor(adj_matrix).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    train_losses = []
    test_losses = []
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X, adj_matrix)
            
            # Reshape for loss calculation
            batch_y_reshaped = batch_y.squeeze(1)  # Remove prediction horizon dimension
            
            loss = criterion(outputs, batch_y_reshaped)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Testing
        model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_X, adj_matrix)
                batch_y_reshaped = batch_y.squeeze(1)
                
                loss = criterion(outputs, batch_y_reshaped)
                test_loss += loss.item()
        
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')
    
    return train_losses, test_losses

def plot_training_curves(train_losses, test_losses):
    """Plot training curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('STGNN Training Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('sumo/stgnn_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training pipeline."""
    print("=== STGNN Traffic Prediction Model ===")
    
    # Initialize data processor
    processor = STGNNDataProcessor(
        adj_matrix_path='sumo/adj_matrix.npy',
        features_path='sumo/time_series_features.csv'
    )
    
    # Load data
    adj_matrix, features_df = processor.load_data()
    
    # Prepare features
    X, y = processor.prepare_features(sequence_length=6, prediction_horizon=1)
    
    # Create train/test split
    X_train, X_test, y_train, y_test = processor.create_train_test_split(X, y)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Initialize model
    num_nodes = adj_matrix.shape[0]
    num_features = 2  # speed and flow
    
    model = STGNN(
        num_nodes=num_nodes,
        num_features=num_features,
        hidden_dim=32,  # Reduced for limited data
        num_layers=2,
        dropout=0.1
    )
    
    print(f"Model initialized with {num_nodes} nodes and {num_features} features")
    
    # Train model
    train_losses, test_losses = train_stgnn(
        model, train_loader, test_loader, adj_matrix,
        num_epochs=50, learning_rate=0.001
    )
    
    # Plot results
    plot_training_curves(train_losses, test_losses)
    
    # Save model
    torch.save(model.state_dict(), 'sumo/stgnn_model.pth')
    print("Model saved to sumo/stgnn_model.pth")
    
    print("=== Training Complete ===")

if __name__ == "__main__":
    main()
