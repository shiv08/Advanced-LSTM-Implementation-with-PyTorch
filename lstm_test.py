import torch
import torch.nn as nn
import numpy as np
from lstm import create_lstm_model
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Tuple
import math

# Custom Dataset for Sequence Data
class SequenceDataset(Dataset):
    """Generate sequences where each number is the sum of previous two numbers plus noise"""
    def __init__(self, num_sequences: int, seq_length: int, noise_level: float = 0.1):
        self.sequences = []
        self.targets = []
        
        for _ in range(num_sequences):
            # Start with two random numbers between 0 and 1
            seq = [np.random.rand(), np.random.rand()]
            
            # Generate sequence
            for i in range(seq_length):
                # Next number is sum of previous two plus noise
                next_val = seq[-1] + seq[-2] + np.random.normal(0, noise_level)
                seq.append(next_val)
            
            # Convert to input sequence and target
            input_seq = seq[:-1]  # All but last
            target = seq[-1]      # Last value
            
            self.sequences.append(input_seq)
            self.targets.append(target)
            
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = torch.FloatTensor(self.sequences[idx]).unsqueeze(-1)  # Add feature dimension
        target = torch.FloatTensor([self.targets[idx]])
        return sequence, target

# Training function
def train_model(model: nn.Module, 
                train_loader: DataLoader,
                val_loader: DataLoader,
                num_epochs: int,
                learning_rate: float,
                device: torch.device) -> Tuple[list, list]:
    """Train the LSTM model and return training history"""
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses

# Visualization function
def plot_training_history(train_losses: list, val_losses: list):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_model_predictions(model: nn.Module, 
                         test_loader: DataLoader, 
                         device: torch.device,
                         num_examples: int = 5):
    """Test model predictions and compare with actual values"""
    model.eval()
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            predictions = model(sequences)
            
            # Print some example predictions
            for i in range(min(num_examples, len(sequences))):
                print(f"\nSequence: {sequences[i].cpu().numpy().flatten()[-5:]}")  # Last 5 numbers
                print(f"Predicted next value: {predictions[i].item():.4f}")
                print(f"Actual next value: {targets[i].item():.4f}")
                print(f"Absolute Error: {abs(predictions[i].item() - targets[i].item()):.4f}")
            
            break  # Only show first batch

# Main testing script
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    config = {
        'input_size': 1,         # Single feature (number sequence)
        'hidden_size': 32,       # Size of LSTM hidden state
        'num_layers': 2,         # Number of LSTM layers
        'output_size': 1,        # Predict one number
        'dropout': 0.1,
        'bidirectional': True    # Use bidirectional LSTM
    }
    
    # Dataset parameters
    num_sequences = 1000
    seq_length = 10
    batch_size = 32
    
    # Create datasets
    train_dataset = SequenceDataset(num_sequences, seq_length)
    val_dataset = SequenceDataset(num_sequences // 5, seq_length)  # 20% size of training
    test_dataset = SequenceDataset(num_sequences // 5, seq_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_lstm_model(config).to(device)
    
    # Training parameters
    num_epochs = 30
    learning_rate = 0.001
    
    print("Starting training...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, num_epochs, learning_rate, device
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses)
    
    # Test model predictions
    print("\nTesting model predictions:")
    test_model_predictions(model, test_loader, device)