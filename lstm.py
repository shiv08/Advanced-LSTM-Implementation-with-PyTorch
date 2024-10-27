import torch
import torch.nn as nn
import math
from typing import Tuple, List, Optional

class LSTMCell(nn.Module):
    """
    LSTM Cell implementation with layer normalization.
    
    Mathematical formulation of LSTM:
    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)    # Forget gate
    i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    # Input gate
    g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)  # Candidate cell state
    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)    # Output gate
    
    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t       # New cell state
    h_t = o_t ⊙ tanh(c_t)                  # New hidden state
    
    where:
    - σ is the sigmoid function
    - ⊙ is element-wise multiplication
    - [h_{t-1}, x_t] represents concatenation
    """
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Combined weight matrices for efficiency
        # W_ih combines weights for [i_t, f_t, g_t, o_t] for input x_t
        # W_hh combines weights for [i_t, f_t, g_t, o_t] for hidden state h_{t-1}
        self.weight_ih = nn.Linear(input_size, 4 * hidden_size)
        self.weight_hh = nn.Linear(hidden_size, 4 * hidden_size)
        
        # Layer Normalization for better training stability
        self.layer_norm_x = nn.LayerNorm(4 * hidden_size)  # Normalize gate pre-activations
        self.layer_norm_h = nn.LayerNorm(hidden_size)      # Normalize hidden state
        self.layer_norm_c = nn.LayerNorm(hidden_size)      # Normalize cell state

        self.init_parameters()

    def init_parameters(self) -> None:
        """
        Initialize parameters using best practices:
        1. Orthogonal initialization for better gradient flow
        2. Initialize forget gate bias to 1.0 to prevent forgetting at start of training
        """
        for weight in [self.weight_ih.weight, self.weight_hh.weight]:
            nn.init.orthogonal_(weight)
        
        # Set forget gate bias to 1.0 (helps with learning long sequences)
        nn.init.constant_(self.weight_ih.bias[self.hidden_size:2*self.hidden_size], 1.0)
        nn.init.constant_(self.weight_hh.bias[self.hidden_size:2*self.hidden_size], 1.0)

    def forward(self, x: torch.Tensor, 
                hidden_state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of LSTM cell.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            hidden_state: Tuple of (h_{t-1}, c_{t-1}) each of shape (batch_size, hidden_size)
        
        Returns:
            Tuple of (h_t, c_t) representing new hidden and cell states
        """
        h_prev, c_prev = hidden_state
        
        # Combined matrix multiplication for all gates
        # Shape: (batch_size, 4 * hidden_size)
        gates_x = self.weight_ih(x)          # Transform input
        gates_h = self.weight_hh(h_prev)     # Transform previous hidden state
        
        # Apply layer normalization
        gates_x = self.layer_norm_x(gates_x)
        gates = gates_x + gates_h  # Combined gate pre-activations
        
        # Split into individual gates
        # Each gate shape: (batch_size, hidden_size)
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)
        
        # Apply gate non-linearities
        i_t = torch.sigmoid(i_gate)  # Input gate
        f_t = torch.sigmoid(f_gate)  # Forget gate
        g_t = torch.tanh(g_gate)     # Cell state candidate
        o_t = torch.sigmoid(o_gate)  # Output gate
        
        # Update cell state: c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
        c_t = f_t * c_prev + i_t * g_t
        c_t = self.layer_norm_c(c_t)
        
        # Update hidden state: h_t = o_t ⊙ tanh(c_t)
        h_t = o_t * torch.tanh(c_t)
        h_t = self.layer_norm_h(h_t)
        
        if self.dropout is not None:
            h_t = self.dropout(h_t)
            
        return h_t, c_t

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state and cell state with zeros."""
        return (torch.zeros(batch_size, self.hidden_size, device=device),
                torch.zeros(batch_size, self.hidden_size, device=device))


class StackedLSTM(nn.Module):
    """
    Stacked LSTM implementation supporting multiple layers.
    Each layer processes the output of the previous layer.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Create list of LSTM cells, one for each layer
        self.layers = nn.ModuleList([
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size, 
                    dropout if i < num_layers - 1 else 0.0)  # No dropout on last layer
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, 
                hidden_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Process input sequence through stacked LSTM layers.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            hidden_states: Optional initial hidden states for each layer
        
        Returns:
            Tuple of (output, hidden_states) where output has shape (batch_size, seq_length, hidden_size)
        """
        batch_size, seq_length, _ = x.size()
        device = x.device
        
        if hidden_states is None:
            hidden_states = [layer.init_hidden(batch_size, device) for layer in self.layers]
        
        layer_outputs = []
        for t in range(seq_length):
            input_t = x[:, t, :]
            for i, lstm_cell in enumerate(self.layers):
                input_t, cell_state = lstm_cell(input_t, hidden_states[i])
                hidden_states[i] = (input_t, cell_state)
            layer_outputs.append(input_t)
            
        # Stack outputs along sequence dimension
        output = torch.stack(layer_outputs, dim=1)
        return output, hidden_states


class LSTMNetwork(nn.Module):
    """
    Complete LSTM network with bidirectional support.
    
    In bidirectional mode:
    - Forward LSTM processes sequence from left to right
    - Backward LSTM processes sequence from right to left
    - Outputs are concatenated for final prediction
    """
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_layers: int, 
                 output_size: int, 
                 dropout: float = 0.0,
                 bidirectional: bool = False):
        super().__init__()
        self.bidirectional = bidirectional
        
        # Forward direction LSTM
        self.stacked_lstm = StackedLSTM(input_size, hidden_size, num_layers, dropout)
        
        # Optional backward direction LSTM for bidirectional processing
        if bidirectional:
            self.reverse_lstm = StackedLSTM(input_size, hidden_size, num_layers, dropout)
            hidden_size *= 2  # Double hidden size due to concatenation
            
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, 
                hidden_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> torch.Tensor:
        """
        Forward pass of the network.
        
        For bidirectional processing:
        1. Process sequence normally with forward LSTM
        2. Process reversed sequence with backward LSTM
        3. Concatenate both outputs
        4. Apply final linear transformation
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            hidden_states: Optional initial hidden states
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Forward direction
        output, hidden_states = self.stacked_lstm(x, hidden_states)
        
        if self.bidirectional:
            # Process sequence in reverse direction
            reverse_output, _ = self.reverse_lstm(torch.flip(x, [1]))
            # Flip back to align with forward sequence
            reverse_output = torch.flip(reverse_output, [1])
            # Concatenate forward and backward outputs along feature dimension
            output = torch.cat([output, reverse_output], dim=-1)
        
        # Apply dropout before final layer
        output = self.dropout(output)
        # Use final timestep output for prediction
        final_output = self.fc(output[:, -1, :])
        return final_output


def create_lstm_model(config: dict) -> LSTMNetwork:
    """
    Factory function to create an LSTM model with specified configuration.
    
    Args:
        config: Dictionary containing model parameters:
            - input_size: Size of input features
            - hidden_size: Size of LSTM hidden state
            - num_layers: Number of stacked LSTM layers
            - output_size: Size of final output
            - dropout: Dropout probability (optional)
            - bidirectional: Whether to use bidirectional LSTM (optional)
    """
    return LSTMNetwork(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_size=config['output_size'],
        dropout=config.get('dropout', 0.0),
        bidirectional=config.get('bidirectional', False)
    )

# Example usage
if __name__ == "__main__":
    # Configuration for a bidirectional LSTM
    config = {
        'input_size': 3,
        'hidden_size': 64,
        'num_layers': 2,
        'output_size': 1,
        'dropout': 0.3,
        'bidirectional': True  # Enable bidirectional processing
    }
    
    # Create model
    model = create_lstm_model(config)
    
    # Generate dummy input
    batch_size, seq_length = 32, 10
    x = torch.randn(batch_size, seq_length, config['input_size'])
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")