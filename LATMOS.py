import math
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import random 
from mamba_ssm import Mamba, Mamba2

class BaseSeqModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BaseSeqModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Input embedding MLP
        self.seq_input_proj = nn.Linear(input_size, hidden_size)
        
        # Initialize encoder (to be implemented by child classes)
        self.encoder = None
        
        # Decoder MLPs - shared across all implementations
        self.decoder_state = nn.Sequential(
            nn.Linear(self.hidden_size, (self.hidden_size+self.output_size)//2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear((self.hidden_size+self.output_size)//2, self.output_size),
            # nn.Softmax(dim=-1)
        )

        self.decoder_acceptance = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size // 2, 2),
            # nn.Softmax(dim=-1)
        )
    
    def forward(self, x, hidden=None):
        '''
        @return
        state_output (batch_size, seq_length, num_states)
        acceptance_output (batch_size, seq_length, 1)
        '''
        # Embed the input sequence
        x = self.seq_input_proj(x)
        
        # Encode the input sequence
        hidden = self.seq_to_seq(x, hidden)
        
        # Decode using both decoders
        state_output = self.decoder_state(hidden)
        acceptance_output = self.decoder_acceptance(hidden)
        
        return state_output, acceptance_output
    
    def seq_to_seq(self, x, hidden=None):
        raise NotImplementedError("Encode method must be implemented by child classes")

    def get_model_size(self):
        """Returns the total number of parameters in the model in a human-readable format."""
        params = sum(p.numel() for p in self.parameters())
        if params >= 1e9:
            return f"{params/1e9:.2f}B"
        elif params >= 1e6:
            return f"{params/1e6:.2f}M"
        elif params >= 1e3:
            return f"{params/1e3:.2f}K"
        return str(params)
    
class GRUModel(BaseSeqModel):
    def __init__(self, input_size, hidden_size, output_size, num_layers=4):
        super(GRUModel, self).__init__(input_size, hidden_size, output_size)
        self.seq = nn.GRU(
            hidden_size,  # Use hidden size after embedding
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
    
    def seq_to_seq(self, x, hidden=None):
        output, _ = self.seq(x, hidden)
        return output

class AttentionModel(BaseSeqModel):
    def __init__(self, input_size, hidden_size, output_size, num_layers=4, nhead=4, max_seq_length=1000):
        super(AttentionModel, self).__init__(input_size, hidden_size, output_size)
        # Create sinusoidal positional encoding
        pe = torch.zeros(1, max_seq_length, hidden_size)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoder', pe)
        
        self.seq = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=nhead,
                dim_feedforward=hidden_size,
                batch_first=True
            ), 
            num_layers=num_layers
        )
    
    def seq_to_seq(self, x, hidden=None):        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        output = self.seq(x, mask=mask, is_causal=True)
        # output = self.encoder(x)
        return output

class MambaModel(BaseSeqModel):
    def __init__(self, input_size, hidden_size, output_size, num_layers=4):
        super(MambaModel, self).__init__(input_size, hidden_size, output_size)
        self.seq = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=hidden_size, # Model dimension d_model
            d_state=128,  # SSM state expansion factor, typically 64 or 128
            d_conv=8,    # Local convolution width
            expand=4,    # Block expansion factor
        )
        self.norm = nn.LayerNorm(hidden_size)
    
    def seq_to_seq(self, x, hidden=None):
        return self.seq(x)

def create_model(model_type, input_size, hidden_size, output_size, device):
    model_classes = {
        'gru': GRUModel,
        'attention': AttentionModel,
        'ssm': MambaModel 
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(model_classes.keys())}")
    
    model = model_classes[model_type](input_size, hidden_size, output_size)
    return model.to(device)