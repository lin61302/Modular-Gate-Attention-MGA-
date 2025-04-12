# baseline.py
import torch
import torch.nn as nn
from typing import Optional, Callable

class TransformerLayer(nn.Module):
    """
    A standard Transformer Encoder Layer implementation (Pre-LN style).

    Args:
        d_model (int): Input and output feature dimension.
        n_heads (int): Number of attention heads.
        d_ff (int): Hidden dimension of the feed-forward network.
        dropout (float): Dropout probability.
        activation (Callable): Activation function for FFN (e.g., nn.ReLU or nn.GELU).
                               Defaults to nn.GELU.
    """
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float,
                 activation: Callable = nn.GELU):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Layer Normalizations (Pre-LN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Multi-Head Self-Attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            activation(),
            nn.Dropout(dropout), # Commonly placed here
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the Transformer layer (Pre-LN).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            key_padding_mask (torch.Tensor, optional): Mask for padding tokens (True where padded).
                                                        Shape (batch_size, seq_len). Defaults to None.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # --- Self-Attention Block ---
        residual = x
        x_norm = self.norm1(x)
        # Query, Key, Value are the same for self-attention
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm,
                                        key_padding_mask=key_padding_mask,
                                        need_weights=False) # Don't need attention weights output
        x = residual + self.dropout1(attn_output) # Add & Drop

        # --- Feed-Forward Block ---
        residual = x
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = residual + self.dropout2(ffn_output) # Add & Drop

        return x
