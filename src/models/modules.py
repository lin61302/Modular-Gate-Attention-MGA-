# src/models/modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class LocalAttention(nn.Module):
    """
    Local Attention Path Module.

    Can operate either as standard multi-head self-attention (if window_size <= 0)
    or as windowed multi-head self-attention (if window_size > 0).

    Args:
        d_model (int): Input and output feature dimension.
        n_heads (int): Number of attention heads.
        window_size (int): The radius of the attention window (w). Attention is computed
                           within [i-w, i+w]. If <= 0, performs full self-attention.
        dropout (float): Dropout probability.
        use_stable_local_norm (bool): If True and window_size > 0, applies an additional
                                      LayerNorm before the windowed attention calculation
                                      as a potential stability enhancement. Defaults to False.
    """
    def __init__(self, d_model: int, n_heads: int, window_size: int, dropout: float, use_stable_local_norm: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size if window_size is not None else 0 # Treat None as 0
        self.dropout = dropout
        self.use_stable_local_norm = use_stable_local_norm

        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        # Optional LayerNorm specifically for local path stability enhancement
        if self.use_stable_local_norm and self.window_size > 0:
            self.norm_local = nn.LayerNorm(d_model)
        else:
            self.norm_local = nn.Identity() # No-op if not used

        # Potential area for specific initialization or regularization for windowed attention weights,
        # although standard initialization and optimizer weight decay often suffice.

    def _create_window_mask(self, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        """Creates the attention mask for windowed attention."""
        if self.window_size <= 0:
            return None # No mask for full attention

        idx = torch.arange(seq_len, device=device)
        # Mask is True for positions *outside* the window [i-w, i+w]
        mask = ~((idx[None, :] >= idx[:, None] - self.window_size) &
                   (idx[None, :] <= idx[:, None] + self.window_size))
        # Shape: (seq_len, seq_len). MHA will broadcast this.
        return mask

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            key_padding_mask (torch.Tensor, optional): Mask for padding tokens.
                                                        Shape (batch_size, seq_len). Defaults to None.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        batch_size, seq_len, _ = x.shape

        # Apply optional internal norm for stability
        x_norm = self.norm_local(x)

        # Create window mask if needed
        attn_mask = self._create_window_mask(seq_len, x.device)

        # nn.MultiheadAttention expects attn_mask where True indicates masking.
        # key_padding_mask indicates padding (True = pad).
        # MHA automatically combines these.
        attn_output, _ = self.attn(x_norm, x_norm, x_norm,
                                   key_padding_mask=key_padding_mask,
                                   attn_mask=attn_mask,
                                   need_weights=False) # Don't need weights output for fusion
        return attn_output

class GlobalLatentAttention(nn.Module):
    """
    Global Latent Bottleneck Attention Module.

    Aggregates sequence information via learned latent vectors.

    Args:
        d_model (int): Input and output feature dimension.
        n_heads (int): Number of attention heads.
        num_latents (int): Number of latent vectors (L).
        dropout (float): Dropout probability.
    """
    def __init__(self, d_model: int, n_heads: int, num_latents: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.num_latents = num_latents
        self.d_model = d_model
        self.n_heads = n_heads

        # Learnable latent vectors
        self.latents = nn.Parameter(torch.empty(1, num_latents, d_model))
        nn.init.xavier_uniform_(self.latents) # Standard initialization

        # Attention layers for the two stages
        self.attn_t2l = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_l2t = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Normalization for updated latents
        self.norm_latent = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model) # Optional norm on final output

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            key_padding_mask (torch.Tensor, optional): Mask for padding tokens in x.
                                                        Shape (batch_size, seq_len). Defaults to None.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        batch_size = x.size(0)
        latents = self.latents.expand(batch_size, -1, -1) # Shape: (batch_size, L, d_model)

        # 1. Token-to-Latent Aggregation
        # Latents attend to tokens (Latents are queries, Tokens are keys/values)
        updated_latents_prime, _ = self.attn_t2l(query=latents, key=x, value=x,
                                                 key_padding_mask=key_padding_mask,
                                                 need_weights=False)
        # Add residual and normalize
        updated_latents = self.norm_latent(latents + updated_latents_prime) # Shape: (batch_size, L, d_model)

        # 2. Latent-to-Token Broadcast
        # Tokens attend to updated latents (Tokens are queries, Latents are keys/values)
        # No key_padding_mask needed here as latents are not padded.
        global_context_output, _ = self.attn_l2t(query=x, key=updated_latents, value=updated_latents,
                                                  key_padding_mask=None, # Latents are never padded
                                                  need_weights=False)
        # Optionally normalize the final output of this path
        # global_context_output = self.norm_out(global_context_output)
        return global_context_output

class GRUStateModule(nn.Module):
    """
    Recurrent State Path Module using GRU.

    Args:
        d_model (int): Input and output feature dimension.
        gru_hidden (int): Hidden dimension of the GRU. Typically d_model.
        dropout (float): Dropout probability applied between GRU layers if num_layers > 1.
                         Note: MGA uses single layer GRU, so dropout here might be ineffective
                         unless explicitly configured differently or applied externally.
        num_gru_layers (int): Number of GRU layers. Defaults to 1.
    """
    def __init__(self, d_model: int, gru_hidden: int, dropout: float, num_gru_layers: int = 1):
        super().__init__()
        self.d_model = d_model
        self.gru_hidden = gru_hidden
        # Apply dropout only if more than one layer, standard GRU behavior
        gru_dropout_rate = dropout if num_gru_layers > 1 else 0
        self.gru = nn.GRU(d_model, gru_hidden, batch_first=True, num_layers=num_gru_layers, dropout=gru_dropout_rate)

        # Add projection if GRU hidden size differs from d_model
        self.out_proj = nn.Linear(gru_hidden, d_model) if gru_hidden != d_model else nn.Identity()

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            key_padding_mask (torch.Tensor, optional): Not directly used by standard nn.GRU
                                                        unless input is packed. Padding affects hidden state
                                                        but output masking should happen later.
        Returns:
            torch.Tensor: Output sequence of hidden states, shape (batch_size, seq_len, d_model).
        """
        # GRU processes the full sequence including padding.
        # The final hidden state is implicitly handled by the module.
        # We only need the output sequence (hidden state at each timestep).
        gru_out_seq, _ = self.gru(x) # gru_out shape: (batch, seq_len, gru_hidden)

        # Project back to d_model if necessary
        projected_out = self.out_proj(gru_out_seq) # Shape: (batch, seq_len, d_model)
        return projected_out

class GatingNetwork(nn.Module):
    """
    Token-wise Gating Network (2-layer MLP with Softmax).

    Args:
        d_model (int): Input feature dimension (dimension of token representations).
        gate_hidden (int): Hidden dimension of the MLP.
        num_outputs (int): Number of paths to gate (typically 3 for MGA).
        activation (str): Activation function for the hidden layer ('relu' or 'gelu').
                          Defaults to 'relu'.
    """
    def __init__(self, d_model: int, gate_hidden: int, num_outputs: int, activation: str = 'relu'):
        super().__init__()
        if activation.lower() == 'gelu':
            act_fn = nn.GELU()
        elif activation.lower() == 'relu':
            act_fn = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function for GatingNetwork: {activation}")

        self.net = nn.Sequential(
            nn.Linear(d_model, gate_hidden),
            act_fn,
            nn.Linear(gate_hidden, num_outputs)
        )
        # Consider standard initialization if defaults are not sufficient

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (token representations) of shape (batch, seq, d_model).

        Returns:
            torch.Tensor: Gate weights of shape (batch, seq, num_outputs), summing to 1 across last dim.
        """
        logits = self.net(x)
        gate_weights = F.softmax(logits, dim=-1)
        return gate_weights

# --- MGA Layer (Integrator) ---

class MGALayer(nn.Module):
    """
    A single layer of Modular Gated Attention (Pre-LN style).

    Combines outputs from Local Attention, Global Latent Attention, and GRU State modules
    using a learned gating mechanism.

    Args:
        d_model (int): Model dimension.
        n_heads (int): Number of attention heads (for Local and Global modules).
        d_ff (int): Dimension of the feed-forward network hidden layer.
        dropout (float): Dropout probability.
        local_window_size (int): Window size for the LocalAttention module. <= 0 means full attention.
        num_latents (int): Number of latent vectors (L) for the GlobalLatentAttention module.
        gru_hidden (int): Hidden dimension for the GRUStateModule (usually d_model).
        gate_hidden (int): Hidden dimension for the GatingNetwork MLP.
        activation (str): Activation function ('relu' or 'gelu') for FFN and Gating MLP. Defaults to 'gelu'.
        use_stable_local_norm (bool): Whether to use extra norm in LocalAttention. Defaults to False.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float,
                 local_window_size: int, num_latents: int, gru_hidden: int, gate_hidden: int,
                 activation: str = 'gelu', use_stable_local_norm: bool = False):
        super().__init__()
        self.num_modules = 3

        # Instantiate Modules
        self.local_mod = LocalAttention(d_model, n_heads, local_window_size, dropout, use_stable_local_norm)
        self.global_mod = GlobalLatentAttention(d_model, n_heads, num_latents, dropout)
        self.state_mod = GRUStateModule(d_model, gru_hidden, dropout, num_gru_layers=1) # Single layer GRU
        self.gate = GatingNetwork(d_model, gate_hidden, self.num_modules, activation=activation)

        # Layer Norms (Pre-LN style)
        self.norm_input = nn.LayerNorm(d_model) # Applied before modules/gating
        self.norm_ffn = nn.LayerNorm(d_model)   # Applied before FFN

        # Dropouts
        self.dropout_fusion = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        # Feed-Forward Network
        if activation.lower() == 'gelu':
            ffn_act_fn = nn.GELU()
        elif activation.lower() == 'relu':
            ffn_act_fn = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function for FFN: {activation}")

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            ffn_act_fn,
            nn.Dropout(dropout), # Dropout often applied after activation in FFN
            nn.Linear(d_ff, d_model)
        )

        # For analysis
        self.last_gate_weights = None

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            key_padding_mask (torch.Tensor, optional): Mask for padding tokens.
                                                        Shape (batch_size, seq_len). Defaults to None.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        residual_main = x

        # --- Pre-Normalization ---
        x_norm = self.norm_input(x)

        # --- Parallel Module Computations ---
        # Pass padding mask to attention modules
        local_out = self.local_mod(x_norm, key_padding_mask)
        global_out = self.global_mod(x_norm, key_padding_mask)
        state_out = self.state_mod(x_norm) # GRU processes full sequence here

        # --- Gating & Fusion ---
        # Gating depends on the normalized input for stability
        gate_weights = self.gate(x_norm) # Shape: (batch, seq, 3)
        self.last_gate_weights = gate_weights.detach().mean(dim=(0, 1)) # Store average for analysis

        fused_out = (gate_weights[..., 0:1] * local_out +
                     gate_weights[..., 1:2] * global_out +
                     gate_weights[..., 2:3] * state_out)

        # --- First Residual Connection ---
        x = residual_main + self.dropout_fusion(fused_out)

        # --- FFN Block (with Pre-LN) ---
        residual_ffn = x
        x_norm_ffn = self.norm_ffn(x)
        ffn_out = self.ffn(x_norm_ffn)

        # --- Second Residual Connection ---
        x = residual_ffn + self.dropout_ffn(ffn_out)

        return x
