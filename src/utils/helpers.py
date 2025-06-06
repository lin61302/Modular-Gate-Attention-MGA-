# helpers.py
import torch
import torch.nn as nn
import math
from typing import Optional

class PositionalEncoding(nn.Module):
    """
    Learned Positional Encoding module.

    Args:
        d_model (int): The embedding dimension.
        max_len (int): The maximum sequence length for which to learn embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        if max_len <= 0:
            raise ValueError("max_len must be positive")
        self.embedding = nn.Embedding(max_len, d_model)
        # Initialize positional embeddings (optional, often helps)
        nn.init.uniform_(self.embedding.weight, -0.02, 0.02)
        print(f"Initialized PositionalEncoding with max_len={max_len}, d_model={d_model}")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Return positional embeddings for the input sequence length. The input is
        not modified; instead, a tensor containing the learned positional
        embeddings is returned for the caller to combine with the input as
        needed.

        Args:
            input_ids (torch.Tensor): Input token IDs to determine sequence length
                                      and device. Shape: (batch_size, seq_len).

        Returns:
            torch.Tensor: Positional embeddings for the sequence.
                          Shape: (batch_size, seq_len, d_model).
        """
        batch_size, seq_len = input_ids.shape
        if seq_len > self.embedding.num_embeddings:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds PositionalEncoding "
                f"max_len ({self.embedding.num_embeddings})"
            )

        # Create position IDs [0, 1, ..., seq_len-1]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0) # Shape: (1, seq_len)
        # Expand positions for the batch dimension if needed (though broadcasting works)
        # positions = positions.expand(batch_size, -1) # Shape: (batch_size, seq_len)

        # Return the learned embeddings for these positions
        # Broadcasting will automatically handle the batch dimension
        return self.embedding(positions) # Shape: (1, seq_len, d_model) -> broadcast to (batch_size, seq_len, d_model)

# You could add other helper functions here later (e.g., custom initializers, mask creators)
