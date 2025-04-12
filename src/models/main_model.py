# src/models/main_model.py

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Any

# Assuming these imports point to your file structure
from src.models.mga_layer import MGALayer
from src.models.baseline import TransformerLayer
from src.utils.helpers import PositionalEncoding

class TransformerModel(nn.Module):
    """
    A generic Transformer-based model wrapper that can use either standard
    Transformer layers or MGA layers. Includes embedding, positional encoding,
    a stack of layers, final normalization, and a classification head.

    Args:
        vocab_size (int): Size of the vocabulary.
        d_model (int): Dimension of the model (embeddings, hidden states).
        n_heads (int): Number of attention heads.
        d_ff (int): Dimension of the feed-forward network hidden layer.
        n_layers (int): Number of Transformer/MGA layers.
        dropout (float): Dropout probability.
        max_len (int): Maximum sequence length for positional encoding.
        padding_idx (int): Index of the padding token in the vocabulary.
        num_classes (int): Number of output classes for the classifier.
        use_mga (bool): If True, use MGALayer; otherwise, use TransformerLayer.
                        Defaults to False.
        pretrained_embeds (torch.Tensor, optional): Pre-trained embedding weights
                                                    to initialize the token embedding layer.
                                                    Defaults to None.
        freeze_embeddings (bool): If True and pretrained_embeds is provided,
                                  freeze the embedding layer weights. Defaults to False.
        mga_configs (Dict[str, Any], optional): Dictionary containing specific arguments
                                                for MGALayer if use_mga is True.
                                                Expected keys: "local_window_size", "num_latents",
                                                "gru_hidden", "gate_hidden", "activation",
                                                "use_stable_local_norm". Defaults to None.
    """
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 n_layers: int,
                 dropout: float,
                 max_len: int,
                 padding_idx: int,
                 num_classes: int,
                 use_mga: bool = False,
                 pretrained_embeds: Optional[torch.Tensor] = None,
                 freeze_embeddings: bool = False,
                 mga_configs: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.use_mga = use_mga
        self.d_model = d_model

        # --- Embedding Layers ---
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        if pretrained_embeds is not None:
            print("Loading pre-trained embeddings...")
            if pretrained_embeds.shape[0] != vocab_size or pretrained_embeds.shape[1] != d_model:
                 print(f"Warning: Pretrained embedding shape {pretrained_embeds.shape} "
                       f"does not match vocab_size {vocab_size} and d_model {d_model}. "
                       f"Ensure tokenizer and model dims are consistent.")
                 # Decide how to handle mismatch: error out or partial load?
                 # Simple copy might fail here. Let's assume shapes match for now.
            self.token_emb.weight.data.copy_(pretrained_embeds)
            if freeze_embeddings:
                print("Freezing embedding layer.")
                self.token_emb.weight.requires_grad = False
            else:
                 print("Fine-tuning embedding layer.")

        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.dropout_emb = nn.Dropout(dropout)
        # Scaling factor for embeddings (common practice)
        self.scale = math.sqrt(d_model)

        # --- Transformer/MGA Layers ---
        if use_mga:
            if mga_configs is None:
                raise ValueError("mga_configs dictionary must be provided when use_mga is True.")
            # Ensure all required MGA args are present
            required_mga_keys = {"local_window_size", "num_latents", "gru_hidden", "gate_hidden"}
            if not required_mga_keys.issubset(mga_configs.keys()):
                raise ValueError(f"mga_configs missing required keys: {required_mga_keys - mga_configs.keys()}")

            self.layers = nn.ModuleList([
                MGALayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout,
                         local_window_size=mga_configs["local_window_size"],
                         num_latents=mga_configs["num_latents"],
                         gru_hidden=mga_configs["gru_hidden"],
                         gate_hidden=mga_configs["gate_hidden"],
                         activation=mga_configs.get('activation', 'gelu'), # Default activation
                         use_stable_local_norm=mga_configs.get('use_stable_local_norm', False) # Default stability
                         )
                for _ in range(n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                TransformerLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
                for _ in range(n_layers)
            ])

        # --- Output Layers ---
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        print(f"Initialized TransformerModel: use_mga={use_mga}, n_layers={n_layers}, d_model={d_model}, n_heads={n_heads}")

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs. Shape: (batch_size, seq_len).
            attention_mask (torch.Tensor, optional): Mask indicating non-padding tokens (1)
                                                      and padding tokens (0).
                                                      Shape: (batch_size, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output logits from the classifier head. Shape: (batch_size, num_classes).
        """
        batch_size, seq_len = input_ids.shape

        # Create key_padding_mask (True where padded) from attention_mask (True where not padded)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        # Handle potential length mismatch with positional encoding
        if seq_len > self.pos_enc.embedding.num_embeddings:
             print(f"Warning: Input seq len {seq_len} > pos emb max {self.pos_enc.embedding.num_embeddings}. Truncating.")
             input_ids = input_ids[:, :self.pos_enc.embedding.num_embeddings]
             if key_padding_mask is not None:
                 key_padding_mask = key_padding_mask[:, :self.pos_enc.embedding.num_embeddings]
             seq_len = input_ids.size(1) # Update seq_len after truncation

        # --- Embeddings & Positional Encoding ---
        x_emb = self.token_emb(input_ids) * self.scale
        x_pos = self.pos_enc(input_ids) # Pass input_ids to get correct shape/device
        x = x_emb + x_pos
        x = self.dropout_emb(x)

        # --- Pass through Layers ---
        for layer in self.layers:
             x = layer(x, key_padding_mask=key_padding_mask)

        # --- Final Normalization and Classification ---
        x = self.final_norm(x)

        # Use representation of the first token ([CLS] or equivalent) for classification
        # Assumes the first token is used for sequence-level tasks.
        # For token classification, you would use the whole sequence `x`.
        cls_representation = x[:, 0, :] # Shape: (batch_size, d_model)

        logits = self.classifier(cls_representation) # Shape: (batch_size, num_classes)
        return logits

    def get_avg_gate_weights(self) -> Optional[torch.Tensor]:
        """
        Retrieves the average gate weights across layers from the last forward pass.
        Only applicable if use_mga is True. Averages across batch and sequence dimensions
        for each layer, then averages across layers.

        Returns:
            Optional[torch.Tensor]: A tensor of shape (3,) representing the average
                                    gate weights [Local, Global, State] across all layers
                                    and the last batch, or None if not using MGA or no
                                    weights were recorded. Returns zeros if called before
                                    forward pass.
        """
        if not self.use_mga:
            print("Warning: get_avg_gate_weights called on non-MGA model.")
            return None
        weights = []
        for layer in self.layers:
            # Check if it's an MGALayer and has the attribute populated
            if isinstance(layer, MGALayer) and hasattr(layer, 'last_gate_weights') and layer.last_gate_weights is not None:
                weights.append(layer.last_gate_weights) # Shape (3,) per layer

        if not weights:
             # Return zeros on the correct device if called before forward pass or if no weights found
             return torch.zeros(3, device=self.token_emb.weight.device)

        # Stack weights from all layers (num_layers, 3) and average across layers
        avg_weights = torch.stack(weights).mean(dim=0) # Shape (3,)
        return avg_weights

    def resize_token_embeddings(self, new_num_tokens: int):
        """
        Resizes the token embeddings layer. Useful if the tokenizer vocabulary
        was expanded (e.g., with special tokens) after model initialization.
        New embeddings are typically initialized randomly.

        Args:
            new_num_tokens (int): The new size of the vocabulary.
        """
        old_embeddings = self.token_emb
        new_embeddings = nn.Embedding(new_num_tokens, self.d_model, padding_idx=old_embeddings.padding_idx)

        # Initialize new embeddings (e.g., using default initialization)
        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)

        # Copy weights from the old embeddings where possible
        n = min(old_embeddings.num_embeddings, new_num_tokens)
        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        self.token_emb = new_embeddings
        print(f"Resized token embeddings from {old_embeddings.num_embeddings} to {new_num_tokens}")
