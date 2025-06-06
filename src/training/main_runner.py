# main_runner.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import math
import random
import time
import gc
from typing import Dict, Any, Tuple, Optional

# Assuming imports from other project files
from src.models.main_model import TransformerModel
from src.training.train_loop import train_epoch, evaluate

# Import scheduler carefully
try:
    from transformers import get_linear_schedule_with_warmup
except ImportError:
    print("Warning: transformers library not found or import failed. LR scheduler will not be used.")
    get_linear_schedule_with_warmup = None

def run_experiment(model_type: str,
                   config: Dict[str, Any],
                   datasets: Tuple[DataLoader, DataLoader, DataLoader],
                   device: torch.device,
                   seed: int,
                   pretrained_embeddings: Optional[torch.Tensor] = None,
                   vocab_size: Optional[int] = None, # Pass vocab size if not in config
                   pad_idx: Optional[int] = None,    # Pass pad_idx if not in config
                   num_classes: int = 2,            # Default to binary classification
                   model_checkpoint_path: Optional[str] = None, # Path to load a model from
                   eval_only: bool = False          # Flag to only run evaluation
                   ) -> Dict[str, Any]:
    """
    Orchestrates a full experiment run (training and evaluation).

    Args:
        model_type: 'MGA' or 'Baseline'.
        config: Dictionary containing hyperparameters.
        datasets: Tuple of (train_loader, val_loader, test_loader).
        device: Device to run on.
        seed: Random seed.
        pretrained_embeddings: Optional tensor of pre-trained embeddings.
        vocab_size: Size of the vocabulary.
        pad_idx: Padding index.
        num_classes: Number of output classes.
        model_checkpoint_path: Path to load existing model weights (for eval_only).
        eval_only: If True, skip training and only evaluate.

    Returns:
        Dictionary containing results metrics.
    """
    print(f"\n--- Running {model_type} (Seed {seed}) ---")
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        # Optional: for full reproducibility, potentially at cost of speed
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    train_loader, val_loader, test_loader = datasets

    # --- Model Initialization ---
    is_mga = (model_type.upper() == "MGA")
    mga_configs = {
        k: config[k] for k in ["local_window_size", "num_latents", "gru_hidden", "gate_hidden"] if k in config
    }
    # Add optional MGA configs if present
    mga_configs['activation'] = config.get('activation', 'gelu')
    mga_configs['use_stable_local_norm'] = config.get('use_stable_local_norm', False)

    if not vocab_size: vocab_size = config.get('vocab_size')
    if not pad_idx: pad_idx = config.get('padding_idx')
    if not vocab_size or pad_idx is None : # Check again after trying config
         raise ValueError("vocab_size and padding_idx must be provided either directly or in config.")

    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        max_len=config.get("MODEL_MAX_LEN", config["MAX_SEQ_LEN"] + 10), # Use buffered length
        padding_idx=pad_idx,
        num_classes=num_classes,
        use_mga=is_mga,
        pretrained_embeds=pretrained_embeddings,
        freeze_embeddings=config.get('freeze_embeddings', False), # Get freeze flag from config
        mga_configs=mga_configs if is_mga else None
    ).to(device)

    # Load checkpoint if provided
    if model_checkpoint_path:
        try:
            print(f"Loading model checkpoint from: {model_checkpoint_path}")
            model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Proceeding without loading.")

    # --- Optimizer, Criterion, Scheduler ---
    # Use weight decay from config if specified, else default for AdamW might be 0 or 0.01
    optimizer = optim.AdamW(model.parameters(),
                            lr=config["lr"],
                            weight_decay=config.get("weight_decay", 0.0)) # Default to 0 if not in config
    criterion = nn.CrossEntropyLoss()
    scheduler = None
    if not eval_only and get_linear_schedule_with_warmup:
        accumulation_steps = config.get("gradient_accumulation_steps", 1)
        # Calculate total steps carefully, considering accumulation
        num_training_steps = math.ceil(len(train_loader) / accumulation_steps) * config["epochs"]
        num_warmup_steps = int(config.get("warmup_proportion", 0.06) * num_training_steps)
        print(f"Total effective training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    results = {
        "train_loss": [], "train_acc": [], "train_time": [],
        "val_loss": [], "val_acc": [], "val_f1": [], "val_time": [], "val_mem": [],
        "gate_weights": [], # Stores list of numpy arrays (shape (3,)) per epoch
        "test_loss": None, "test_acc": None, "test_f1": None,
        "test_time": None, "test_mem": None, "test_gate_weights": None, # Final avg gate weights
        "best_val_f1": 0.0 # Or best val accuracy depending on criterion
    }

    # --- Training & Validation Loop ---
    if not eval_only:
        print(f"Starting Training for {config['epochs']} epochs...")
        global current_epoch
        for epoch in range(1, config["epochs"] + 1):
            current_epoch = epoch
            if device.type == 'cuda': torch.cuda.empty_cache(); gc.collect()

            avg_train_loss, avg_train_acc, train_time = train_epoch(
                model, train_loader, optimizer, criterion, device, scheduler,
                config.get("gradient_accumulation_steps", 1), config.get("max_grad_norm", 1.0),
                epoch_num=epoch, total_epochs=config['epochs']
            )

            if device.type == 'cuda': torch.cuda.empty_cache(); gc.collect()
            val_loss, val_acc, val_f1, val_time, val_mem, gate_w = evaluate(
                model, val_loader, criterion, device, profile=True
            )

            # Store results
            results["train_loss"].append(avg_train_loss)
            results["train_acc"].append(avg_train_acc)
            results["train_time"].append(train_time)
            results["val_loss"].append(val_loss)
            results["val_acc"].append(val_acc)
            results["val_f1"].append(val_f1)
            results["val_time"].append(val_time)
            results["val_mem"].append(val_mem)
            if gate_w is not None:
                results["gate_weights"].append(gate_w.cpu().numpy()) # Store numpy array

            # Update best validation metric (using F1 here, could use accuracy)
            if val_f1 > results["best_val_f1"]:
                results["best_val_f1"] = val_f1
                # Optional: Save best model checkpoint
                # torch.save(model.state_dict(), f"best_model_{model_type}_seed{seed}.pt")
                # print(f"  New best validation F1: {val_f1:.4f}. Model saved.")

            # Log epoch results
            print(f"  Epoch {epoch:02d}/{config['epochs']} | Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.3f} | Val Acc: {val_acc:.3f}, F1: {val_f1:.3f}")
            print(f"    Time (Train/Val Inf): {train_time:.1f}s / {val_time*1000:.1f}ms | Peak Mem Inc: {val_mem:.1f}MB")
            if model_type.upper() == "MGA" and gate_w is not None and gate_w.numel() == 3:
                print(f"    Avg Gates (L/G/S): {gate_w[0]:.3f} / {gate_w[1]:.3f} / {gate_w[2]:.3f}")

    # --- Final Test Evaluation ---
    print(f"--- Final Test Evaluation {model_type} (Seed {seed}) ---")
    if device.type == 'cuda': torch.cuda.empty_cache(); gc.collect()
    # If only evaluating, model checkpoint should have been loaded earlier
    if eval_only and not model_checkpoint_path:
        print("Warning: Evaluating without training and no checkpoint loaded.")

    test_loss, test_acc, test_f1, test_time, test_mem, test_gate_w = evaluate(
        model, test_loader, criterion, device, profile=True
    )
    print(f"  Test Loss: {test_loss:.4f}, Acc: {test_acc:.3f}, F1: {test_f1:.3f}")
    print(f"    Test Inf Time: {test_time*1000:.1f}ms | Peak Mem Inc: {test_mem:.1f}MB")
    if model_type.upper() == "MGA" and test_gate_w is not None and test_gate_w.numel() == 3:
        print(f"    Avg Test Gates (L/G/S): {test_gate_w[0]:.3f} / {test_gate_w[1]:.3f} / {test_gate_w[2]:.3f}")

    # Store final test results
    results["test_loss"] = test_loss
    results["test_acc"] = test_acc
    results["test_f1"] = test_f1
    results["test_time"] = test_time
    results["test_mem"] = test_mem
    results["test_gate_weights"] = test_gate_w.cpu().numpy() if test_gate_w is not None else None

    # Clean up
    del model, optimizer, criterion, scheduler
    if device.type == 'cuda': torch.cuda.empty_cache(); gc.collect()

    return results

# Note: This file provides the core function. An executable script (like main.py)
# would handle argument parsing, loading configs/datasets, and calling this function.
