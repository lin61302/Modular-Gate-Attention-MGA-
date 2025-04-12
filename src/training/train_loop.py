import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm.auto import tqdm
import gc
from sklearn.metrics import accuracy_score, f1_score
from typing import Tuple, Optional, Dict, Any

# Assume TransformerModel class is defined elsewhere and has get_avg_gate_weights method if MGA
# from src.models.main_model import TransformerModel # Example import

# Define global epoch variable for tqdm description if needed outside the function
current_epoch = 0

def train_epoch(model: nn.Module,
                loader: DataLoader,
                optimizer: optim.Optimizer,
                criterion: nn.Module,
                device: torch.device,
                scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], # Correct type hint
                accumulation_steps: int = 1,
                max_grad_norm: float = 1.0,
                epoch_num: int = 0, # Pass epoch number for logging
                total_epochs: int = 1 # Pass total epochs for logging
                ) -> Tuple[float, float, float]:
    """
    Performs a single training epoch.

    Args:
        model: The model to train.
        loader: DataLoader for the training set.
        optimizer: The optimizer.
        criterion: The loss function.
        device: The device to train on (cuda or cpu).
        scheduler: Optional learning rate scheduler.
        accumulation_steps: Number of steps to accumulate gradients over.
        max_grad_norm: Maximum norm for gradient clipping.
        epoch_num: Current epoch number (for logging).
        total_epochs: Total number of epochs (for logging).

    Returns:
        Tuple containing average loss, average accuracy, and epoch duration.
    """
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    start_time = time.time()
    optimizer.zero_grad() # Zero gradients once before the loop

    train_pbar = tqdm(loader, desc=f"Epoch {epoch_num}/{total_epochs} Training", leave=False)

    for i, batch in enumerate(train_pbar):
        # Move batch data to device - check if batch is a dict (Hugging Face) or tuple
        if isinstance(batch, dict):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
        elif isinstance(batch, (list, tuple)):
            input_ids, labels, attention_mask = batch # Assuming (ids, labels, mask) order
            input_ids, labels, attention_mask = input_ids.to(device), labels.to(device), attention_mask.to(device)
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        # Forward pass
        # Note: Assuming model's forward expects input_ids and optionally attention_mask
        # Key padding mask is handled internally by the model if needed
        logits = model(input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        # Normalize loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()

        # Accumulate metrics
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size * accumulation_steps # De-normalize loss for logging
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

        # Optimizer step
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True) # More memory efficient

        # Update progress bar
        train_pbar.set_postfix({'Loss': total_loss / total_samples if total_samples else 0,
                              'Acc': total_correct / total_samples if total_samples else 0})

    epoch_time = time.time() - start_time
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0
    return avg_loss, avg_acc, epoch_time

@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device,
             profile: bool = False
             ) -> Tuple[float, float, float, float, float, Optional[torch.Tensor]]:
    """
    Evaluates the model on a given dataset.

    Args:
        model: The model to evaluate.
        loader: DataLoader for the evaluation set.
        criterion: The loss function.
        device: The device to evaluate on.
        profile: Whether to profile inference time and memory.

    Returns:
        Tuple containing average loss, accuracy, macro F1 score,
        average inference time per batch, peak GPU memory increase (MB),
        and average gate weights (or None).
    """
    model.eval()
    total_loss, total_samples = 0.0, 0
    gate_weights_list = []
    inference_times = []
    all_preds = []
    all_labels = []
    start_mem = torch.cuda.memory_allocated(device) if profile and device.type == 'cuda' else 0
    max_mem = start_mem

    eval_pbar = tqdm(loader, desc="Evaluating", leave=False)
    for batch in eval_pbar:
        # Move batch data to device
        if isinstance(batch, dict):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
        elif isinstance(batch, (list, tuple)):
            input_ids, labels, attention_mask = batch
            input_ids, labels, attention_mask = input_ids.to(device), labels.to(device), attention_mask.to(device)
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        batch_size = labels.size(0)
        inf_start = time.time()
        logits = model(input_ids, attention_mask=attention_mask)
        inference_times.append(time.time() - inf_start)

        loss = criterion(logits, labels)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        total_samples += batch_size

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if profile and device.type == 'cuda':
             max_mem = max(max_mem, torch.cuda.memory_allocated(device))

        # Collect gate weights if MGA model (check attribute existence)
        if hasattr(model, 'use_mga') and model.use_mga and hasattr(model, 'get_avg_gate_weights'):
             avg_layer_weights = model.get_avg_gate_weights() # Should return tensor of shape (3,)
             if avg_layer_weights is not None:
                 gate_weights_list.append(avg_layer_weights.cpu()) # Store tensor

    # Calculate final metrics
    avg_inf_time = np.mean(inference_times) if inference_times else 0
    peak_mem_increase = (max_mem - start_mem) / (1024**2) if profile and device.type == 'cuda' else 0
    avg_gate_weights = torch.stack(gate_weights_list).mean(dim=0) if gate_weights_list else torch.zeros(3) # Avg across batches/layers
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0
    # Use macro F1 for potentially imbalanced classes, binary otherwise is fine too
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) if all_labels else 0

    # Make sure gate weights are returned correctly
    final_gates = avg_gate_weights if gate_weights_list else None

    return avg_loss, accuracy, f1, avg_inf_time, peak_mem_increase, final_gates
