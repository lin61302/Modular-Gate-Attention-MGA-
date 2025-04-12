# dataloader_selective_copy.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from typing import List, Tuple, Dict, Any

# --- Task Specific Constants ---
PAD = "[PAD]"
CLS = "[CLS]"
UNK = "[UNK]"
DISTRACTORS = ['D1', 'D2']
PAYLOAD_TOKENS = ['P0', 'P1']
MARKER_A, MARKER_B = 'A', 'B'
VOCAB = [PAD, CLS, UNK] + DISTRACTORS + [MARKER_A, MARKER_B] + PAYLOAD_TOKENS
vocab_map = {t: i for i, t in enumerate(VOCAB)}
vocab_size = len(VOCAB)
pad_idx = vocab_map[PAD]
PAYLOAD_LEN = 2 # As per notebook config

# --- Data Generation ---
def generate_copy_sequence(seq_length: int, payload_len: int) -> Tuple[List[str], int]:
    """Generates a sequence for the simplified selective copy task."""
    # Ensure enough space for markers, payloads, and at least one distractor between
    if seq_length < (payload_len * 2 + 4):
        raise ValueError(f"Sequence length {seq_length} too short for payload {payload_len}")

    # Place A near start, B near middle/end, ensure separation
    max_pos_A = max(1, seq_length // 3) # Ensure A is not too far
    pos_A = random.randint(1, max_pos_A)

    min_B_start = pos_A + payload_len + 1 # Need at least one distractor after payload1
    max_B_start = seq_length - payload_len - 2 # Need marker B and payload2 space
    if min_B_start > max_B_start:
        # This can happen if seq_length is small relative to payload_len
        # Adjust placement logic if needed, or retry generation
         pos_A = 1 # Force A early
         min_B_start = pos_A + payload_len + 1
         max_B_start = seq_length - payload_len - 2
         if min_B_start > max_B_start: # Still impossible, raise error
              raise ValueError("Cannot place markers and payloads within sequence length.")

    pos_B = random.randint(min_B_start, max_B_start)

    payload1 = [random.choice(PAYLOAD_TOKENS) for _ in range(payload_len)]
    label = random.choice([0, 1])
    payload2 = payload1[:] if label == 1 else [random.choice(PAYLOAD_TOKENS) for _ in range(payload_len)]
    while label == 0 and payload2 == payload1: # Ensure difference for label 0
        payload2 = [random.choice(PAYLOAD_TOKENS) for _ in range(payload_len)]

    # Initialize with distractors
    seq = [random.choice(DISTRACTORS) for _ in range(seq_length)]

    # Place markers and payloads
    seq[pos_A] = MARKER_A
    seq[pos_A+1 : pos_A+1+payload_len] = payload1
    seq[pos_B] = MARKER_B
    seq[pos_B+1 : pos_B+1+payload_len] = payload2

    # Ensure final sequence is exactly seq_length (it should be by design)
    return seq[:seq_length], label

# --- Dataset Class ---
class CopyDataset(Dataset):
    def __init__(self, num_samples: int, seq_length: int, payload_len: int):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.payload_len = payload_len
        # Generate on the fly in __getitem__

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[List[str], int]:
        # Generate sequence and label on the fly
        seq, label = generate_copy_sequence(self.seq_length, self.payload_len)
        return seq, label

# --- Collate Function ---
def collate_fn(batch: List[Tuple[List[str], int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collates batch, adds CLS, tokenizes, pads, creates mask."""
    sequences, labels = zip(*batch)
    # Note: seq_length should be fixed within a dataset split (train/val/test)
    max_len = max(len(s) for s in sequences) # Should be constant == self.seq_length
    batch_size = len(batch)

    input_ids = torch.full((batch_size, max_len + 1), pad_idx, dtype=torch.long) # +1 for CLS
    # key_padding_mask: True where padded, False otherwise.
    key_padding_mask = torch.ones((batch_size, max_len + 1), dtype=torch.bool)

    for i, seq in enumerate(sequences):
        len_s = len(seq)
        token_ids = [vocab_map.get(t, vocab_map[UNK]) for t in seq]

        input_ids[i, 0] = vocab_map[CLS]
        input_ids[i, 1:len_s+1] = torch.tensor(token_ids)
        key_padding_mask[i, :len_s+1] = False # Mark non-padded tokens as False

    return input_ids, torch.tensor(labels, dtype=torch.long), key_padding_mask

# --- Main Loader Function ---
def get_selective_copy_dataloaders(
    config: Dict[str, Any],
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """
    Creates DataLoaders for the Selective Copy task.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing keys like
                                 'batch_size', 'n_train_samples', 'n_eval_samples',
                                 'seq_len_train', 'seq_len_eval_id', 'seq_len_eval_ood'.
        num_workers (int): Number of workers for DataLoader.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, int, int]:
            Train, Validation (ID), Test (OOD) DataLoaders, vocab_size, pad_idx.
    """
    print("Generating Selective Copy datasets...")
    bs = config['batch_size']
    train_len = config['seq_len_train']
    eval_id_len = config['seq_len_eval_id']
    eval_ood_len = config['seq_len_eval_ood']

    train_d = CopyDataset(config['n_train_samples'], train_len, PAYLOAD_LEN)
    eval_id_d = CopyDataset(config['n_eval_samples'], eval_id_len, PAYLOAD_LEN)
    eval_ood_d = CopyDataset(config['n_eval_samples'], eval_ood_len, PAYLOAD_LEN)

    train_loader = DataLoader(train_d, batch_size=bs, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    # Use eval_id_len for validation during training
    val_loader = DataLoader(eval_id_d, batch_size=bs, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    # Use eval_ood_len for final testing
    test_loader = DataLoader(eval_ood_d, batch_size=bs // 2 if bs > 1 else 1, shuffle=False, collate_fn=collate_fn, num_workers=num_workers) # Smaller batch for OOD if longer

    return train_loader, val_loader, test_loader, vocab_size, pad_idx
