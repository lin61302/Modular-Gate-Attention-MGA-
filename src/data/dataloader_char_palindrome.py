# dataloader_char_palindrome.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from typing import List, Tuple, Dict, Any

# --- Task Specific Constants ---
PAD = "[PAD]"
CLS = "[CLS]"
UNK = "[UNK]"
DISTRACTORS = ['D1', 'D2', 'D3']
MARKERS = ['A', 'B']
PAYLOAD_ALPHABET = ['P0', 'P1', 'P2']
VOCAB = [PAD, CLS, UNK] + DISTRACTORS + MARKERS + PAYLOAD_ALPHABET
vocab_map = {t: i for i, t in enumerate(VOCAB)}
vocab_size = len(VOCAB)
pad_idx = vocab_map[PAD]

# --- Data Generation ---
def generate_palindrome_sequence(
    max_seq_len: int,
    min_payload: int,
    max_payload: int,
    max_init_dist: int,
    max_trailing_dist: int
) -> Tuple[List[str], int]:
    """Generates a sequence for the character palindrome task."""
    payload_len = random.randint(min_payload, max_payload)
    label = random.choice([0, 1]) # 1 = Palindrome, 0 = Not

    # Create payload
    if label == 1:
        half_len = math.ceil(payload_len / 2)
        first_half = [random.choice(PAYLOAD_ALPHABET) for _ in range(half_len)]
        # Create second half by reversing first half (handle odd/even length)
        second_half = first_half[:payload_len // 2][::-1]
        payload = first_half + second_half
    else:
        payload = [random.choice(PAYLOAD_ALPHABET) for _ in range(payload_len)]
        while payload == payload[::-1]: # Ensure it's not accidentally a palindrome
             payload = [random.choice(PAYLOAD_ALPHABET) for _ in range(payload_len)]

    num_initial_dist = random.randint(0, max_init_dist)
    num_trailing_dist = random.randint(0, max_trail_dist)

    # Ensure total length doesn't exceed max_seq_len - 1 (for CLS)
    total_non_distractor = 1 + payload_len + 1 # MarkerA + payload + MarkerB
    max_total_dist = max_seq_len - 1 - total_non_distractor
    if max_total_dist < 0:
         # Should not happen with reasonable config, but handle gracefully
         # Option 1: Raise error if config is impossible
         raise ValueError(f"max_seq_len {max_seq_len} too small for markers and min_payload {min_payload}")
         # Option 2: Adjust payload length dynamically (more complex)
         # Option 3: Truncate payload (might break task logic) - chosen here for simplicity if needed
         # payload = payload[:max_seq_len - 1 - 2] # Max payload possible
         # payload_len = len(payload)
         # max_total_dist = 0

    if num_initial_dist + num_trailing_dist > max_total_dist:
        total_dist_generated = num_initial_dist + num_trailing_dist
        num_initial_dist = math.floor(num_initial_dist * max_total_dist / total_dist_generated) if total_dist_generated > 0 else 0
        num_trailing_dist = max_total_dist - num_initial_dist

    initial_dist = [random.choice(DISTRACTORS) for _ in range(num_initial_dist)]
    trailing_dist = [random.choice(DISTRACTORS) for _ in range(num_trailing_dist)]

    # Construct sequence (excluding CLS, added by collator)
    sequence_tokens = initial_dist + [MARKERS[0]] + payload + [MARKERS[1]] + trailing_dist

    # Pad with distractors to reach exact length (max_seq_len - 1)
    current_len = len(sequence_tokens)
    target_len = max_seq_len - 1
    if current_len < target_len:
        sequence_tokens += [random.choice(DISTRACTORS) for _ in range(target_len - current_len)]
    elif current_len > target_len: # Should ideally not happen with above logic
        sequence_tokens = sequence_tokens[:target_len]

    return sequence_tokens, label

# --- Dataset Class ---
class PalindromeDataset(Dataset):
     def __init__(self, num_samples: int, config: Dict[str, Any]):
        self.num_samples = num_samples
        self.config = config
        # Generate on the fly

     def __len__(self) -> int:
        return self.num_samples

     def __getitem__(self, idx: int) -> Tuple[List[str], int]:
         seq, label = generate_palindrome_sequence(
             self.config['MAX_SEQ_LEN'],
             self.config['min_payload_len'],
             self.config['max_payload_len'],
             self.config['max_initial_distractors'],
             self.config['max_trailing_distractors']
         )
         return seq, label

# --- Collate Function (Same as Selective Copy) ---
def collate_fn(batch: List[Tuple[List[str], int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collates batch, adds CLS, tokenizes, pads, creates mask."""
    sequences, labels = zip(*batch)
    max_len = max(len(s) for s in sequences) # Should be constant == MAX_SEQ_LEN - 1
    batch_size = len(batch)

    input_ids = torch.full((batch_size, max_len + 1), pad_idx, dtype=torch.long)  # +1 for CLS
    attention_mask = torch.zeros((batch_size, max_len + 1), dtype=torch.long)  # 1 where token present

    for i, seq in enumerate(sequences):
        len_s = len(seq)
        token_ids = [vocab_map.get(t, vocab_map[UNK]) for t in seq]

        input_ids[i, 0] = vocab_map[CLS]
        input_ids[i, 1:len_s+1] = torch.tensor(token_ids)
        attention_mask[i, :len_s+1] = 1  # Mark valid tokens

    return input_ids, torch.tensor(labels, dtype=torch.long), attention_mask

# --- Main Loader Function ---
def get_char_palindrome_dataloaders(
    config: Dict[str, Any],
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """
    Creates DataLoaders for the Character Palindrome task.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing task params.
        num_workers (int): Number of workers for DataLoader.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, int, int]:
            Train, Validation, Test DataLoaders, vocab_size, pad_idx.
    """
    print("Generating Character Palindrome datasets...")
    bs = config['batch_size']

    # Pass the whole config dict to the Dataset constructor
    train_d = PalindromeDataset(config['N_TRAIN_EXAMPLES'], config)
    val_d = PalindromeDataset(config['N_VAL_EXAMPLES'], config)
    test_d = PalindromeDataset(config['N_TEST_EXAMPLES'], config)

    train_loader = DataLoader(train_d, batch_size=bs, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_d, batch_size=bs, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_d, batch_size=bs, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    return train_loader, val_loader, test_loader, vocab_size, pad_idx
