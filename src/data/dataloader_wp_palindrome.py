# dataloader_wp_palindrome.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from typing import List, Tuple, Dict, Any
from transformers import PreTrainedTokenizerBase, DataCollatorWithPadding, AutoModel

# --- Data Generation ---
def generate_wp_palindrome_sequence(config: Dict[str, Any], tokenizer: PreTrainedTokenizerBase) -> Tuple[List[str], int]:
    """Generates token list (before IDs) for the WordPiece palindrome task."""
    max_seq_len = config["MAX_SEQ_LEN"]
    payload_len_char = random.randint(config["min_payload_len_char"], config["max_payload_len_char"])
    label = random.choice([0, 1])

    # Create character payload
    char_payload = [random.choice(config["payload_alphabet_char"]) for _ in range(payload_len_char)]
    if label == 1:
        half_len = math.ceil(payload_len_char / 2)
        first_half = char_payload[:half_len]
        second_half = first_half[:payload_len_char // 2][::-1]
        char_payload = first_half + second_half
    else:
        max_tries = 10
        tries = 0
        while char_payload == char_payload[::-1] and tries < max_tries:
             char_payload = [random.choice(config["payload_alphabet_char"]) for _ in range(payload_len_char)]
             tries += 1
        if char_payload == char_payload[::-1]:
            char_payload[-1] = config["payload_alphabet_char"][(config["payload_alphabet_char"].index(char_payload[-1]) + 1) % len(config["payload_alphabet_char"])]

    payload_str = "".join(char_payload)
    payload_tokens = tokenizer.tokenize(payload_str)
    payload_len_tok = len(payload_tokens)

    # BERT uses [SEP] token (ID 102 typically)
    sep_token = tokenizer.sep_token
    marker_a_tok = [sep_token]
    marker_b_tok = [sep_token]

    # Calculate available space for distractors
    payload_markers_len = payload_len_tok + len(marker_a_tok) + len(marker_b_tok)
    target_len = max_seq_len - 1 # Account for [CLS] token added later
    max_total_distractors = target_len - payload_markers_len

    if max_total_distractors < 0:
        # If tokenized payload + markers exceed max_len, need to regenerate
        # print(f"Warning: Payload tokens ({payload_len_tok}) + markers too long for max_seq_len {max_seq_len}. Retrying...")
        # Returning None signals retry needed in the Dataset's __getitem__
        return None, None

    # Generate distractors
    num_initial_dist = random.randint(0, max_total_distractors)
    num_trailing_dist = max_total_distractors - num_initial_dist
    initial_dist = random.choices(config["distractor_tokens"], k=num_initial_dist)
    trailing_dist = random.choices(config["distractor_tokens"], k=num_trailing_dist)

    # Construct sequence (list of tokens)
    sequence_tokens = initial_dist + marker_a_tok + payload_tokens + marker_b_tok + trailing_dist

    # Ensure exact length by padding/truncating distractors if necessary
    # (Padding is handled by DataCollator, but ensure we don't exceed target)
    current_len = len(sequence_tokens)
    if current_len > target_len:
        sequence_tokens = sequence_tokens[:target_len]
    # Padding to target_len will be done by collator

    return sequence_tokens, label

# --- Dataset Class ---
class WpPalindromeDataset(Dataset):
     def __init__(self, num_samples: int, config: Dict[str, Any], tokenizer: PreTrainedTokenizerBase):
        self.num_samples = num_samples
        self.config = config
        self.tokenizer = tokenizer
        self.cls_token_id = tokenizer.cls_token_id
        self.pad_idx = tokenizer.pad_token_id # Store pad_idx
        print(f"WpPalindromeDataset created for {num_samples} samples (generated on the fly).")

     def __len__(self) -> int:
        return self.num_samples

     def __getitem__(self, idx: int) -> Dict[str, Any]:
         # Keep trying until a valid sequence is generated
         while True:
            seq_tokens, label = generate_wp_palindrome_sequence(self.config, self.tokenizer)
            if seq_tokens is not None: # Check if generation was successful
                # Convert tokens to IDs, add CLS
                input_ids = [self.cls_token_id] + self.tokenizer.convert_tokens_to_ids(seq_tokens)

                # Create attention mask (1 for real tokens, 0 for padding)
                # Padding will be handled by DataCollator, so mask is initially all 1s
                attention_mask = [1] * len(input_ids)

                # Return unpadded sequences and mask; collator will pad
                return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label}
            else:
                # Regeneration was needed, loop will try again
                pass

# --- Collate Function (using HF DataCollator) ---
# This is often simpler than custom padding logic
# collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

# --- Main Loader Function ---
def get_wp_palindrome_dataloaders(
    config: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase, # Pass the loaded tokenizer
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """
    Creates DataLoaders for the WordPiece Palindrome task.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        tokenizer (PreTrainedTokenizerBase): Initialized Hugging Face tokenizer.
        num_workers (int): Number of workers for DataLoader.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, int, int]:
            Train, Validation, Test DataLoaders, vocab_size, pad_idx.
    """
    print("Generating WordPiece Palindrome datasets...")
    bs = config['batch_size']
    pad_idx_local = tokenizer.pad_token_id # Get from tokenizer
    vocab_size_local = tokenizer.vocab_size # Get from tokenizer

    train_d = WpPalindromeDataset(config['N_TRAIN_EXAMPLES'], config, tokenizer)
    val_d = WpPalindromeDataset(config['N_VAL_EXAMPLES'], config, tokenizer)
    test_d = WpPalindromeDataset(config['N_TEST_EXAMPLES'], config, tokenizer)

    # Use HF DataCollator for padding
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length', max_length=config['MAX_SEQ_LEN'], return_tensors="pt")

    train_loader = DataLoader(train_d, batch_size=bs, shuffle=True, collate_fn=collator, num_workers=num_workers)
    val_loader = DataLoader(val_d, batch_size=bs, shuffle=False, collate_fn=collator, num_workers=num_workers)
    test_loader = DataLoader(test_d, batch_size=bs, shuffle=False, collate_fn=collator, num_workers=num_workers)

    # Note: The collator will return a dictionary batch, which train_loop.py handles.
    return train_loader, val_loader, test_loader, vocab_size_local, pad_idx_local
