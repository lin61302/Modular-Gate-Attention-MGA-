# main.py

import torch
import argparse
import json
import os
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset # Example dataset/loader

# Adjust path if necessary, assuming main.py is in repo/ and src/ is a sibling directory
# If running from repo/, these imports should work directly.
from src.training.main_runner import run_experiment

# --- Placeholder Data Loading ---
# IMPORTANT: Replace these with your actual data loading logic!
# You'll need to load your specific datasets (train, val, test)
# and return DataLoaders, vocab_size, and pad_idx.

def load_datasets(data_path: str, batch_size: int, max_seq_len: int) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """
    Placeholder function to load datasets.
    Replace with your actual data loading logic.

    Args:
        data_path (str): Path to the dataset directory or files.
        batch_size (int): Batch size for DataLoaders.
        max_seq_len (int): Maximum sequence length for padding/truncation.

    Returns:
        Tuple containing:
        - train_loader (DataLoader): DataLoader for the training set.
        - val_loader (DataLoader): DataLoader for the validation set.
        - test_loader (DataLoader): DataLoader for the test set.
        - vocab_size (int): The size of the vocabulary.
        - pad_idx (int): The index used for padding tokens.
    """
    print(f"--- Placeholder: Loading datasets from {data_path} ---")
    print(f"--- Placeholder: Using Batch Size: {batch_size}, Max Seq Len: {max_seq_len} ---")

    # --- Dummy Data Generation ---
    # Replace this section with your actual data reading and tokenization
    vocab_size = 1000  # Example vocab size
    pad_idx = 0      # Example padding index
    num_samples_train = 500
    num_samples_val = 100
    num_samples_test = 100
    num_classes = 2 # Assuming binary classification, adjust if needed

    def create_dummy_data(num_samples):
        # Dummy input_ids (random ints up to vocab_size)
        input_ids = torch.randint(1, vocab_size, (num_samples, max_seq_len))
        # Dummy attention mask (mostly 1s, some 0s for padding simulation)
        attention_mask = torch.ones((num_samples, max_seq_len), dtype=torch.long)
        for i in range(num_samples): # Add some random padding
             pad_len = random.randint(0, max_seq_len // 4)
             if pad_len > 0:
                 input_ids[i, -pad_len:] = pad_idx
                 attention_mask[i, -pad_len:] = 0
        # Dummy labels
        labels = torch.randint(0, num_classes, (num_samples,))
        return TensorDataset(input_ids, attention_mask, labels) # Format: (input_ids, attention_mask, labels)

    train_dataset = create_dummy_data(num_samples_train)
    val_dataset = create_dummy_data(num_samples_val)
    test_dataset = create_dummy_data(num_samples_test)

    # --- Create DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"--- Placeholder: Returning Dummy DataLoaders (Train/Val/Test): {len(train_loader)}/{len(val_loader)}/{len(test_loader)} batches ---")
    print(f"--- Placeholder: Vocab Size: {vocab_size}, Padding Index: {pad_idx} ---")

    # IMPORTANT: Ensure the batch format matches what train_epoch/evaluate expect.
    # The current dummy data creates batches of (input_ids, attention_mask, labels) tuples.
    # Modify if your model/training loop expects dicts (e.g., {'input_ids': ..., 'attention_mask': ..., 'labels': ...})
    return train_loader, val_loader, test_loader, vocab_size, pad_idx

def load_embeddings(embedding_path: str, d_model: int, vocab_size: int) -> Optional[torch.Tensor]:
    """
    Placeholder function to load pre-trained embeddings.
    Replace with your actual embedding loading logic.

    Args:
        embedding_path (str): Path to the embedding file (e.g., .pt, .txt, .vec).
        d_model (int): Expected embedding dimension.
        vocab_size (int): Expected vocabulary size.

    Returns:
        Optional[torch.Tensor]: Tensor of shape (vocab_size, d_model) or None.
    """
    if not embedding_path:
        return None
    print(f"--- Placeholder: Loading embeddings from {embedding_path} ---")
    # --- Dummy Embedding Generation ---
    # Replace this with logic to load from file (GloVe, Word2Vec, FastText, etc.)
    # Ensure the loaded tensor has shape (vocab_size, d_model)
    print(f"--- Placeholder: Creating Dummy Embeddings ({vocab_size} x {d_model}) ---")
    try:
        embeddings = torch.randn(vocab_size, d_model)
        print(f"--- Placeholder: Successfully created dummy embeddings ---")
        return embeddings
    except Exception as e:
        print(f"Error creating/loading dummy embeddings: {e}")
        return None
# --- End Placeholder Data Loading ---


def main(args):
    # --- Configuration ---
    print("--- Loading Configuration ---")
    try:
        with open(args.config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {args.config_path}")
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config_path}")
        return
    except json.JSONDecodeError:
         print(f"Error: Could not decode JSON from {args.config_path}")
         return

    # Override config with command-line arguments where applicable
    config['seed'] = args.seed
    config['model_type'] = args.model_type # Store model type in config for reference if needed
    config['epochs'] = args.epochs if args.epochs is not None else config.get('epochs', 10) # Default epochs if not in args/config
    config['batch_size'] = args.batch_size if args.batch_size is not None else config.get('batch_size', 32)
    config['lr'] = args.lr if args.lr is not None else config.get('lr', 1e-4)
    # Ensure essential MGA configs are present if model is MGA
    if args.model_type.upper() == "MGA":
         required_mga_keys = {"local_window_size", "num_latents", "gru_hidden", "gate_hidden"}
         if not required_mga_keys.issubset(config.keys()):
              print(f"Error: MGA model selected, but config is missing required keys: {required_mga_keys - config.keys()}")
              return
    # Update sequence length from args if provided, else use config
    config['MAX_SEQ_LEN'] = args.max_seq_len if args.max_seq_len is not None else config.get('MAX_SEQ_LEN', 128)

    # --- Device Setup ---
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        if args.device == 'cuda':
            print("CUDA requested but not available, using CPU.")
        device = torch.device('cpu')
        print("Using CPU")

    # --- Seed ---
    print(f"Setting random seed: {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    # --- Data Loading ---
    # IMPORTANT: Replace load_datasets call with your actual implementation
    train_loader, val_loader, test_loader, vocab_size, pad_idx = load_datasets(
        args.data_path,
        config['batch_size'],
        config['MAX_SEQ_LEN']
    )
    datasets = (train_loader, val_loader, test_loader)

    # --- Embeddings ---
    # IMPORTANT: Replace load_embeddings call if using pre-trained embeddings
    pretrained_embeddings = load_embeddings(
        args.embedding_path,
        config['d_model'], # Assumes d_model is in config
        vocab_size
    )
    if pretrained_embeddings is not None:
        print("Loaded pre-trained embeddings.")
        # Optional: Check embedding shape consistency
        if pretrained_embeddings.shape[0] != vocab_size or pretrained_embeddings.shape[1] != config['d_model']:
            print(f"Warning: Loaded embedding shape {pretrained_embeddings.shape} "
                  f"does not match vocab_size {vocab_size} and d_model {config['d_model']}. Check consistency.")
            # Decide how to handle mismatch (e.g., exit, resize, ignore)
            # For now, we proceed, but TransformerModel will also warn/potentially error.

    # --- Output Directory ---
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved to: {args.output_dir}")

    # --- Run Experiment ---
    results = run_experiment(
        model_type=args.model_type,
        config=config,
        datasets=datasets,
        device=device,
        seed=args.seed,
        pretrained_embeddings=pretrained_embeddings,
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        num_classes=args.num_classes,
        model_checkpoint_path=args.checkpoint_path,
        eval_only=args.eval_only
    )

    # --- Save Results ---
    results_filename = f"results_{args.model_type}_seed{args.seed}.json"
    results_path = os.path.join(args.output_dir, results_filename)
    print(f"Saving results to {results_path}")
    try:
        # Convert numpy arrays (like gate weights) to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                 serializable_results[key] = [arr.tolist() for arr in value]
            elif isinstance(value, (torch.Tensor)): # Convert potential tensor outputs
                 serializable_results[key] = value.cpu().numpy().tolist()
            else:
                serializable_results[key] = value

        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        print("Results saved successfully.")
    except TypeError as e:
        print(f"Error saving results: {e}. Some results might not be JSON serializable.")
    except Exception as e:
         print(f"An unexpected error occurred during result saving: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MGA or Baseline Transformer Experiment")

    # Required arguments
    parser.add_argument("--model-type", type=str, required=True, choices=["MGA", "Baseline"],
                        help="Type of model to run ('MGA' or 'Baseline').")
    parser.add_argument("--config-path", type=str, required=True,
                        help="Path to the JSON configuration file for model hyperparameters.")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to the dataset (specific format depends on load_datasets function).")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save results and potential checkpoints.")

    # Optional arguments with defaults
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to use ('cuda' or 'cpu'). Default: 'cuda'.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility. Default: 42.")
    parser.add_argument("--num-classes", type=int, default=2,
                        help="Number of output classes for classification. Default: 2.")
    parser.add_argument("--epochs", type=int, default=None, # Default handled in main()
                        help="Number of training epochs (overrides config if set).")
    parser.add_argument("--batch-size", type=int, default=None, # Default handled in main()
                        help="Batch size (overrides config if set).")
    parser.add_argument("--lr", type=float, default=None, # Default handled in main()
                         help="Peak learning rate (overrides config if set).")
    parser.add_argument("--max-seq-len", type=int, default=None, # Default handled in main()
                        help="Maximum sequence length (overrides config if set).")
    parser.add_argument("--embedding-path", type=str, default=None,
                        help="Path to pre-trained embedding file (optional).")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Path to a model checkpoint (.pt file) to load for evaluation or resuming.")
    parser.add_argument("--eval-only", action="store_true",
                        help="If set, skip training and only run evaluation (requires --checkpoint-path).")

    args = parser.parse_args()

    # Validate eval_only requirement
    if args.eval_only and not args.checkpoint_path:
        print("Warning: --eval-only flag is set, but --checkpoint-path is not provided. Evaluating a randomly initialized model.")
        # Alternatively, could raise an error:
        # parser.error("--eval-only requires --checkpoint-path to be set.")

    main(args)
