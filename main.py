# main.py

import torch
import argparse
import json
import os
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset # Example, not used if loading scripts work
import math # Added for math.ceil
from typing import Tuple, Dict, Any, Optional # Added typing

# --- Import Dataloader Functions ---
from src.data.dataloader_selective_copy import get_selective_copy_dataloaders
from src.data.dataloader_char_palindrome import get_char_palindrome_dataloaders
from src.data.dataloader_wp_palindrome import get_wp_palindrome_dataloaders

# --- Import Experiment Runner ---
from src.training.main_runner import run_experiment

# --- Import Tokenizer if needed ---
try:
    from transformers import AutoTokenizer
except ImportError:
    print("Warning: transformers library not found. WordPiece task will likely fail.")
    AutoTokenizer = None

try: 
    from transformers import AutoModel
except ImportError:
    print("Warning: transformers library not found. WordPiece task will likely fail.")
    AutoModel = None
    


# --- Placeholder Embedding Loading ---
# Keep this simple for now, focus is on dataloading logic

def load_embeddings(embedding_path: str, d_model: int, vocab_size: int) -> Optional[torch.Tensor]:
    """
    Placeholder function to load pre-trained embeddings.
    Replace with your actual embedding loading logic if needed externally.
    For WP Palindrome, embeddings are loaded within TransformerModel usually.
    """
    if not embedding_path:
        return None
    print(f"--- Placeholder: Attempting to load embeddings from {embedding_path} ---")
    # Implement actual loading from file (e.g., .pt, .vec) matching vocab_size and d_model
    # Example:
    # if os.path.exists(embedding_path):
    #     try:
    #         embeddings = torch.load(embedding_path) # Adjust based on file type
    #         if embeddings.shape == (vocab_size, d_model):
    #              print(f"--- Placeholder: Successfully loaded embeddings ---")
    #              return embeddings
    #         else:
    #              print(f"Warning: Embedding shape mismatch. Expected ({vocab_size}, {d_model}), got {embeddings.shape}")
    #              return None
    #     except Exception as e:
    #         print(f"Error loading embeddings: {e}")
    #         return None
    # else:
    #      print(f"Embedding path {embedding_path} not found.")
    #      return None
    print("--- Placeholder: No embedding loading implemented. Returning None. ---")
    return None
# --- End Placeholder Embedding Loading ---


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

    # --- Merge/Override Config with Args ---
    config['seed'] = args.seed
    config['model_type'] = args.model_type # Store model type in config
    config['task'] = args.task # Store task in config
    config['epochs'] = args.epochs if args.epochs is not None else config.get('epochs', 10)
    config['batch_size'] = args.batch_size if args.batch_size is not None else config.get('batch_size', 32)
    config['lr'] = args.lr if args.lr is not None else config.get('lr', 1e-4)
    # Use task-specific seq_len defaults if not overridden by args
    if args.max_seq_len is None:
        if args.task == 'selective_copy':
            # Needs train/id/ood lengths, handle inside dataloader or pass explicitly?
            # For now, main.py only needs one max length for pos encoding buffer.
            # We'll use the OOD length from the notebook config as the max needed.
            config['MAX_SEQ_LEN'] = config.get('seq_len_eval_ood', 200)
        elif args.task == 'char_palindrome':
            config['MAX_SEQ_LEN'] = config.get('MAX_SEQ_LEN', 64)
        elif args.task == 'wp_palindrome':
            config['MAX_SEQ_LEN'] = config.get('MAX_SEQ_LEN', 512)
        else:
             config['MAX_SEQ_LEN'] = config.get('MAX_SEQ_LEN', 128) # Default fallback
    else:
         config['MAX_SEQ_LEN'] = args.max_seq_len # Override with arg

    # Ensure MODEL_MAX_LEN (for pos encoding) is sufficient
    config['MODEL_MAX_LEN'] = config.get('MODEL_MAX_LEN', config['MAX_SEQ_LEN'] + 20) # Add buffer

    # Ensure MGA configs are present if needed
    if args.model_type.upper() == "MGA":
         required_mga_keys = {"local_window_size", "num_latents", "gru_hidden", "gate_hidden"} # Base keys
         if args.task == 'wp_palindrome':
             required_mga_keys.add("local_kernel_size") # Add kernel size if WP task
             required_mga_keys.discard("local_window_size") # Remove window size if WP task
         elif args.task in ['selective_copy', 'char_palindrome']:
             required_mga_keys.add("local_window_size")
             # No kernel size needed for these tasks in the notebook

         if not required_mga_keys.issubset(config.keys()):
              print(f"Error: MGA model selected for task {args.task}, but config is missing required keys: {required_mga_keys - set(config.keys())}")
              return
         # Add optional MGA configs if present in main config
         config['activation'] = config.get('activation', 'gelu') # Default in MGA layer is gelu/relu based on task
         config['use_stable_local_norm'] = config.get('use_stable_local_norm', False)

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
    print(f"--- Loading data for task: {args.task} ---")
    pretrained_embeddings = None # Default to None
    vocab_size = None
    pad_idx = None
    tokenizer_obj = None # For WP task

    if args.task == 'selective_copy':
        train_loader, val_loader, test_loader, vocab_size, pad_idx = get_selective_copy_dataloaders(config)
        # Note: val_loader is ID, test_loader is OOD for this task based on script logic
        print("Selective Copy DataLoaders loaded.")
    elif args.task == 'char_palindrome':
        train_loader, val_loader, test_loader, vocab_size, pad_idx = get_char_palindrome_dataloaders(config)
        print("Character Palindrome DataLoaders loaded.")
    elif args.task == 'wp_palindrome':
        if AutoTokenizer is None:
            print("Error: Transformers library needed for wp_palindrome task.")
            return
        try:
            model_name_for_tokenizer = config.get("MODEL_NAME", "bert-base-uncased") # Get from config or default
            print(f"Loading tokenizer for WP Palindrome: {model_name_for_tokenizer}")
            tokenizer_obj = AutoTokenizer.from_pretrained(model_name_for_tokenizer)
            train_loader, val_loader, test_loader, vocab_size, pad_idx = get_wp_palindrome_dataloaders(config, tokenizer_obj)
            print("WordPiece Palindrome DataLoaders loaded.")
            # Load pre-trained embeddings if specified (or handle inside model)
            if args.embedding_path: # Explicit path overrides default logic
                 pretrained_embeddings = load_embeddings(
                     args.embedding_path,
                     config['d_model'],
                     vocab_size
                 )
            else:
                 # Load default BERT embeddings to pass to model
                  print("Loading default pre-trained embeddings for WP Palindrome task...")
                  try:
                      bert_model = AutoModel.from_pretrained(model_name_for_tokenizer)
                      pretrained_embeddings = bert_model.embeddings.word_embeddings.weight.clone().detach()
                      if pretrained_embeddings.shape[0] != vocab_size or pretrained_embeddings.shape[1] != config['d_model']:
                           print(f"Warning: Loaded embedding shape {pretrained_embeddings.shape} does not match tokenizer vocab {vocab_size} and d_model {config['d_model']}.")
                           pretrained_embeddings = None # Don't use mismatched embeddings
                      else:
                           print("Successfully loaded default embeddings.")
                      del bert_model # Free memory
                  except Exception as e:
                       print(f"Could not load default embeddings: {e}. Proceeding without pre-trained embeddings.")
                       pretrained_embeddings = None

        except Exception as e:
            print(f"Error loading data for wp_palindrome: {e}")
            return
    else:
        print(f"Error: Unknown task '{args.task}'")
        return

    datasets = (train_loader, val_loader, test_loader)

    # --- Output Directory ---
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved to: {args.output_dir}")

    # --- Run Experiment ---
    # Pass vocab_size and pad_idx explicitly to the runner
    results = run_experiment(
        model_type=args.model_type,
        config=config,
        datasets=datasets,
        device=device,
        seed=args.seed,
        pretrained_embeddings=pretrained_embeddings, # Pass loaded embeddings (can be None)
        vocab_size=vocab_size, # Pass determined vocab size
        pad_idx=pad_idx,       # Pass determined pad index
        num_classes=args.num_classes, # Use arg for num_classes (default 2)
        model_checkpoint_path=args.checkpoint_path,
        eval_only=args.eval_only
    )

    # --- Save Results ---
    results_filename = f"results_{args.task}_{args.model_type}_seed{args.seed}.json"
    results_path = os.path.join(args.output_dir, results_filename)
    print(f"Saving results to {results_path}")
    try:
        # Convert numpy arrays/tensors to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                 serializable_results[key] = [arr.tolist() for arr in value]
            elif isinstance(value, (torch.Tensor)):
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

    # Task Selection
    parser.add_argument("--task", type=str, required=True,
                        choices=["selective_copy", "char_palindrome", "wp_palindrome"],
                        help="Task to run.")

    # Required arguments
    parser.add_argument("--model-type", type=str, required=True, choices=["MGA", "Baseline"],
                        help="Type of model to run ('MGA' or 'Baseline').")
    parser.add_argument("--config-path", type=str, required=True,
                        help="Path to the JSON configuration file for model hyperparameters.")
    # Data path is now optional, as HF datasets might be loaded directly
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to the dataset (required for non-HF tasks).")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save results and potential checkpoints.")

    # Optional arguments with defaults
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to use ('cuda' or 'cpu'). Default: 'cuda'.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility. Default: 42.")
    parser.add_argument("--num-classes", type=int, default=2,
                        help="Number of output classes for classification. Default: 2.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (overrides config if set).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (overrides config if set).")
    parser.add_argument("--lr", type=float, default=None,
                         help="Peak learning rate (overrides config if set).")
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="Maximum sequence length (overrides relevant config value if set).")
    parser.add_argument("--embedding-path", type=str, default=None,
                        help="Path to explicitly load pre-trained embedding file (optional, might override defaults).")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Path to a model checkpoint (.pt file) to load for evaluation or resuming.")
    parser.add_argument("--eval-only", action="store_true",
                        help="If set, skip training and only run evaluation (requires --checkpoint-path).")

    args = parser.parse_args()

    # Validate data path for non-HF tasks
    if args.task in ['selective_copy', 'char_palindrome'] and not args.data_path:
         parser.error(f"--data-path is required for task '{args.task}'")

    # Validate eval_only requirement
    if args.eval_only and not args.checkpoint_path:
        print("Warning: --eval-only flag is set, but --checkpoint-path is not provided. Evaluating a randomly initialized model.")

    # Ensure transformers library is available if wp_palindrome is selected
    if args.task == 'wp_palindrome' and AutoTokenizer is None:
         parser.error("The 'transformers' library is required for the 'wp_palindrome' task. Please install it.")

    main(args)
