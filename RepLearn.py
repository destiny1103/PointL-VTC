import argparse
import sys
import numpy as np
import pandas as pd
import os
os.environ["OMP_NUM_THREADS"] = "5"
import datetime
import json
import random

import torch
import torch.multiprocessing as mp
from torch.optim import Adam
from tqdm import tqdm
import traceback

# Local imports
from src.gnn_model import build_encoder
from src.dataset import get_dataset
from src.utils import data_preprocessing
from src.train2Rep import train_batch_size, train_subgraph_sampling


class Logger:
    """Log classes that only output to log files"""
    def __init__(self, log_file):
        self.log = open(log_file, 'a', encoding='utf-8')

    def write(self, message):
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.log.flush()

    def close(self):
        self.log.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RepLearn - Representation Learning with Graph Neural Networks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # === GPU device andParallel Processing ===
    parser.add_argument('--gpu', type=int, nargs='+', default=[6, 7], help='GPU device IDs to use (e.g., --gpu 6 7)')
    parser.add_argument("--max_workers", type=int, default=6, help="Number of parallel workers")

    # === Path Configuration ===
    parser.add_argument("--base_dir", type=str, default="data/TPC1data/input")
    parser.add_argument("--aug_timestamp", type=str, default="20260418_114004", help="Timestamp of StructAug output (required if using augmented data)")

    # === Hyperparameters ===
    parser.add_argument("--gnn_model", type=str, default="GAT_Khop", choices=["GCN", "GAT", "SAGE", "GIN", "GAT_Khop"])
    parser.add_argument("--use_augmented_features", type=bool, default=False)
    parser.add_argument("--use_augmented_graph", type=bool, default=True) # False True
    parser.add_argument("--training_strategy", type=str, default="subgraph", choices=["batch_size", "subgraph"],)
    parser.add_argument("--output_criterion", type=str, default="last_epoch", choices=["min_loss", "max_auc", "max_acc", "last_epoch"])

    # === Model Parameters ===
    parser.add_argument("--max_epoch", type=int, default=50, help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of clusters")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--embedding_size", type=int, default=24, help="Embedding dimension")
    parser.add_argument("--weight_decay", type=float, default=5e-3, help="Weight decay")
    parser.add_argument("--alpha", type=float, default=0.2, help="LeakyReLU alpha")
    parser.add_argument("--loss_weight", type=bool, default=True, help="Use weighted loss for imbalanced edges")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # === Subgraph Parameters ===
    parser.add_argument("--subgraph_sample_method", type=str, default="random", choices=["by_point", "random"])
    parser.add_argument("--subgraph_k", type=int, default=3, help="K-hop for subgraph sampling")
    parser.add_argument("--subgraph_stride", type=int, default=5, help="Stride for by_point sampling")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for subgraph training")

    return parser.parse_args()


def seed_all(seed=42):
    """Set random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_data(dataset_name, args):
    """
    Load data based on augmentation settings.

    4 scenarios:
    1. original graph + original features
    2. original graph + augmented features
    3. augmented graph + original features
    4. augmented graph + augmented features

    Returns:
        dataset: PyG Data object with x, edge_index, y
    """
    input_dir = os.path.join(args.base_dir, 'input')
    # Always load original dataset for labels (and potentially features/graph)
    original_dataset = get_dataset(dataset_name, root_dir=input_dir)
    data = original_dataset[0]

    # Load augmented features if needed
    if args.use_augmented_features:
        if args.aug_timestamp is None:
            raise ValueError("aug_timestamp required when use_augmented_features=True")
        aug_feat_path = os.path.join(
            args.base_dir, 'AugOutput', args.aug_timestamp, dataset_name, 'node_embeddings.txt'
        )
        if not os.path.exists(aug_feat_path):
            raise FileNotFoundError(f"Augmented features not found: {aug_feat_path}")
        x = np.loadtxt(aug_feat_path)
        data.x = torch.tensor(x, dtype=torch.float32)

    # Load augmented graph if needed
    if args.use_augmented_graph:
        if args.aug_timestamp is None:
            raise ValueError("aug_timestamp required when use_augmented_graph=True")
        aug_graph_path = os.path.join(
            args.base_dir, 'AugOutput', args.aug_timestamp, dataset_name, 'augmented_graph.txt'
        )
        if not os.path.exists(aug_graph_path):
            raise FileNotFoundError(f"Augmented graph not found: {aug_graph_path}")
        edge_data = np.loadtxt(aug_graph_path)
        if edge_data.ndim == 1:
            edge_data = edge_data.reshape(1, -1)
        src = torch.tensor(edge_data[:, 0], dtype=torch.long)
        dst = torch.tensor(edge_data[:, 1], dtype=torch.long)
        data.edge_index = torch.stack([src, dst], dim=0)
        if edge_data.shape[1] > 2:
            data.edge_weight = torch.tensor(edge_data[:, 2], dtype=torch.float)

    return data


def train_one_dataset(dataset_name, args, device, output_dir):
    """
    Train representation learning on a single dataset.

    Args:
        dataset_name: name of dataset
        args: parsed arguments
        device: torch device
        output_dir: output directory for this timestamp

    Returns:
        status: 'SUCCESS' or error message
    """
    seed_all(42)
    
    # Create dataset output directory
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"[{dataset_name}] Using device: {device}")
    print(f"{'='*60}")
    print(f"\n--- Processing dataset: {dataset_name} ---")

    try:
        # Load data
        data = load_data(dataset_name, args)

        # Preprocess data (adds adj, adj_label)
        data = data_preprocessing(data)

        # Move to device
        data = data.to(device)

        # Get input dimension
        in_channels = data.x.size(1)

        # Build model
        model = build_encoder(
            model_name=args.gnn_model,
            in_channels=in_channels,
            hidden_channels=args.hidden_size,
            out_channels=args.embedding_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            alpha=args.alpha
        ).to(device)

        # Optimizer
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Train based on strategy
        if args.training_strategy == 'batch_size':
            result = train_batch_size(model, data, args, device, optimizer)
        else:  # subgraph
            result = train_subgraph_sampling(model, data, args, device, optimizer)

        # Save history
        history_df = pd.DataFrame(result['history'])
        history_path = os.path.join(dataset_output_dir, 'history.txt')
        with open(history_path, 'w') as f:
            f.write(f"--- Training History for {dataset_name} ---\n")
            f.write(f"Best epoch: {result['best_epoch']} (criterion: {args.output_criterion})\n\n")
            history_df.to_string(f, index=False)

        # Save best labels
        labels_path = os.path.join(dataset_output_dir, 'labels_best.txt')
        pd.DataFrame(result['labels_best']).to_csv(labels_path, index=False, header=False)

        # Save best features
        features_path = os.path.join(dataset_output_dir, 'features_best.txt')
        pd.DataFrame(result['z_best']).to_csv(features_path, index=False, header=False)

        print(f"[{dataset_name}] Completed - Best epoch: {result['best_epoch']}")
        print(f"{'='*60}\n")
        return 'SUCCESS'

    except Exception as e:
        error_msg = f"[{dataset_name}] Error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        return error_msg


def process_dataset_wrapper(params):
    """
    Wrapper function for parallel processing.

    Args:
        params: tuple of (dataset_path, args, output_dir, gpu_id)
    """
    dataset_path, args, output_dir, gpu_id = params
    dataset_name = os.path.basename(dataset_path)

    # gpu_id is the local CUDA index after CUDA_VISIBLE_DEVICES filtering.
    if args.cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    return train_one_dataset(dataset_name, args, device, output_dir)


def save_config(args, output_dir, timestamp):
    """Save configuration to JSON file."""
    config = {
        "timestamp": timestamp,
        "base_dir": args.base_dir,
        "hyperparameters": {
            "gnn_model": args.gnn_model,
            "use_augmented_features": args.use_augmented_features,
            "use_augmented_graph": args.use_augmented_graph,
            "training_strategy": args.training_strategy,
            "subgraph_sample_method": args.subgraph_sample_method,
            "output_criterion": args.output_criterion,
            "max_epoch": args.max_epoch,
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "max_workers": args.max_workers
        },
        "augmentation_source": {
            "timestamp": args.aug_timestamp,
            "used_aug_features": args.use_augmented_features,
            "used_aug_graph": args.use_augmented_graph
        },
        "training_config": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "n_clusters": args.n_clusters,
            "hidden_size": args.hidden_size,
            "embedding_size": args.embedding_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "alpha": args.alpha
        },
        "subgraph_config": {
            "subgraph_k": args.subgraph_k,
            "subgraph_stride": args.subgraph_stride,
            "batch_size": args.batch_size
        }
    }

    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")


def main():
    """Main entry point."""
    args = parse_args()
    args.cuda = torch.cuda.is_available()

    prev_stdout = sys.stdout
    prev_stderr = sys.stderr

    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.base_dir, 'RepOutput', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup main log file
    main_log_file = os.path.join(output_dir, 'main_log.txt')
    main_logger = Logger(main_log_file)

    # Redirect to log by default to keep terminal clean
    sys.stdout = main_logger
    sys.stderr = main_logger

    print(f"\n=== Representation Learning (RepLearn) ===")
    print(f"Output directory: {output_dir}")
    print(f"Main log file: {main_log_file}")
    print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in args.gpu)
    num_local_gpus = len(args.gpu)

    print(f"Max Workers: {args.max_workers}, GPUs: {args.gpu} (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")

    # Record hyperparameters to log
    print(f"\n{'='*60}")
    print("Hyperparameters Configuration:")
    print(f"{'='*60}")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")
    print(f"{'='*60}\n")

    # Also print hyperparameters to terminal only
    print(f"\n{'='*60}", file=prev_stdout)
    print("Hyperparameters Configuration:", file=prev_stdout)
    print(f"{'='*60}", file=prev_stdout)
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}", file=prev_stdout)
    print(f"{'='*60}\n", file=prev_stdout)

    # Validate augmented data requirements
    if (args.use_augmented_features or args.use_augmented_graph) and args.aug_timestamp is None:
        print("Error: --aug_timestamp required when using augmented data")
        sys.stdout = prev_stdout
        sys.stderr = prev_stderr
        main_logger.close()
        sys.exit(1)

    # Save configuration
    save_config(args, output_dir, timestamp)

    # Find all datasets
    input_dir = os.path.join(args.base_dir, 'input')
    if not os.path.exists(input_dir):
        print(f"Error: Path {input_dir} does not exist.")
        sys.stdout = prev_stdout
        sys.stderr = prev_stderr
        main_logger.close()
        sys.exit(1)

    datasets = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])

    if not datasets:
        print("No datasets found.")
        sys.stdout = prev_stdout
        sys.stderr = prev_stderr
        main_logger.close()
        sys.exit(1)

    print(f"Found {len(datasets)} datasets in {input_dir}")

    # Restore stdout/stderr for terminal output (tqdm only)
    sys.stdout = prev_stdout
    sys.stderr = prev_stderr

    # Set multiprocessing start method
    if args.max_workers > 1:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    # Build task list
    tasks = []
    for i, name in enumerate(datasets):
        dataset_path = os.path.join(input_dir, name)
        if args.max_workers == 1:
            gpu_id = 0
        else:
            gpu_id = i % num_local_gpus
        tasks.append((dataset_path, args, output_dir, gpu_id))

    # Process datasets
    if args.max_workers == 1:
        results = []
        for task in tqdm(tasks, desc="Processing", unit="dataset"):
            sys.stdout = main_logger
            sys.stderr = main_logger
            dataset_name = os.path.basename(task[0])
            print(f"\n>>> Processing Dataset: {dataset_name}")
            sys.stdout = prev_stdout
            sys.stderr = prev_stderr
            
            result = process_dataset_wrapper(task)
            results.append(result)
    else:
        with mp.Pool(processes=args.max_workers) as pool:
            results = list(tqdm(pool.imap(process_dataset_wrapper, tasks), 
                              total=len(tasks), desc="Processing", unit="dataset"))

    # Summary
    success_count = sum(1 for r in results if r == 'SUCCESS')
    sys.stdout = main_logger
    sys.stderr = main_logger
    print(f"\n=== All Datasets Completed ===")
    print(f"Processing completed: {success_count}/{len(tasks)} successful")
    print(f"Results saved to: {output_dir}")
    print(f"Log file: {main_log_file}\n")
    print(f"\nProcessing completed: {success_count}/{len(tasks)} successful")
    print(f"End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    sys.stdout = prev_stdout
    sys.stderr = prev_stderr
    main_logger.close()


if __name__ == "__main__":
    main()
