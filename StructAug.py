import argparse
import random
import time
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from datetime import datetime
import sys
import os
import torch.multiprocessing as mp
import torch_geometric.transforms as T
from torch_geometric.data import Data
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from src.dataset import get_augdata
from src.utils import calculate_edge_accuracy
from src.gnn_model import augment_edges_dot
from src.model import GCNLinkPredictor, GATLinkPredictor
from src.train2Aug import train as pull_train
from src.train2Aug import test as pull_test


_RUNTIME_TUNED = False


class Logger:
    """仅输出到日志文件的日志类"""
    def __init__(self, log_file):
        self.log = open(log_file, 'a', encoding='utf-8')

    def write(self, message):
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.log.flush()

    def close(self):
        self.log.close()


def log_main_run_info(logger, args, output_root, main_log_file):
    """Write to main log"""
    logger.write(f"\n=== Structure Augmentation Training ===\n")
    logger.write(f"Output directory: {output_root}\n")
    logger.write(f"Main log file: {main_log_file}\n")
    logger.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    logger.write(f"\n{'='*60}\n")
    logger.write("Hyperparameters Configuration:\n")
    logger.write(f"{'='*60}\n")
    for arg, value in sorted(vars(args).items()):
        logger.write(f"  {arg}: {value}\n")
    logger.write(f"{'='*60}\n\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, nargs='+', default=[6, 7],
                        help='GPU device IDs to use (e.g., --gpu 6 7)')
    parser.add_argument('--seed', type=int, default=42)
    # Data and thread count
    parser.add_argument('--custom_root', type=str, default="data/TPC1data/input")
    parser.add_argument('--max_workers', type=int, default=4, help='Max concurrent datasets processed')
    parser.add_argument('--cpu_threads', type=int, default=2, help='Per-process PyTorch intra-op CPU threads')
    parser.add_argument('--interop_threads', type=int, default=2, help='Per-process PyTorch inter-op CPU threads')
    parser.add_argument('--blas_threads', type=int, default=2, help='Per-process BLAS/OpenMP thread cap')

    # Training hyperparameters
    parser.add_argument('--num_augmentations', type=int, default=5, help='Total number of augmentations to perform')
    parser.add_argument('--test-ratio', type=float, default=0.2)
    parser.add_argument('--units', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for optimizer')
    parser.add_argument('--link_model', type=str, default='gat', choices=['gcn', 'gat'], help='Link predictor backbone for PULL training')
    parser.add_argument('--link_heads', type=int, default=2, help='Number of attention heads for GATLinkPredictor')
    parser.add_argument('--link_dropout', type=float, default=0.3, help='Dropout used in GATLinkPredictor')

    # Structural reinforcement parameters
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience for first augmentation')
    parser.add_argument('--min_epochs_before_augment', type=int, default=50, help='Minimum epochs before first augmentation')
    parser.add_argument('--augment_interval', type=int, default=2, help='Perform structure augmentation every N epochs after first augmentation')
    parser.add_argument('--augment_strategy', type=str, default='degree_asc', choices=['degree_desc', 'degree_asc', 'random_sel', 'interval'], help='Node selection strategy for edge augmentation')
    parser.add_argument('--edge_add_ratio', type=float, default=0.1, help='Ratio of edges to add per augmentation (based on original graph)')
    parser.add_argument('--augment_prob_threshold', type=float, default=0.6, help='Probability threshold for adding edges')
    parser.add_argument('--pull_train_edge_ratio', type=float, default=0.01, help='Fixed ratio of edges added by PULL training per epoch')
    parser.add_argument('--pull_inner_epochs', type=int, default=100, help='Inner optimization epochs inside each outer epoch')
    parser.add_argument('--pull_inner_patience', type=int, default=10, help='Early stop patience for inner optimization loop')
    parser.add_argument('--pull_inner_min_delta', type=float, default=1e-4, help='Minimum loss improvement for inner-loop early stop')

    return parser.parse_args()


def configure_runtime_threads(args):
    """Limit CPU oversubscription per process and keep GPU workers lightweight."""
    global _RUNTIME_TUNED

    blas_threads = str(max(1, args.blas_threads))
    for env_name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[env_name] = blas_threads

    if not _RUNTIME_TUNED:
        torch.set_num_threads(max(1, args.cpu_threads))
        try:
            torch.set_num_interop_threads(max(1, args.interop_threads))
        except RuntimeError:
            # Inter-op thread count may already be initialized by runtime.
            pass
        _RUNTIME_TUNED = True


def setup_environment(dataset_name, gpu_id, args, output_root):
    """gpu_id: local CUDA device index (0, 1, ...) after CUDA_VISIBLE_DEVICES filtering."""
    configure_runtime_threads(args)

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset_output_dir = os.path.join(output_root, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    log_file = os.path.join(dataset_output_dir, 'training_log.txt')

    logger = Logger(log_file)
    prev_stdout = sys.stdout
    prev_stderr = sys.stderr
    sys.stdout = logger
    sys.stderr = logger

    print(f"\n{'='*60}")
    print(f"[{dataset_name}] Using device: {device}")
    print(f"PID: {os.getpid()}, local_gpu_id: {gpu_id}")
    print(f"CPU threads (torch intra/inter): {torch.get_num_threads()}/{torch.get_num_interop_threads()}")
    print(f"BLAS threads env: OMP={os.environ.get('OMP_NUM_THREADS')}, MKL={os.environ.get('MKL_NUM_THREADS')}, OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')}")
    print(f"Log file: {log_file}")
    print(f"{'='*60}")

    return device, logger, log_file, prev_stdout, prev_stderr, dataset_output_dir


def build_link_predictor(args, num_features, device):
    model_name = args.link_model.lower()
    if model_name == 'gcn':
        return GCNLinkPredictor(num_features, args.units, args.units).to(device)
    if model_name == 'gat':
        return GATLinkPredictor(
            num_features,
            args.units,
            args.units,
            heads=args.link_heads,
            dropout=args.link_dropout,
        ).to(device)
    raise ValueError(f"Unsupported link_model: {args.link_model}")


def check_aug(state, epoch, args):
    if not state['first_augment_done']:
        early_stop = state['patience_counter'] > args.patience and epoch >= args.min_epochs_before_augment
        max_epoch = epoch >= 100
        if early_stop or max_epoch:
            return True, 'early_stop' if early_stop else 'max_epoch'
    elif (epoch - state['first_augment_epoch']) % args.augment_interval == 0:
        return True, 'interval'
    return False, None


def structure_augmentation(model, state, fulldata, epoch, args, device,
                           inference_edge_index=None, inference_edge_weight=None):
    full_edge_index = fulldata.edge_index
    full_edge_weight = fulldata.edge_weight
    node_features = fulldata.node_features
    node_labels = fulldata.node_labels
    add_edges = fulldata.add_edges

    if inference_edge_index is None:
        inference_edge_index = full_edge_index
    if inference_edge_weight is None:
        inference_edge_weight = torch.ones(inference_edge_index.size(1), device=device)

    print(f"\n>>> Performing Structure Augmentation at Epoch {epoch} ...")
    print(f"    Inference graph edges: {inference_edge_index.size(1) // 2} (PULL-expanded)")
    print(f"    Output graph edges: {full_edge_index.size(1) // 2} (independent base)")

    with torch.no_grad():
        if hasattr(model, 'encode'):
            full_z = model.encode(node_features, inference_edge_index, inference_edge_weight)
        else:
            full_z = model(node_features, inference_edge_index, inference_edge_weight)

    new_edge_index, new_edge_weight, state['node_enhancement_count'] = augment_edges_dot(
        full_z, full_edge_index, add_edges, args.augment_strategy,
        prob_threshold=args.augment_prob_threshold,
        node_enhancement_count=state['node_enhancement_count']
    )

    if new_edge_index.size(1) > 0:
        edge_metrics = calculate_edge_accuracy(new_edge_index, node_labels, device)
        state['all_new_edges'].append(new_edge_index.clone())
        state['all_augmentation_accs'].append(edge_metrics)

        new_edge_aug_id = torch.full((new_edge_index.size(1),), state['augmentation_count'],
                                     dtype=torch.long, device=device)

        fulldata.edge_index = torch.cat([full_edge_index, new_edge_index], dim=1)
        fulldata.edge_weight = torch.cat([full_edge_weight, new_edge_weight])
        state['edge_augmentation_id'] = torch.cat([state['edge_augmentation_id'], new_edge_aug_id])

        print(f"    Added {new_edge_index.size(1) // 2} new edges")
        print(f"    This augmentation Acc: {edge_metrics['accuracy']:.4f}")
        print(f"    Label-1 edges: {edge_metrics['label_1_count']}, Label-5 edges: {edge_metrics['label_5_count']}, Total (1+5): {edge_metrics['label_15_total']}")
    else:
        print(f"    No new edges added")
        state['all_augmentation_accs'].append({
            'accuracy': 0.0,
            'label_1_count': 0,
            'label_5_count': 0,
            'label_15_total': 0
        })

    return fulldata, full_z


def resplit_augdata(fulldata, edge_augmentation_id, args, device):
    augmented_data = Data(
        x=fulldata.node_features,
        edge_index=fulldata.edge_index,
        edge_weight=fulldata.edge_weight,
        edge_aug_id=edge_augmentation_id,
        y=fulldata.node_labels
    )

    transform = T.RandomLinkSplit(
        num_val=0.0,
        num_test=args.test_ratio,
        is_undirected=True,
        add_negative_train_samples=False
    )

    train_data, _, test_data = transform(augmented_data)
    train_data = train_data.to(device)
    test_data = test_data.to(device)

    test_pos_mask = test_data.edge_label == 1
    test_pos_edge_index = test_data.edge_label_index[:, test_pos_mask].to(device)

    return train_data, test_data, test_pos_edge_index


def finalize_and_save(state, fulldata, dataset_output_dir, log_file, epoch, device, dataset_name):
    print(f"\n{'='*60}")
    print("Training Finished.")
    print(f'[{dataset_name}] Best Epoch: {state["best_epoch"]:02d}, Best Test AUC: {state["global_best_auc"]:.4f}')
    print(f"Total Epochs: {epoch}")
    print(f"Total Augmentations: {state['augmentation_count']}")

    print(f"\n--- Augmentation Details ---")
    total_label_1 = 0
    total_label_5 = 0
    total_label_15 = 0

    for i, metrics in enumerate(state['all_augmentation_accs'], 1):
        print(f"  Augmentation #{i}: Acc = {metrics['accuracy']:.4f}, Label-1: {metrics['label_1_count']}, Label-5: {metrics['label_5_count']}, Total (1+5): {metrics['label_15_total']}")
        total_label_1 += metrics['label_1_count']
        total_label_5 += metrics['label_5_count']
        total_label_15 += metrics['label_15_total']

    if len(state['all_new_edges']) > 0:
        all_new_edges_combined = torch.cat(state['all_new_edges'], dim=1)
        overall_metrics = calculate_edge_accuracy(all_new_edges_combined, fulldata.node_labels, device)
        print(f"\n>>> Overall Augmentation Accuracy: {overall_metrics['accuracy']:.4f}")
        print(f"    (Based on {all_new_edges_combined.size(1) // 2} total new edges)")
        print(f"    Total Label-1 edges: {total_label_1}")
        print(f"    Total Label-5 edges: {total_label_5}")
        print(f"    Total (Label-1 + Label-5): {total_label_15}")
    else:
        print(f"\n>>> No edges were added during training")
        print(f"    Total Label-1 edges: {total_label_1}")
        print(f"    Total Label-5 edges: {total_label_5}")
        print(f"    Total (Label-1 + Label-5): {total_label_15}")

    print(f"\nFinal Graph: {fulldata.edge_index.size(1) // 2} edges (from initial {fulldata.initial_edges})")

    edge_aug_id_cpu = state['edge_augmentation_id'].cpu().numpy()
    unique_ids, counts = np.unique(edge_aug_id_cpu, return_counts=True)
    print(f"\n--- Edge Distribution by Source ---")
    for aug_id, count in zip(unique_ids, counts):
        if aug_id == 0:
            print(f"  Original edges: {count // 2}")
        else:
            print(f"  Augmentation #{int(aug_id)}: {count // 2}")

    graph_filename = 'augmented_graph.txt'
    save_graph_structure(fulldata.edge_index, fulldata.edge_weight, graph_filename,
                        dataset_output_dir, state['edge_augmentation_id'])
    print(f"✅ Final augmented graph saved to {os.path.join(dataset_output_dir, graph_filename)}")

    if state['final_node_embedding'] is not None:
        z_filename = 'node_embeddings.txt'
        z_filepath = os.path.join(dataset_output_dir, z_filename)
        node_embedding_cpu = state['final_node_embedding'].cpu().numpy()
        np.savetxt(z_filepath, node_embedding_cpu, fmt='%.6f')
        print(f"✅ Node embeddings saved to {z_filepath}")

    print(f"✅ Training log saved to {log_file}")

    print(f"{'='*60}\n")


def save_graph_structure(edge_index, edge_weight, filename, save_dir, edge_aug_id=None):
    filepath = os.path.join(save_dir, filename)
    if isinstance(edge_index, torch.Tensor):
        src = edge_index[0].detach().cpu().numpy()
        dst = edge_index[1].detach().cpu().numpy()
    else:
        src, dst = edge_index[0], edge_index[1]

    if edge_weight is not None:
        if isinstance(edge_weight, torch.Tensor):
            w = edge_weight.detach().cpu().numpy()
        else:
            w = edge_weight
    else:
        w = np.ones_like(src, dtype=float)

    if edge_aug_id is not None:
        if isinstance(edge_aug_id, torch.Tensor):
            edge_aug_id = edge_aug_id.detach().cpu().numpy()
        data_to_save = np.stack((src, dst, w, edge_aug_id), axis=1)
        np.savetxt(filepath, data_to_save, fmt='%d %d %.6f %d')
    else:
        data_to_save = np.stack((src, dst, w), axis=1)
        np.savetxt(filepath, data_to_save, fmt='%d %d %.6f')


def train_process(dataset_name, gpu_id, args, output_root):
    device, logger, log_file, prev_stdout, prev_stderr, dataset_output_dir = setup_environment(
        dataset_name, gpu_id, args, output_root
    )

    train_data, test_data, fulldata = get_augdata(
        root=args.custom_root,
        name=dataset_name,
        test_ratio=args.test_ratio,
        edge_add_ratio=args.edge_add_ratio,
        device=device
    )
    print(f"\n--- Data Loaded [{dataset_name}] ---")
    print(f"Results will be saved to: {dataset_output_dir}\n")

    model = build_link_predictor(args, fulldata.num_features, device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = BCEWithLogitsLoss()

    state = {
        'best_test_auc': 0.0,
        'global_best_auc': 0.0,
        'best_epoch': 0,
        'patience_counter': 0,
        'first_augment_done': False,
        'first_augment_epoch': 0,
        'augmentation_count': 0,
        'z': None,
        'all_new_edges': [],
        'all_augmentation_accs': [],
        'edge_augmentation_id': torch.zeros(fulldata.edge_index.size(1), dtype=torch.long, device=device),
        'final_node_embedding': None,
        'node_enhancement_count': torch.zeros(fulldata.node_features.size(0), dtype=torch.long, device=device),
        'latest_pull_edge_index': train_data.edge_index,
        'latest_pull_edge_weight': torch.ones(train_data.edge_index.size(1), device=device),
    }

    print(f"Start Training with Structure Augmentation...")
    print(f"Training backbone: PULL train2Aug.py + {model.__class__.__name__}")
    print(f"PULL fixed expansion ratio per epoch: {args.pull_train_edge_ratio:.4f}")
    print(f"Augmentation strategy: {args.augment_strategy}")
    print(f"Strategy: Early stopping (patience={args.patience}) OR Max 100 epochs -> Augment -> Every {args.augment_interval} epochs")
    print(f"Total Augmentations: {args.num_augmentations}")
    print(f"Node Selection: {args.augment_strategy}")
    print(f"Min epochs before 1st augmentation: {args.min_epochs_before_augment}\n")

    epoch = 0
    while state['augmentation_count'] < args.num_augmentations:
        epoch += 1

        loss, z, pull_edge_index, pull_edge_weight = pull_train(
            model,
            optimizer,
            train_data,
            criterion,
            epoch,
            state['z'],
            edge_add_ratio=args.pull_train_edge_ratio,
            cumulative_add=False,
            max_inner_epochs=args.pull_inner_epochs,
            early_stop_patience=args.pull_inner_patience,
            min_delta=args.pull_inner_min_delta,
        )
        state['z'] = z
        state['latest_pull_edge_index'] = pull_edge_index
        state['latest_pull_edge_weight'] = pull_edge_weight

        test_auc = pull_test(test_data, train_data, model)

        if not state['first_augment_done']:
            if test_auc > state['best_test_auc']:
                state['best_test_auc'] = test_auc
                state['best_epoch'] = epoch
                state['patience_counter'] = 0
            else:
                state['patience_counter'] += 1

        should_augment, reason = check_aug(state, epoch, args)

        if should_augment:
            state['augmentation_count'] += 1

            print(f"\n{'='*60}")
            if reason == 'early_stop':
                print(f"Early stopping triggered at epoch {epoch} (Best AUC: {state['best_test_auc']:.4f} at epoch {state['best_epoch']})")
            elif reason == 'max_epoch':
                print(f"Max epochs (100) reached at epoch {epoch} (Best AUC: {state['best_test_auc']:.4f} at epoch {state['best_epoch']})")
            else:
                print(f"Interval augmentation at epoch {epoch}")
            print(f"Performing AUGMENTATION #{state['augmentation_count']}/{args.num_augmentations}...")
            print("Using decoupled augmentation: inference=PULL-expanded graph, output=base full graph")
            print(f"{'='*60}")

            fulldata, full_z = structure_augmentation(
                model,
                state,
                fulldata,
                epoch,
                args,
                device,
                inference_edge_index=state['latest_pull_edge_index'],
                inference_edge_weight=state['latest_pull_edge_weight'],
            )
            state['final_node_embedding'] = full_z

            train_data, test_data, test_pos_edge_index = resplit_augdata(
                fulldata, state['edge_augmentation_id'], args, device
            )

            print(f"    Graph updated: Total edges = {fulldata.edge_index.size(1) // 2}")
            print(f"    Data re-split: Train edges = {train_data.edge_index.size(1) // 2}, Test edges = {test_pos_edge_index.size(1)}")
            print(f"{'='*60}\n")

            if not state['first_augment_done']:
                state['first_augment_done'] = True
                state['first_augment_epoch'] = epoch

            if state['augmentation_count'] >= args.num_augmentations:
                break

        loss_value = loss.item() if hasattr(loss, 'item') else loss
        
        if not state['first_augment_done']:
            print(f'Epoch: {epoch:02d}, Loss: {loss_value:.4f}, Test AUC: {test_auc:.4f} [Best: {state["best_test_auc"]:.4f}, Patience: {state["patience_counter"]}/{args.patience}]')
        else:
            print(f'Epoch: {epoch:02d}, Loss: {loss_value:.4f}, Test AUC: {test_auc:.4f} [Aug #{state["augmentation_count"]}/{args.num_augmentations}]')

        if test_auc > state['global_best_auc']:
            state['global_best_auc'] = test_auc
            state['best_epoch'] = epoch

    finalize_and_save(
        state,
        fulldata,
        dataset_output_dir,
        log_file,
        epoch,
        device,
        dataset_name,
    )

    sys.stdout = prev_stdout
    sys.stderr = prev_stderr
    logger.close()


def main():
    t0 = time.time()
    args = parse_args()
    global_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    prev_stdout = sys.stdout
    prev_stderr = sys.stderr

    base_dir = os.path.dirname(args.custom_root)
    output_root = os.path.join(base_dir, 'AugOutput', global_timestamp)
    os.makedirs(output_root, exist_ok=True)
    
    main_log_file = os.path.join(output_root, 'main_log.txt')
    main_logger = Logger(main_log_file)

    print(f"\n=== Structure Augmentation Training ===")
    print(f"Output directory: {output_root}")
    print(f"Main log file: {main_log_file}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    log_main_run_info(main_logger, args, output_root, main_log_file)

    if not os.path.exists(args.custom_root):
        print(f"Error: Path {args.custom_root} does not exist.")
        sys.stdout = prev_stdout
        sys.stderr = prev_stderr
        main_logger.close()
        return

    datasets = sorted([d for d in os.listdir(args.custom_root) if os.path.isdir(os.path.join(args.custom_root, d))])

    if not datasets:
        print("No datasets found.")
        sys.stdout = prev_stdout
        sys.stderr = prev_stderr
        main_logger.close()
        return

    print(f"Found {len(datasets)} datasets in {args.custom_root}")

    configure_runtime_threads(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in args.gpu)
    num_local_gpus = len(args.gpu)
    effective_workers = max(1, args.max_workers)
    if torch.cuda.is_available():
        effective_workers = min(effective_workers, num_local_gpus)
    else:
        effective_workers = 1

    if effective_workers != args.max_workers:
        print(f"Adjusting max_workers from {args.max_workers} to {effective_workers} to avoid CPU oversubscription.")

    print(f"Max Workers: {effective_workers}, GPUs: {args.gpu} (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")
    print(f"Runtime CPU cap: torch intra/inter = {torch.get_num_threads()}/{torch.get_num_interop_threads()}, BLAS={os.environ.get('OMP_NUM_THREADS')}")

    sys.stdout = prev_stdout
    sys.stderr = prev_stderr

    print(f"Found {len(datasets)} datasets")
    print(f"Max Workers: {effective_workers}, GPUs: {args.gpu}")
    print(f"\nProcessing datasets...\n")

    if effective_workers == 1:
        for name in tqdm(datasets, desc="Processing", unit="dataset"):
            sys.stdout = main_logger
            sys.stderr = main_logger
            print(f"\n>>> Processing Dataset: {name}")
            sys.stdout = prev_stdout
            sys.stderr = prev_stderr

            train_process(name, 0, args, output_root)
    else:
        mp.set_start_method('spawn', force=True)

        tasks = []
        for i, name in enumerate(datasets):
            local_gpu_id = i % num_local_gpus
            tasks.append((name, local_gpu_id, args, output_root))

        with mp.Pool(processes=effective_workers) as pool:
            list(tqdm(pool.starmap(train_process, tasks), total=len(tasks), desc="Processing", unit="dataset"))

    elapsed = time.time() - t0
    print(f"\n=== All Datasets Completed ===")
    print(f"Total time: {elapsed:.2f}s ({elapsed/60:.1f}min)")
    print(f"Log file: {main_log_file}")
    print()

    sys.stdout = main_logger
    sys.stderr = main_logger
    print(f"\nTotal Elapsed: {elapsed:.2f}s ({elapsed/60:.1f}min)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    sys.stdout = prev_stdout
    sys.stderr = prev_stderr
    main_logger.close()


if __name__ == '__main__':
    main()
