import os
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from .gnn_model import GATKhopEncoder
from .utils import (
    data_preprocessing, get_M, 
    subgraph_sample_by_point, subgraph_sample_randomly, 
    build_dataloader, evaluate
)
from .clustering import eva


def compute_initial_metrics(x, y, n_clusters):
    """
    Compute initial clustering metrics before training (using raw features).
    
    Args:
        x: node features (Tensor or ndarray)
        y: ground truth labels (ndarray)
        n_clusters: number of clusters
    
    Returns:
        dict: {'epoch': 'INITIAL', 'loss': None, 'auc': None, 'acc': float, ...}
    """
    if isinstance(x, torch.Tensor):
        raw_x = x.cpu().numpy()
    else:
        raw_x = x
    
    kmeans_init = KMeans(n_clusters=n_clusters, n_init=20, random_state=0).fit(raw_x)
    acc_init, nmi_init, ari_init, f1_init = eva(y, kmeans_init.labels_)
    
    return {
        'epoch': 'INITIAL',
        'loss': None,
        'auc': None,
        'acc': acc_init,
        'nmi': nmi_init,
        'ari': ari_init,
        'f1': f1_init
    }


def select_best_result(history, output_criterion, z_history, labels_history):
    """
    Select best epoch based on output criterion.
    
    Args:
        history: list of epoch metrics dicts
        output_criterion: 'min_loss', 'max_auc', 'max_acc', or 'last_epoch'
        z_history: dict mapping epoch -> embeddings
        labels_history: dict mapping epoch -> cluster labels
    
    Returns:
        best_epoch, best_z, best_labels
    """
    # Filter out INITIAL for min_loss (since loss is None)
    valid_history = [h for h in history if h['epoch'] != 'INITIAL']
    
    if output_criterion == 'min_loss':
        best_entry = min(valid_history, key=lambda h: h['loss'])
    elif output_criterion == 'max_auc':
        best_entry = max(valid_history, key=lambda h: h['auc'] if h['auc'] is not None else -1)
    elif output_criterion == 'max_acc':
        # Include INITIAL for max_acc
        best_entry = max(history, key=lambda h: h['acc'] if h['acc'] is not None else -1)
    elif output_criterion == 'last_epoch':
        # Keep embeddings/labels from the final training epoch
        best_entry = valid_history[-1]
    else:
        raise ValueError(f"Unknown output_criterion: {output_criterion}")
    
    best_epoch = best_entry['epoch']
    best_z = z_history.get(best_epoch)
    best_labels = labels_history.get(best_epoch)
    
    return best_epoch, best_z, best_labels


def train_batch_size(model, dataset, args, device, optimizer):
    """
    Full graph batch training strategy.
    
    Args:
        model: GNN encoder (GCN/GAT/SAGE/GIN/GAT_Khop)
        dataset: PyG Data object (must be preprocessed with data_preprocessing)
        args: parameter object containing:
            - max_epoch, n_clusters, output_criterion
            - gnn_model: model type string
            - loss_weight: whether to use weighted loss
        device: torch device
        optimizer: optimizer
    Returns:
        dict: {
            'z_best': ndarray (n_nodes, embedding_size),
            'labels_best': ndarray (n_nodes,),
            'history': list of epoch metrics,
            'best_epoch': int or 'INITIAL',
            'model': trained model
        }
    """
    model.train()
    is_gat_khop = isinstance(model, GATKhopEncoder) or args.gnn_model.upper() == 'GAT_KHOP'
    
    # Prepare data
    adj = dataset.adj.to(device)
    adj_label = dataset.adj_label.to(device)
    x = dataset.x.to(device) if isinstance(dataset.x, torch.Tensor) else torch.tensor(dataset.x, dtype=torch.float32, device=device)
    y = dataset.y.cpu().numpy() if isinstance(dataset.y, torch.Tensor) else dataset.y
    num_nodes = x.size(0)
    
    # For non-GAT_Khop models, prepare edge_index
    if not is_gat_khop:
        edge_index = (adj_label > 0.5).nonzero(as_tuple=False).t().contiguous()
    
    # Compute M matrix for GAT_Khop
    if is_gat_khop:
        M = get_M(adj).to(device)
        # Compute weights for weighted loss
        eye_mask = (1 - torch.eye(num_nodes, device=device))
        pos = (adj_label > 0.5).float() * eye_mask
        neg = (adj_label <= 0.5).float() * eye_mask
        num_pos = pos.sum()
        num_neg = neg.sum()
        w_pos = (num_neg / (num_pos + 1e-8)).clamp(max=50.0)
        weights = pos * w_pos + neg
    
    # History tracking
    epoch_metrics_history = []
    z_history = {}
    labels_history = {}
    
    # Compute initial metrics
    initial_metrics = compute_initial_metrics(x, y, args.n_clusters)
    epoch_metrics_history.append(initial_metrics)
    
    # Store initial features for potential selection
    z_history['INITIAL'] = x.cpu().numpy()
    kmeans_init = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=0).fit(z_history['INITIAL'])
    labels_history['INITIAL'] = kmeans_init.labels_
    
    # Training loop
    disable_tqdm = os.environ.get("TQDM_DISABLE", "0") == "1"
    for epoch in tqdm(range(args.max_epoch), desc="Training", disable=disable_tqdm):
        model.train()
        
        optimizer.zero_grad()
        
        if is_gat_khop:
            # GAT_Khop: returns (A_pred, z)
            A_pred, z = model(x, adj, M)
            y_pred = A_pred.view(-1)
            y_true = adj_label.view(-1)
            w = weights.view(-1)
            
            if getattr(args, 'loss_weight', True):
                loss = F.binary_cross_entropy(y_pred, y_true, weight=w, reduction='mean')
            else:
                loss = F.binary_cross_entropy(y_pred, y_true, reduction='mean')
        else:
            # Other GNN models: returns z only
            z = model(x, edge_index)
            
            # Dot product decoding (consistent with GAT_Khop)
            z_i = z.unsqueeze(1).expand(-1, num_nodes, -1)  # (N, N, d)
            z_j = z.unsqueeze(0).expand(num_nodes, -1, -1)  # (N, N, d)
            logits = (z_i * z_j).sum(dim=-1).view(-1)  # (N*N,)
            y_pred = torch.sigmoid(logits)
            
            # Construct full adjacency matrix labels
            adj_dense = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).float()
            y_true = adj_dense.view(-1)
            
            loss = F.binary_cross_entropy(y_pred, y_true, reduction='mean')
        
        loss.backward()
        optimizer.step()
        
        # Compute AUC
        with torch.no_grad():
            try:
                auc = roc_auc_score(y_true.cpu().numpy(), y_pred.detach().cpu().numpy())
            except ValueError:
                auc = 0.0
        
        # Clustering evaluation
        z_np = z.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=0).fit(z_np)
        acc, nmi, ari, f1 = eva(y, kmeans.labels_)
        
        # Record history
        epoch_metrics_history.append({
            'epoch': epoch,
            'loss': loss.item(),
            'auc': auc,
            'acc': acc,
            'nmi': nmi,
            'ari': ari,
            'f1': f1
        })
        
        # Store embeddings and labels
        z_history[epoch] = z_np
        labels_history[epoch] = kmeans.labels_
    
    # Select best result
    best_epoch, best_z, best_labels = select_best_result(
        epoch_metrics_history, args.output_criterion, z_history, labels_history
    )
    
    return {
        'z_best': best_z,
        'labels_best': best_labels,
        'history': epoch_metrics_history,
        'best_epoch': best_epoch,
        'model': model
    }


def train_subgraph_sampling(model, dataset, args, device, optimizer):
    """
    Subgraph sampling training strategy using DataLoader.
    
    Args:
        model: GNN encoder (GCN/GAT/SAGE/GIN/GAT_Khop)
        dataset: PyG Data object (must be preprocessed with data_preprocessing)
        args: parameter object containing:
            - max_epoch, n_clusters, output_criterion
            - gnn_model: model type string
            - subgraph_sample_method: 'by_point' or 'random'
            - subgraph_k: k-hop radius (default 3)
            - subgraph_stride: stride for by_point sampling (default 5)
            - batch_size: batch size for DataLoader (default 32)
            - loss_weight: whether to use weighted loss
        device: torch device
        optimizer: optimizer
    Returns:
        dict: same structure as train_batch_size
    """
    is_gat_khop = isinstance(model, GATKhopEncoder) or args.gnn_model.upper() == 'GAT_KHOP'
    
    # Prepare data
    adj = dataset.adj.to(device)
    adj_label = dataset.adj_label.to(device)
    x = dataset.x.to(device) if isinstance(dataset.x, torch.Tensor) else torch.tensor(dataset.x, dtype=torch.float32, device=device)
    y = dataset.y.cpu().numpy() if isinstance(dataset.y, torch.Tensor) else dataset.y
    num_nodes = x.size(0)
    
    # Sample subgraphs
    k = getattr(args, 'subgraph_k', 3)
    stride = getattr(args, 'subgraph_stride', 5)
    sample_method = getattr(args, 'subgraph_sample_method', 'by_point')
    batch_size = getattr(args, 'batch_size', 32)
    
    if sample_method == 'by_point':
        subgraphs = subgraph_sample_by_point(adj, adj_label, x, k=k, stride=stride, ensure_cover=True)
    else:  # random
        subgraphs = subgraph_sample_randomly(adj, adj_label, x, k=k)
    
    loader = build_dataloader(subgraphs, batch_size=batch_size, shuffle=True)
    
    # History tracking
    epoch_metrics_history = []
    z_history = {}
    labels_history = {}
    
    # Compute initial metrics
    initial_metrics = compute_initial_metrics(x, y, args.n_clusters)
    epoch_metrics_history.append(initial_metrics)
    
    # Store initial features
    z_history['INITIAL'] = x.cpu().numpy()
    kmeans_init = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=0).fit(z_history['INITIAL'])
    labels_history['INITIAL'] = kmeans_init.labels_
    
    # Training loop
    disable_tqdm = os.environ.get("TQDM_DISABLE", "0") == "1"
    for epoch in tqdm(range(args.max_epoch), desc="Training", disable=disable_tqdm):
        model.train()
        total_loss = 0.0
        epoch_y_pred = []
        epoch_y_true = []
        
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            if is_gat_khop:
                # Compute dense matrices for GAT_Khop
                adj_dense = to_dense_adj(batch.edge_index, max_num_nodes=batch.num_nodes).squeeze(0).float()
                M_dense_raw = to_dense_adj(batch.edge_index, edge_attr=batch.M_e, max_num_nodes=batch.num_nodes).squeeze(0).float()
                if M_dense_raw.dim() == 2:
                    M_denses = M_dense_raw.unsqueeze(0)
                elif M_dense_raw.dim() == 3:
                    # to_dense_adj with multi-dimensional edge_attr returns [N, N, t]
                    M_denses = M_dense_raw.permute(2, 0, 1).contiguous()
                else:
                    raise ValueError(f"Unsupported M_dense_raw shape: {tuple(M_dense_raw.shape)}")
                w_dense = to_dense_adj(batch.edge_index, edge_attr=batch.w_e, max_num_nodes=batch.num_nodes).squeeze(0)
                
                A_pred, z = model(batch.x, adj_dense, M_denses)
                
                y_pred = A_pred.view(-1)
                y_true = adj_dense.view(-1)
                w = w_dense.view(-1)
                
                if getattr(args, 'loss_weight', True):
                    loss = F.binary_cross_entropy(y_pred, y_true, weight=w, reduction='mean')
                else:
                    loss = F.binary_cross_entropy(y_pred, y_true, reduction='mean')
            else:
                # Other GNN models: dot product decoding (consistent with GAT_Khop)
                z = model(batch.x, batch.edge_index)
                
                # Dot product decoding
                z = F.normalize(z, p=2, dim=1)
                A_pred = torch.sigmoid(torch.matmul(z, z.t()))
                y_pred = A_pred.view(-1)
                    
                # Construct full adjacency matrix labels
                adj_dense = to_dense_adj(batch.edge_index, max_num_nodes=batch.num_nodes).squeeze(0).float()
                y_true = adj_dense.view(-1)
                
                loss = F.binary_cross_entropy(y_pred, y_true, reduction='mean')
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            epoch_y_pred.extend(y_pred.detach().cpu().numpy())
            epoch_y_true.extend(y_true.detach().cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        
        # Compute AUC
        try:
            auc = roc_auc_score(epoch_y_true, epoch_y_pred)
        except ValueError:
            auc = 0.0
        
        # Get full graph embeddings for clustering
        z_full = evaluate_full_graph(model, loader, device, is_gat_khop)
        z_np = z_full.numpy()
        
        # Clustering evaluation
        kmeans = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=0).fit(z_np)
        acc, nmi, ari, f1 = eva(y, kmeans.labels_)
        
        # Record history
        epoch_metrics_history.append({
            'epoch': epoch,
            'loss': avg_loss,
            'auc': auc,
            'acc': acc,
            'nmi': nmi,
            'ari': ari,
            'f1': f1
        })
        
        z_history[epoch] = z_np
        labels_history[epoch] = kmeans.labels_
    
    # Select best result
    best_epoch, best_z, best_labels = select_best_result(
        epoch_metrics_history, args.output_criterion, z_history, labels_history
    )
    
    return {
        'z_best': best_z,
        'labels_best': best_labels,
        'history': epoch_metrics_history,
        'best_epoch': best_epoch,
        'model': model
    }


def evaluate_full_graph(model, loader, device, is_gat_khop):
    """
    Evaluate model on full graph to get embeddings for all nodes.
    
    Args:
        model: GNN encoder
        loader: DataLoader with subgraphs
        device: torch device
        is_gat_khop: whether model is GAT_Khop
    
    Returns:
        z: full graph embeddings [N, d]
    """
    model.eval()
    with torch.no_grad():
        num_nodes = max([data.nodes.max().item() for data in loader.dataset]) + 1
        embed_dim = None
        z_sum = {}
        counts = {}
        
        for batch in loader:
            batch = batch.to(device)
            
            if is_gat_khop:
                adj_dense = to_dense_adj(batch.edge_index, max_num_nodes=batch.num_nodes).squeeze(0).float()
                M_dense_raw = to_dense_adj(batch.edge_index, edge_attr=batch.M_e, max_num_nodes=batch.num_nodes).squeeze(0).float()
                if M_dense_raw.dim() == 2:
                    M_denses = M_dense_raw.unsqueeze(0)
                elif M_dense_raw.dim() == 3:
                    M_denses = M_dense_raw.permute(2, 0, 1).contiguous()
                else:
                    raise ValueError(f"Unsupported M_dense_raw shape: {tuple(M_dense_raw.shape)}")
                _, batch_z = model(batch.x, adj_dense, M_denses)
            else:
                batch_z = model(batch.x, batch.edge_index)
            
            if embed_dim is None:
                embed_dim = batch_z.size(1)
            
            for i, nid in enumerate(batch.nodes.cpu().numpy()):
                nid = int(nid)
                if nid not in z_sum:
                    z_sum[nid] = batch_z[i].cpu()
                    counts[nid] = 1
                else:
                    z_sum[nid] += batch_z[i].cpu()
                    counts[nid] += 1
        
        z = torch.zeros((num_nodes, embed_dim))
        for nid, vec in z_sum.items():
            z[nid] = vec / counts[nid]
    
    return z


def sample_negative_edges(edge_index, num_nodes, num_neg_samples):
    """
    Sample negative edges (non-existing edges).
    
    Args:
        edge_index: existing edges [2, E]
        num_nodes: number of nodes
        num_neg_samples: number of negative samples
    
    Returns:
        neg_edge_index: negative edges [2, num_neg_samples]
    """
    device = edge_index.device
    
    # Create set of existing edges
    existing = set()
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        existing.add((src, dst))
        existing.add((dst, src))
    
    neg_edges = []
    while len(neg_edges) < num_neg_samples:
        src = torch.randint(0, num_nodes, (1,)).item()
        dst = torch.randint(0, num_nodes, (1,)).item()
        if src != dst and (src, dst) not in existing:
            neg_edges.append([src, dst])
            existing.add((src, dst))
            existing.add((dst, src))
    
    neg_edge_index = torch.tensor(neg_edges, device=device).t()
    return neg_edge_index
