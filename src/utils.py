import numpy as np
import torch
import os
from sklearn.preprocessing import normalize
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch

class MyData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'label_edge_index':
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

def save_graph_structure(edge_index, edge_weight, epoch, save_dir, edge_aug_id=None):
    if isinstance(epoch, int):
        filename = os.path.join(save_dir, f'graph_inf_epoch_{epoch:03d}.txt')
    elif epoch.endswith('.txt'):
        filename = os.path.join(save_dir, epoch)
    else:
        filename = os.path.join(save_dir, f'graph_inf_epoch_{epoch}.txt')
    
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
            aug_id = edge_aug_id.detach().cpu().numpy()
        else:
            aug_id = edge_aug_id
    else:
        aug_id = np.zeros_like(src, dtype=int)

    data_to_save = np.stack((src, dst, w, aug_id), axis=1)
    
    np.savetxt(filename, data_to_save, fmt='%d %d %.6f %d')


def data_preprocessing(dataset):
    device = dataset.edge_index.device

    if hasattr(dataset, 'edge_weight') and dataset.edge_weight is not None:
        edge_values = dataset.edge_weight.to(device)
    else:
        edge_values = torch.ones(dataset.edge_index.shape[1], device=device)
    
    dataset.adj = torch.sparse_coo_tensor(
        dataset.edge_index.to(device), 
        edge_values,
        torch.Size([dataset.x.shape[0], dataset.x.shape[0]])
    ).to_dense()
    dataset.adj_label = dataset.adj

    dataset.adj += torch.eye(dataset.x.shape[0], device=device)
    
    row_sum = dataset.adj.sum(dim=1, keepdim=True)
    row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
    dataset.adj = dataset.adj / row_sum

    return dataset


def get_M(adj, t=2):
    if t < 1:
        raise ValueError("t must be >= 1")

    device = adj.device

    col_sum = adj.sum(dim=0, keepdim=True)
    col_sum = torch.where(col_sum == 0, torch.ones_like(col_sum), col_sum)
    tran_prob = adj / col_sum

    m_list = []
    current = tran_prob.clone()
    for _ in range(t):
        m_list.append(current)
        current = torch.mm(current, tran_prob)

    return torch.stack(m_list, dim=0).to(device)


def subgraph_sample_by_point(adj, adj_label, x, k=3, stride=5, ensure_cover=True, order=None):
    device = adj.device
    N = adj.size(0)
    if order is None:
        order = torch.arange(N, device=device)
    else:
        order = order.to(device)

    edge_index = (adj_label > 0.5).nonzero(as_tuple=False).t().contiguous()  # [2,E]
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]

    centers = order[::stride]
    covered = torch.zeros(N, dtype=torch.bool, device=device)
    subgraphs = []

    def make_one(center_idx: int):
        subset, ei_sub, mapping, _ = k_hop_subgraph(int(center_idx), k, edge_index, num_nodes=N, relabel_nodes=False)
        xs  = x[subset]
        adjs = adj[subset][:, subset]
        adjs_label = adj_label[subset][:, subset]

        Ms = get_M(adjs).to(device)

        n_local = adjs_label.size(0)
        eye_mask = (1 - torch.eye(n_local, device=device))
        pos = (adjs_label > 0.5).float() * eye_mask
        neg = (adjs_label <= 0.5).float() * eye_mask
        num_pos = pos.sum()
        num_neg = neg.sum()
        
        w_pos = (num_neg / (num_pos + 1e-8)).clamp(max=50.0)
        weights = pos * w_pos + neg

        subgraphs.append({
            'nodes': subset,
            'x': xs,
            'adj': adjs,
            'adj_label': adjs_label,
            'M': Ms,
            'weights': weights
        })
        covered[subset] = True

    for c in centers:
        make_one(int(c))

    if ensure_cover and (~covered).any():
        for c in torch.nonzero(~covered, as_tuple=False).view(-1).tolist():
            make_one(int(c))

    return subgraphs


def subgraph_sample_randomly(adj, adj_label, x, k=3):
    device = adj.device
    num_nodes = adj.size(0)

    edge_index = (adj_label > 0.5).nonzero(as_tuple=False).t().contiguous()
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]

    sampled_nodes = set()
    subgraphs = []

    generator = torch.Generator(device=device).manual_seed(42)
    all_nodes = torch.randperm(num_nodes, generator=generator, device=device)

    def make_one(center_idx: int):
        subset, ei_sub, mapping, _ = k_hop_subgraph(int(center_idx), k, edge_index, num_nodes=num_nodes, relabel_nodes=False)
        xs = x[subset]
        adjs = adj[subset][:, subset]
        adjs_label = adj_label[subset][:, subset]

        Ms = get_M(adjs).to(device)

        n_local = adjs_label.size(0)
        eye_mask = (1 - torch.eye(n_local, device=device))
        pos = (adjs_label > 0.5).float() * eye_mask
        neg = (adjs_label <= 0.5).float() * eye_mask
        num_pos = pos.sum()
        num_neg = neg.sum()
        
        w_pos = (num_neg / (num_pos + 1e-8)).clamp(max=50.0)
        weights = pos * w_pos + neg

        subgraphs.append({
            'nodes': subset,
            'x': xs,
            'adj': adjs,
            'adj_label': adjs_label,
            'M': Ms,
            'weights': weights
        })
        sampled_nodes.update(subset.tolist())

    for node in all_nodes:
        if node.item() not in sampled_nodes:
            make_one(node.item())
        if len(sampled_nodes) >= num_nodes:
            break

    return subgraphs


def to_tensor(arr, dtype=None):
    if arr is None:
        return None
    if isinstance(arr, torch.Tensor):
        return arr.clone().detach().to(dtype) if dtype else arr.clone().detach()
    return torch.tensor(arr, dtype=dtype)


def build_dataloader(subgraphs, batch_size=32, shuffle=True):
    graph_list = []
    for g in subgraphs:
        x = to_tensor(g["x"], torch.float)                  # [n, d]
        nodes = to_tensor(g["nodes"], torch.long)           # [n]
        adj = to_tensor(g["adj"], torch.float)              # [n, n]

        edge_index = (adj > 0).nonzero(as_tuple=False).t().contiguous()  # [2, E]
        row, col = edge_index
        edge_attr = adj[row, col]

        w_mat = to_tensor(g["weights"], torch.float) if g.get("weights", None) is not None else None  # [n, n] or None
        M_mat = to_tensor(g["M"], torch.float) if g.get("M", None) is not None else None              # [t, n, n] or [n, n] or None

        w_e = w_mat[row, col] if w_mat is not None else None # [E] or None
        if M_mat is not None:
            if M_mat.dim() == 2:
                M_e = M_mat[row, col].unsqueeze(-1)  # [E, 1]
            elif M_mat.dim() == 3:
                M_e = M_mat[:, row, col].t().contiguous()  # [E, t]
            else:
                raise ValueError(f"Unsupported M shape: {tuple(M_mat.shape)}")
        else:
            M_e = None

        adj_label = to_tensor(g.get("adj_label"), torch.float)
        label_edge_index = None
        if adj_label is not None:
             label_edge_index = (adj_label > 0.5).nonzero(as_tuple=False).t().contiguous()

        data = MyData(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            label_edge_index=label_edge_index,
            nodes=nodes,
            w_e=w_e,
            M_e=M_e,
            num_nodes=x.size(0),
        )
        graph_list.append(data)

    return DataLoader(graph_list, batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader, device, agg="mean"):
    model.eval()
    with torch.no_grad():
        num_nodes = max([data.nodes.max().item() for data in loader.dataset]) + 1
        embed_dim = None

        if agg == "mean":
            z_sum = {}
            counts = {}
        else:
            z = None

        for batch in loader:
            batch = batch.to(device)

            from torch_geometric.utils import to_dense_adj
            adj_dense = to_dense_adj(batch.edge_index, max_num_nodes=batch.num_nodes).squeeze(0).float()
            M_dense = to_dense_adj(batch.edge_index, edge_attr=batch.M_e, max_num_nodes=batch.num_nodes).squeeze(0).float()

            _, batch_z = model(batch.x, adj_dense, M_dense)  # [m, d]

            if embed_dim is None:
                embed_dim = batch_z.size(1)
                if agg == "last":
                    z = torch.zeros((num_nodes, embed_dim), device="cpu")

            if agg == "mean":
                for i, nid in enumerate(batch.nodes.cpu().numpy()):
                    nid = int(nid)
                    if nid not in z_sum:
                        z_sum[nid] = batch_z[i].cpu()
                        counts[nid] = 1
                    else:
                        z_sum[nid] += batch_z[i].cpu()
                        counts[nid] += 1
            else:  # last
                z[batch.nodes.cpu()] = batch_z.cpu()

        if agg == "mean":
            z = torch.zeros((num_nodes, embed_dim))
            for nid, vec in z_sum.items():
                z[nid] = vec / counts[nid]

    return z


def _normalize_labels(node_labels):
    if isinstance(node_labels, torch.Tensor) and node_labels.ndim > 1:
        return node_labels.argmax(dim=-1)
    return node_labels

def calculate_edge_accuracy(edge_index, node_labels, device=None):
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.tensor(edge_index)
    if not isinstance(node_labels, torch.Tensor):
        node_labels = torch.tensor(node_labels)

    edge_index = edge_index.to(node_labels.device)
    labels = _normalize_labels(node_labels).to(edge_index.device)

    edge_min = torch.min(edge_index[0], edge_index[1])
    edge_max = torch.max(edge_index[0], edge_index[1])
    undirected = torch.stack([edge_min, edge_max], dim=0)
    undirected = torch.unique(undirected, dim=1)

    src = undirected[0]
    dst = undirected[1]

    same_label = (labels[src] == labels[dst])
    total_edges = max(undirected.size(1), 1)
    accuracy = float(same_label.sum().item()) / total_edges

    label_1_count = int(((labels[src] == 1) & (labels[dst] == 1)).sum().item())
    label_5_count = int(((labels[src] == 5) & (labels[dst] == 5)).sum().item())
    label_15_total = int((((labels[src] == 1) & (labels[dst] == 5)) | ((labels[src] == 5) & (labels[dst] == 1))).sum().item())


    return {
        'accuracy': accuracy,
        'label_1_count': label_1_count,
        'label_5_count': label_5_count,
        'label_15_total': label_15_total
    }
