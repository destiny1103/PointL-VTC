
import torch
import torch.nn as nn
from torch_geometric.utils import degree
from torch_geometric.nn import  GCNConv, GATConv
import torch.nn.functional as F
import random
from .layer import GATLayer

def matmul_divide(z, chunk_size=100):
    """
    Matrix multiplication for the large size graph
    """
    num_chunks = z.size(0) // chunk_size
    prob_adj_parts = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        z_chunk = z[start_idx:end_idx, :]

        prob_adj_part = (z_chunk @ z.t()).sigmoid()
        prob_adj_parts.append(prob_adj_part.detach())

    if num_chunks * chunk_size < z.size(0):
        z_chunk = z[num_chunks * chunk_size:, :]
        prob_adj_part = (z_chunk @ z.t()).sigmoid()
        prob_adj_parts.append(prob_adj_part.detach())

    return torch.cat(prob_adj_parts, dim=0)

def top_k_edges(z, edge_idx, n_edge_add, degree, n_top_node):
    """
    Get top-k edges to add
    """
    top_degree = torch.topk(degree, n_top_node)
    top_degree_node_index = top_degree.indices

    prob_adj = matmul_divide(z)
    prob_adj[edge_idx[0, :], edge_idx[1, :]] = -1
    prob_adj = prob_adj - torch.diag(prob_adj.diag())
    prob_adj[top_degree_node_index, :] += 1

    edge_index = (prob_adj > 1).nonzero(as_tuple=False).t().contiguous()
    edge_weight = prob_adj[edge_index[0], edge_index[1]]



    # select top-k edge candidates
    edge_weight_topk = torch.topk(edge_weight, int(n_edge_add/2))
    edge_weight_idx = edge_weight_topk.indices
    edge_weight = edge_weight_topk.values - 1
    edge_index = torch.stack((edge_index[0, :][edge_weight_idx], edge_index[1, :][edge_weight_idx]), 0)

    edge_weight = torch.cat([edge_weight, edge_weight])
    edge_index = torch.cat([edge_index, torch.flip(edge_index, dims=[0])], dim=-1)

    return edge_index, edge_weight

def split_items(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n)]

def select_nodes_by_degree_desc(edge_index, num_nodes, num_select):
    deg_out = degree(edge_index[0], num_nodes)
    deg_in = degree(edge_index[1], num_nodes)
    degrees = deg_out + deg_in
    _, indices = torch.sort(degrees, descending=True)
    return indices[:num_select]

def select_nodes_by_degree_asc(edge_index, num_nodes, num_select):
    deg_out = degree(edge_index[0], num_nodes)
    deg_in = degree(edge_index[1], num_nodes)
    degrees = deg_out + deg_in
    _, indices = torch.sort(degrees, descending=False)
    return indices[:num_select]

def select_nodes_randomly(num_nodes, num_select):
    indices = torch.randperm(num_nodes)[:num_select]
    return indices

def top_k_edges_large(z, m_list, n_edge_add, num_chunks=100):
    """
    Get top-k edges to add (for ogb dataset)
    """
    m_chunks = torch.tensor(split_items(m_list.tolist(), num_chunks))
    indexs = []
    topks = []
    for m_indexs in m_chunks:
        m_indexs=m_indexs.to("cuda:"+str(z.get_device()))
        z_chunk = z[m_indexs,:]
        p = (z_chunk @ z.t()).sigmoid()
        new = torch.zeros_like(p).to("cuda:"+str(p.get_device()))
        mask  = p > 0.5
        new[mask] = p[mask]
        new = new.to_sparse()
        edge_weight_topk = torch.topk(new.values(), n_edge_add)
        topk_idx = edge_weight_topk.indices
        topk_values = edge_weight_topk.values
        edge_index = torch.stack((m_indexs[new.indices()[0,:][topk_idx]], new.indices()[1,:][topk_idx]), 0)
        indexs.append(edge_index)
        topks.append(topk_values)

    indexs = torch.cat(tuple(indexs), 1)
    topks = torch.cat(tuple(topks),0)
    final_topk = torch.topk(topks, int(n_edge_add/2))
    edge_index = torch.stack((indexs[0, :][final_topk.indices], indexs[1,:][final_topk.indices]),0)

    edge_weight = torch.cat([final_topk.values, final_topk.values])
    edge_index = torch.cat([edge_index, torch.flip(edge_index, dims=[0])], dim=-1)

    return edge_index, edge_weight

class GCNLinkPredictor(torch.nn.Module):
    """
    GCN based Link Predictor. we use 2 layers with 16 units for simplicity
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index, edge_weight=None):
        if edge_weight==None:
            x = self.conv1(x, edge_index).relu()
            return self.conv2(x, edge_index)
        else:
            x = self.conv1(x, edge_index, edge_weight).relu()
            return self.conv2(x, edge_index, edge_weight)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z, edge_idx, ratio, epoch):
        n_edge = edge_idx.shape[1]
        n_edge_add = int(n_edge*ratio*(epoch-1))

        # select top-k edge candidates with m
        m = 100
        num_nodes = edge_idx.max().item() + 1
        degrees = degree(edge_idx[0], num_nodes)
        edge_index, edge_weight = top_k_edges(z, edge_idx, n_edge_add, degrees, m)
        
        return edge_index, edge_weight
    
    def decode_all_v2(self, z, edge_index, num_add_edges, node_strategy='degree_asc', avoid_target_repeat=True, target_max_repeat=1):
        num_nodes = z.size(0)
        
        if node_strategy == 'degree_desc':
            selected_nodes = select_nodes_by_degree_desc(edge_index, num_nodes, num_add_edges)
        elif node_strategy == 'degree_asc':
            selected_nodes = select_nodes_by_degree_asc(edge_index, num_nodes, num_add_edges)
        elif node_strategy == 'random_sel':
            selected_nodes = select_nodes_randomly(num_nodes, num_add_edges)
        else:
            raise ValueError(f"Unknown strategy: {node_strategy}")
        
        z_selected = z[selected_nodes]  # [num_select, dim]
        prob = (z_selected @ z.t()).sigmoid()  # [num_select, num_nodes]
        
        edge_min = torch.min(edge_index[0], edge_index[1])
        edge_max = torch.max(edge_index[0], edge_index[1])
        existing_edges_tensor = torch.stack([edge_min, edge_max], dim=0)  # [2, num_edges]
        
        new_edges = []
        new_weights = []
        target_counts = torch.zeros(num_nodes, dtype=torch.long, device=z.device)
        
        for i, node_idx in enumerate(selected_nodes):
            node_idx = int(node_idx)
            node_probs = prob[i]
            
            sorted_probs, sorted_indices = torch.sort(node_probs, descending=True)
            
            for prob_val, target_node in zip(sorted_probs, sorted_indices):
                target_node = int(target_node)
                
                if prob_val.item() <= 0.6:
                    break
                
                if node_idx == target_node:
                    continue
                
                edge_min_val = min(node_idx, target_node)
                edge_max_val = max(node_idx, target_node)
                edge_exists = ((existing_edges_tensor[0] == edge_min_val) & 
                              (existing_edges_tensor[1] == edge_max_val)).any()
                if edge_exists:
                    continue

                if avoid_target_repeat:
                    if target_counts[target_node] >= target_max_repeat:
                        continue
                
                new_edges.append([node_idx, target_node])
                new_weights.append(float(prob_val))
                new_edge_tensor = torch.tensor([[edge_min_val], [edge_max_val]], 
                                               dtype=torch.long, device=z.device)
                existing_edges_tensor = torch.cat([existing_edges_tensor, new_edge_tensor], dim=1)
                if avoid_target_repeat:
                    target_counts[target_node] += 1
                break
        
        if len(new_edges) > 0:
            new_edge_index = torch.tensor(new_edges, device=z.device).t()  # [2, num]
            new_edge_weight = torch.tensor(new_weights, device=z.device)
            
            new_edge_index = torch.cat([
                new_edge_index, 
                torch.stack([new_edge_index[1], new_edge_index[0]])
            ], dim=1)
            new_edge_weight = torch.cat([new_edge_weight, new_edge_weight])
        else:
            new_edge_index = torch.empty((2, 0), device=z.device, dtype=torch.long)
            new_edge_weight = torch.empty(0, device=z.device)
        
        return new_edge_index, new_edge_weight

    def merge_edge(self, edge_index, edge_weight, edge_index_add, edge_weight_add):
        edge_index = torch.cat((edge_index, edge_index_add), 1)
        edge_weight = torch.cat((edge_weight, edge_weight_add))

        return edge_index, edge_weight


class GATLinkPredictor(torch.nn.Module):
    """
    GAT based Link Predictor with the same interface as GCNLinkPredictor.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=1,
        )
        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=1,
        )
        self.dropout = dropout

    def _edge_attr(self, edge_weight):
        if edge_weight is None:
            return None
        return edge_weight.view(-1, 1)

    def encode(self, x, edge_index, edge_weight=None):
        edge_attr = self._edge_attr(edge_weight)
        x = self.conv1(x, edge_index, edge_attr=edge_attr).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index, edge_attr=edge_attr)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z, edge_idx, ratio, epoch):
        n_edge = edge_idx.shape[1]
        n_edge_add = int(n_edge * ratio * (epoch - 1))

        m = 100
        num_nodes = edge_idx.max().item() + 1
        degrees = degree(edge_idx[0], num_nodes)
        edge_index, edge_weight = top_k_edges(z, edge_idx, n_edge_add, degrees, m)

        return edge_index, edge_weight

    def merge_edge(self, edge_index, edge_weight, edge_index_add, edge_weight_add):
        edge_index = torch.cat((edge_index, edge_index_add), 1)
        edge_weight = torch.cat((edge_weight, edge_weight_add))

        return edge_index, edge_weight


def dense_adj_to_edge_index(adj: torch.Tensor, threshold: float = 0.0):
    assert adj.dim() == 2 and adj.size(0) == adj.size(1), "adj should be NxN"
    if threshold > 0:
        mask = adj > threshold
    else:
        mask = adj != 0
    idx = mask.nonzero(as_tuple=False).t().contiguous()  # (2, E)
    w = adj[mask].contiguous()                           # (E,)
    return idx, w


def dot_product_decode(Z: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(Z @ Z.t())


class GAT(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha):
        super(GAT, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv1 = GATLayer(num_features, hidden_size, alpha)
        self.conv2 = GATLayer(hidden_size, embedding_size, alpha)

    def forward(self, x, adj, M):
        h = self.conv1(x, adj, M)
        h = self.conv2(h, adj, M)

        z = F.normalize(h, p=2, dim=1)
        A_pred = self.dot_product_decode(z)
        return A_pred, z

    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred

    def dot_product_logits(self, Z):
        return torch.matmul(Z, Z.t())


class PyGGATNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.5,
                 attn_dropout=0.6, add_self_loops=True, bias=True):
        super().__init__()
        self.conv1 = GATConv(
            in_channels, hidden_channels, heads=heads,
            dropout=attn_dropout, add_self_loops=add_self_loops, bias=bias, concat=True
        )
        self.conv2 = GATConv(
            hidden_channels * heads, out_channels, heads=1,
            dropout=attn_dropout, add_self_loops=add_self_loops, bias=bias, concat=False
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, M: torch.Tensor = None):
        edge_index, edge_weight = dense_adj_to_edge_index(adj)
        if M is not None:
            _, m_w = dense_adj_to_edge_index(M)
            edge_weight = edge_weight * m_w

        h = self.conv1(x, edge_index, edge_weight)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_weight)

        z = F.normalize(h, p=2, dim=1)
        A_pred = dot_product_decode(z)
        return A_pred, z


class PyGGCNNet(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 dropout=0.5,
                 add_self_loops=True,
                 bias=True):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=add_self_loops, bias=bias, normalize=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, add_self_loops=add_self_loops, bias=bias,
                             normalize=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, M: torch.Tensor = None):
        edge_index, edge_weight = dense_adj_to_edge_index(adj)
        if M is not None:
            _, m_w = dense_adj_to_edge_index(M)
            edge_weight = edge_weight * m_w

        h = self.conv1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_weight)

        z = F.normalize(h, p=2, dim=1)
        A_pred = dot_product_decode(z)
        return A_pred, z
