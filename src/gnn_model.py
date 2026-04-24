import math
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv
from torch_geometric.utils import degree


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.convs = nn.ModuleList()
        if num_layers <= 1:
            self.convs.append(GCNConv(in_channels, out_channels))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs[:-1]:
            if edge_weight is None:
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if edge_weight is None:
            x = self.convs[-1](x, edge_index)
        else:
            x = self.convs[-1](x, edge_index, edge_weight)
        return x


class SAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.convs = nn.ModuleList()
        if num_layers <= 1:
            self.convs.append(SAGEConv(in_channels, out_channels))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, heads):
        super().__init__()
        self.convs = nn.ModuleList()
        if num_layers <= 1:
            self.convs.append(GATConv(in_channels, out_channels, heads=heads, concat=False))
        else:
            self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))
            self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.lins = nn.ModuleList()
        if num_layers <= 1:
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
    

class GINEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, mlp_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        if num_layers <= 1:
            mlp = MLP(in_channels, hidden_channels, out_channels, mlp_layers, dropout)
            self.convs.append(GINConv(mlp))
        else:
            mlp = MLP(in_channels, hidden_channels, hidden_channels, mlp_layers, dropout)
            self.convs.append(GINConv(mlp))
            for _ in range(num_layers - 2):
                mlp = MLP(hidden_channels, hidden_channels, hidden_channels, mlp_layers, dropout)
                self.convs.append(GINConv(mlp))
            mlp = MLP(hidden_channels, hidden_channels, out_channels, mlp_layers, dropout)
            self.convs.append(GINConv(mlp))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


# ============ GAT_Khop Encoder (from SubGraph.py) ============
class GATKhopLayer(nn.Module):
    """
    GAT layer with k-hop topology importance matrix M
    Based on SubGraph.py implementation
    """
    def __init__(self, in_features, out_features, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)

        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj, M, concat=True):
        """
        Args:
            x: node features [N, in_features]
            adj: dense adjacency matrix [N, N]
            M: k-hop topology importance matrix [N, N]
            concat: whether to apply ELU activation
        """
        h = torch.mm(x, self.W)

        attn_for_self = torch.mm(h, self.a_self)      # (N, 1)
        attn_for_neighs = torch.mm(h, self.a_neighs)  # (N, 1)
        attn_dense = attn_for_self + torch.transpose(attn_for_neighs, 0, 1)
        attn_dense = torch.mul(attn_dense, M)
        attn_dense = self.leakyrelu(attn_dense)       # (N, N)

        zero_vec = -9e15 * torch.ones_like(adj)
        adj = torch.where(adj > 0, attn_dense, zero_vec)
        attention = F.softmax(adj, dim=1)
        h_prime = torch.matmul(attention, h)

        if concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GATKhopEncoder(nn.Module):
    """
    GAT encoder with k-hop topology importance (from SubGraph.py GAT model)
    
    This model:
    - Takes dense adjacency matrix and M matrix as input
    - Returns (A_pred, z) tuple where A_pred is edge prediction and z is node embeddings
    - Used for representation learning with edge reconstruction task
    """
    def __init__(self, in_channels, hidden_channels, out_channels, alpha=0.2):
        super().__init__()
        self.conv1 = GATKhopLayer(in_channels, hidden_channels, alpha)
        self.conv2 = GATKhopLayer(hidden_channels, out_channels, alpha)

    def _to_M_collection(self, M_denses, num_nodes):
        """
        Normalize M input into shape [t, N, N].
        Accepts [N,N] / [t,N,N] / [N,N,t].
        """
        if not isinstance(M_denses, torch.Tensor):
            raise TypeError("M_denses must be a torch.Tensor")

        if M_denses.dim() == 2:
            return M_denses.unsqueeze(0)

        if M_denses.dim() == 3:
            if M_denses.size(1) == num_nodes and M_denses.size(2) == num_nodes:
                return M_denses
            if M_denses.size(0) == num_nodes and M_denses.size(1) == num_nodes:
                return M_denses.permute(2, 0, 1).contiguous()

        raise ValueError(f"Unsupported M_denses shape: {tuple(M_denses.shape)}")

    def forward(self, x, adj_dense, M_denses):
        """
        Args:
            x: node features [N, in_channels]
            adj_dense: dense adjacency matrix [N, N]
            M_denses: k-hop topology importance matrix collection
                     supports [N, N] / [t, N, N] / [N, N, t]
        
        Returns:
            A_pred: edge prediction matrix [N, N]
            z: normalized node embeddings [N, out_channels]
        """
        num_nodes = adj_dense.size(0)
        M_collection = self._to_M_collection(M_denses, num_nodes)

        h_list = []
        for M_dense in M_collection:
            h = self.conv1(x, adj_dense, M_dense)
            h = self.conv2(h, adj_dense, M_dense)
            h_list.append(h)

        h_mean = torch.stack(h_list, dim=0).mean(dim=0)
        z = F.normalize(h_mean, p=2, dim=1)
        A_pred = torch.sigmoid(torch.matmul(z, z.t()))
        
        return A_pred, z


def build_encoder(model_name, in_channels, hidden_channels, out_channels, num_layers, dropout,
                  gat_heads=1, gin_mlp_layers=2, alpha=0.2):
    """
    Build GNN encoder by model name.
    
    Args:
        model_name: GCN, GAT, SAGE, GIN, or GAT_KHOP
        in_channels: input feature dimension
        hidden_channels: hidden layer dimension
        out_channels: output embedding dimension
        num_layers: number of GNN layers (not used for GAT_KHOP)
        dropout: dropout rate (not used for GAT_KHOP)
        gat_heads: number of attention heads for GAT
        gin_mlp_layers: number of MLP layers for GIN
        alpha: LeakyReLU negative slope for GAT_KHOP
    
    Returns:
        GNN encoder module
    """
    model_name = model_name.upper()
    if model_name == 'GCN':
        return GCNEncoder(in_channels, hidden_channels, out_channels, num_layers, dropout)
    if model_name == 'GAT':
        return GATEncoder(in_channels, hidden_channels, out_channels, num_layers, dropout, gat_heads)
    if model_name == 'SAGE':
        return SAGEEncoder(in_channels, hidden_channels, out_channels, num_layers, dropout)
    if model_name == 'GIN':
        return GINEncoder(in_channels, hidden_channels, out_channels, num_layers, dropout, gin_mlp_layers)
    if model_name == 'GAT_KHOP':
        return GATKhopEncoder(in_channels, hidden_channels, out_channels, alpha)
    raise ValueError(f"Unknown GNN model: {model_name}")


class MLPDecoder(nn.Module):
    """MLP decoder (probability output)."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.lins = nn.ModuleList()
        if num_layers <= 1:
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x).view(-1)


def dot_product_logits(z, edge_index):
    return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)


def dot_product_prob(z, edge_index):
    return torch.sigmoid(dot_product_logits(z, edge_index))


def decode_(z, edge_index, decoder_type='dot', mlp_decoder=None, return_logits=False):
    if decoder_type == 'dot':
        logits = dot_product_logits(z, edge_index)
        return logits if return_logits else torch.sigmoid(logits)
    if decoder_type == 'mlp':
        if mlp_decoder is None:
            raise ValueError('mlp_decoder is required for decoder_type="mlp"')
        probs = mlp_decoder(z[edge_index[0]], z[edge_index[1]])
        if not return_logits:
            return probs
        eps = 1e-12
        probs = torch.clamp(probs, eps, 1 - eps)
        return torch.log(probs / (1 - probs))
    raise ValueError(f"Unknown decoder_type: {decoder_type}")


def decode_prob(z, edge_index, decoder_type='dot', mlp_decoder=None):
    return decode_(z, edge_index, decoder_type=decoder_type, mlp_decoder=mlp_decoder, return_logits=False)

def decode_logits(z, edge_index, decoder_type='dot', mlp_decoder=None):
    return decode_(z, edge_index, decoder_type=decoder_type, mlp_decoder=mlp_decoder, return_logits=True)


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
    return torch.randperm(num_nodes)[:num_select]


def select_nodes_by_interval(num_nodes, num_select):
    if num_select >= num_nodes:
        all_nodes = torch.arange(num_nodes)
        remainder = num_select - num_nodes
        if remainder > 0:
            interval = max(1, num_nodes // remainder)
            extra_nodes = torch.arange(0, num_nodes, interval)[:remainder]
            selected = torch.cat([all_nodes, extra_nodes])
        else:
            selected = all_nodes
        return selected[:num_select]
    else:
        interval = num_nodes / num_select
        indices = torch.tensor([int(i * interval) for i in range(num_select)])
        return indices


def _select_nodes(edge_index, num_nodes, num_select, node_strategy, node_enhancement_count=None):
    if node_strategy == 'degree_desc':
        return select_nodes_by_degree_desc(edge_index, num_nodes, num_select)
    if node_strategy == 'degree_asc':
        return select_nodes_by_degree_asc(edge_index, num_nodes, num_select)
    if node_strategy == 'random_sel':
        return select_nodes_randomly(num_nodes, num_select)
    if node_strategy == 'interval':
        return select_nodes_by_interval(num_nodes, num_select)
    raise ValueError(f"Unknown node_strategy: {node_strategy}")


def _normalized_undirected_edge_ids(edge_index, num_nodes):
    edge_min = torch.minimum(edge_index[0], edge_index[1])
    edge_max = torch.maximum(edge_index[0], edge_index[1])
    return edge_min * num_nodes + edge_max

def augment_edges_dot(z, edge_index, num_add_edges, node_strategy='degree_asc',
                      avoid_target_repeat=True, target_max_repeat=1, prob_threshold=0.65,
                      node_enhancement_count=None):
    num_nodes = z.size(0)
    selected_nodes = _select_nodes(
        edge_index, num_nodes, num_add_edges, node_strategy, node_enhancement_count
    ).to(z.device)

    existing_edge_ids = torch.unique(_normalized_undirected_edge_ids(edge_index, num_nodes))

    # Batch compute probabilities for all selected nodes at once on GPU
    z_selected = z[selected_nodes]  # [num_select, dim]
    probs_matrix = torch.sigmoid(z_selected @ z.t())  # [num_select, num_nodes]

    # Batch sort on GPU and keep candidate filtering on tensor logic.
    sorted_probs, sorted_indices = torch.sort(probs_matrix, dim=1, descending=True)

    new_edges = []
    new_weights = []
    target_counts = torch.zeros(num_nodes, dtype=torch.long, device=z.device)
    
    should_track_enhancement = node_enhancement_count is not None and node_strategy in ['random_sel', 'interval']

    for i, node_idx in enumerate(selected_nodes):
        row_probs = sorted_probs[i]
        row_indices = sorted_indices[i]
        source_nodes = torch.full_like(row_indices, node_idx)
        candidate_edge_ids = _normalized_undirected_edge_ids(
            torch.stack([source_nodes, row_indices], dim=0), num_nodes
        )

        valid_mask = row_probs > prob_threshold
        valid_mask &= row_indices != node_idx
        valid_mask &= ~torch.isin(candidate_edge_ids, existing_edge_ids)

        if avoid_target_repeat:
            valid_mask &= target_counts[row_indices] < target_max_repeat

        valid_positions = valid_mask.nonzero(as_tuple=False)
        if valid_positions.numel() == 0:
            continue

        first_valid = valid_positions[0, 0]
        target_node = row_indices[first_valid]
        prob_val = row_probs[first_valid]

        new_edges.append(torch.stack([node_idx, target_node]))
        new_weights.append(prob_val)
        existing_edge_ids = torch.cat([existing_edge_ids, candidate_edge_ids[first_valid:first_valid + 1]])

        if avoid_target_repeat:
            target_counts[target_node] += 1
        
        if should_track_enhancement:
            node_enhancement_count[node_idx] += 1

    if len(new_edges) == 0:
        return torch.empty((2, 0), device=z.device, dtype=torch.long), torch.empty(0, device=z.device), node_enhancement_count

    new_edge_index = torch.stack(new_edges, dim=1)
    new_edge_weight = torch.stack(new_weights)

    new_edge_index = torch.cat([new_edge_index, torch.stack([new_edge_index[1], new_edge_index[0]])], dim=1)
    new_edge_weight = torch.cat([new_edge_weight, new_edge_weight])

    return new_edge_index, new_edge_weight, node_enhancement_count


def augment_edges_mlp(z, edge_index, num_add_edges, mlp_decoder, node_strategy='degree_asc',
                      avoid_target_repeat=True, target_max_repeat=1, prob_threshold=0.65,
                      node_enhancement_count=None):
    num_nodes = z.size(0)
    selected_nodes = _select_nodes(edge_index, num_nodes, num_add_edges, node_strategy, node_enhancement_count)
    existing_edges_set = set()
    for i in range(edge_index.size(1)):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        edge = (min(src, dst), max(src, dst))
        existing_edges_set.add(edge)
    new_edges = []
    new_weights = []
    target_counts = torch.zeros(num_nodes, dtype=torch.long, device=z.device)
    
    should_track_enhancement = node_enhancement_count is not None and node_strategy in ['random_sel', 'interval']

    for node_idx in selected_nodes.tolist():
        node_emb = z[node_idx].unsqueeze(0).repeat(num_nodes, 1)
        probs = mlp_decoder(node_emb, z)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        edge_added = False
        for prob_val, target_node in zip(sorted_probs, sorted_indices):
            target_node = int(target_node)
            if prob_val.item() <= prob_threshold:
                break
            if node_idx == target_node:
                continue
            edge = (min(node_idx, target_node), max(node_idx, target_node))
            if edge in existing_edges_set:
                continue
            if avoid_target_repeat and target_counts[target_node] >= target_max_repeat:
                continue

            new_edges.append([node_idx, target_node])
            new_weights.append(float(prob_val))
            existing_edges_set.add(edge)
            if avoid_target_repeat:
                target_counts[target_node] += 1
            edge_added = True
            break
        
        if edge_added and should_track_enhancement:
            node_enhancement_count[node_idx] += 1

    if len(new_edges) == 0:
        return torch.empty((2, 0), device=z.device, dtype=torch.long), torch.empty(0, device=z.device), node_enhancement_count

    new_edge_index = torch.tensor(new_edges, device=z.device).t()
    new_edge_weight = torch.tensor(new_weights, device=z.device)

    new_edge_index = torch.cat([new_edge_index, torch.stack([new_edge_index[1], new_edge_index[0]])], dim=1)
    new_edge_weight = torch.cat([new_edge_weight, new_edge_weight])

    return new_edge_index, new_edge_weight, node_enhancement_count
