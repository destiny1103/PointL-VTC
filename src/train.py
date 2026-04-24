from sklearn.metrics import roc_auc_score
import torch
from torch_geometric.utils import negative_sampling

@torch.no_grad()
def calculate_edge_accuracy(edge_index, node_labels, device):
    """
    Calculate the proportion of edges where the labels of nodes on both sides are the same, and the number of edges with labels 1 and 5. 
       
    Parameters:
        edge_index: [2, num_edges] The number of newly added edges
        node_labels: [num_nodes] The node labels

    Returns:
        dict: {
            'accuracy': float The accuracy of the newly added edges,
            'label_1_count': int The number of edges whose two sides are both labeled 1,
            'label_5_count': int The number of edges whose two sides are both labeled 5,
            'label_15_total': int The total number of edges labeled 1 and 5

            }
    """
    if edge_index.size(1) == 0:
        return {
            'accuracy': 0.0,
            'label_1_count': 0,
            'label_5_count': 0,
            'label_15_total': 0
        }
    
    src_labels = node_labels[edge_index[0]]
    dst_labels = node_labels[edge_index[1]]
    

    mask = edge_index[0] < edge_index[1]
    src_labels = src_labels[mask]
    dst_labels = dst_labels[mask]
    
    correct = (src_labels == dst_labels).sum().item()
    total = src_labels.size(0)
    accuracy = correct / total if total > 0 else 0.0
    
    label_1_mask = (src_labels == 1) & (dst_labels == 1)
    label_5_mask = (src_labels == 5) & (dst_labels == 5)
    
    label_1_count = label_1_mask.sum().item()
    label_5_count = label_5_mask.sum().item()
    label_15_total = label_1_count + label_5_count
    
    return {
        'accuracy': accuracy,
        'label_1_count': label_1_count,
        'label_5_count': label_5_count,
        'label_15_total': label_15_total
    }


def train(model, optimizer, train_data, criterion, epoch, z=None,
          edge_add_ratio=0.05, cumulative_add=True):
    """
    train the model.
    :param model: model to use.
    :param optimizer: optimizer for model.
    :param train_data: data to train.
    :param criterion: loss of the model.
    :param epoch: epoch to run.
    :param z: latent variable of edge potential.
    :param edge_add_ratio: edge expansion ratio used by decode_all.
    :param cumulative_add: if True, added edge count grows with epoch.
                           if False, each epoch adds a fixed ratio.
    """
    # initialize index and weights of positive edges
    if epoch == 1:
        pos_edge_index = train_data.edge_index
        pos_edge_weight = train_data.edge_label.new_ones(pos_edge_index.shape[1])
    else:
        decode_epoch = epoch if cumulative_add else 2
        pos_edge_index_add, pos_edge_weight_add = model.decode_all(
            z,
            train_data.edge_index,
            ratio=edge_add_ratio,
            epoch=decode_epoch
        )
        pos_edge_index, pos_edge_weight = model.merge_edge(train_data.edge_index,
                                                           torch.cat((train_data.edge_label, train_data.edge_label)),
                                                           pos_edge_index_add, pos_edge_weight_add)

        pos_edge_index = pos_edge_index.detach()


        pos_edge_weight = pos_edge_weight.detach()

    # train model with early stopping
    max_inner_epochs = 60
    early_stop_patience = 10
    min_delta = 1e-4
    best_inner_loss = float('inf')
    patience_count = 0
    
    for inner_epoch in range(1, max_inner_epochs + 1):
        model.train()
        optimizer.zero_grad()

        # We perform a new round of negative sampling for every training epoch:
        neg_edge_index = negative_sampling(
                edge_index=pos_edge_index, num_nodes=train_data.num_nodes,
                num_neg_samples=int(len(pos_edge_weight)), method='sparse')

        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        z = model.encode(train_data.x, pos_edge_index, pos_edge_weight)

        neg_edge_labels = train_data.edge_label.new_zeros(neg_edge_index.size(1))
        edge_label = torch.cat([pos_edge_weight, neg_edge_labels], dim=0)

        out = model.decode(z, edge_label_index).view(-1)

        if epoch == 1:
            loss = criterion(out, edge_label)
        else:
            neg_edge_index_a = negative_sampling(
                edge_index=pos_edge_index, num_nodes=train_data.num_nodes,
                num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

            edge_label_index_a = torch.cat([train_data.edge_label_index, neg_edge_index_a], dim=-1)
            z_a = model.encode(train_data.x, train_data.edge_index, None)

            neg_edge_labels_a = train_data.edge_label.new_zeros(neg_edge_index_a.size(1))
            edge_label_a = torch.cat([train_data.edge_label, neg_edge_labels_a], dim=0)

            out_a = model.decode(z_a, edge_label_index_a).view(-1)
            loss = criterion(out, edge_label) + criterion(out_a, edge_label_a)

        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        if current_loss < best_inner_loss - min_delta:
            best_inner_loss = current_loss
            patience_count = 0
        else:
            patience_count += 1
        
        if patience_count >= early_stop_patience:
            break

    return loss, z, pos_edge_index, pos_edge_weight

@torch.no_grad()
def test(data, train_data, model):
    """
    test the model
    :param data: real data for the ground truth
    :param train_data: data which part of edges are removed. used as input for the model.
    :param model: model to predict 
    """
    model.eval()
    z = model.encode(data.x, train_data.edge_index, None)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


@torch.no_grad()
def Inference(model, x, train_edge_index, val_edge_index, test_edge_index, ratio, epoch, device):
    """
    Inference function using model's native Structure Augmentation logic.
    
    params:
        model (nn.Module): The trained GCNLinkPredictor model.
        x (Tensor): Node feature matrix with shape [Num_Nodes, Num_Features].
        train_edge_index (Tensor): Edges from the training set (potentially augmented during training).
        val_edge_index (Tensor): Observed positive edges from the validation set.
        test_edge_index (Tensor): Observed positive edges from the test set.
        ratio (float): The hyperparameter controlling the ratio of edges to add.
        epoch (int): Current training epoch. The number of added edges scales dynamically with the epoch.
    return: 
        augmented_edge_index
        augmented_edge_weight
    """
    model.eval()
    
    base_edge_index = torch.cat([train_edge_index, val_edge_index, test_edge_index], dim=1)
    base_edge_weight = torch.ones(base_edge_index.size(1), device=device)  # 置信度为1.0
    
    z = model.encode(x, base_edge_index, None)
    
    add_edge_index, add_edge_weight = model.decode_all(z, base_edge_index, ratio, epoch)

    full_edge_index, full_edge_weight = model.merge_edge(
        base_edge_index, 
        base_edge_weight, 
        add_edge_index, 
        add_edge_weight
    )
    
    return full_edge_index, full_edge_weight, z
