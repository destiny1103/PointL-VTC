from sklearn.metrics import roc_auc_score
import torch
from torch_geometric.utils import negative_sampling


def train(model, optimizer, train_data, criterion, epoch, z=None,
          edge_add_ratio=0.05, cumulative_add=True,
          max_inner_epochs=60, early_stop_patience=10, min_delta=1e-4):
    """
    Train link predictor for one outer epoch.

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
    train_x = train_data.x
    train_edge_index = train_data.edge_index
    train_edge_label = train_data.edge_label
    train_edge_label_index = train_data.edge_label_index
    num_nodes = train_data.num_nodes

    if epoch == 1:
        pos_edge_index = train_edge_index
        pos_edge_weight = train_data.edge_label.new_ones(pos_edge_index.shape[1])
    else:
        decode_epoch = epoch if cumulative_add else 2
        pos_edge_index_add, pos_edge_weight_add = model.decode_all(
            z,
            train_edge_index,
            ratio=edge_add_ratio,
            epoch=decode_epoch
        )
        pos_edge_index, pos_edge_weight = model.merge_edge(
            train_edge_index,
            torch.cat((train_edge_label, train_edge_label)),
            pos_edge_index_add,
            pos_edge_weight_add,
        )

        pos_edge_index = pos_edge_index.detach()
        pos_edge_weight = pos_edge_weight.detach()

    best_inner_loss = float('inf')
    patience_count = 0
    num_neg_samples = int(pos_edge_weight.numel())
    neg_edge_labels = train_edge_label.new_zeros(num_neg_samples)
    edge_label = torch.cat([pos_edge_weight, neg_edge_labels], dim=0)

    # Precompute regularization negatives once per outer epoch (epoch > 1)
    if epoch > 1:
        neg_edge_index_a = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=train_edge_label_index.size(1),
            method='sparse',
        )
        edge_label_index_a = torch.cat([train_edge_label_index, neg_edge_index_a], dim=-1)
        neg_edge_labels_a = train_edge_label.new_zeros(neg_edge_index_a.size(1))
        edge_label_a = torch.cat([train_edge_label, neg_edge_labels_a], dim=0)

    for _ in range(1, max_inner_epochs + 1):
        model.train()
        optimizer.zero_grad()

        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=num_neg_samples,
            method='sparse',
        )

        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        z = model.encode(train_x, pos_edge_index, pos_edge_weight)

        out = model.decode(z, edge_label_index).view(-1)

        if epoch == 1:
            loss = criterion(out, edge_label)
        else:
            z_a = model.encode(train_x, train_edge_index, None)
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
    Evaluate link prediction AUC.
    """
    model.eval()
    z = model.encode(data.x, train_data.edge_index, None)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


@torch.no_grad()
def Inference(model, x, train_edge_index, val_edge_index, test_edge_index, ratio, epoch, device):
    """
    Inference function using model's native structure augmentation logic.
    """
    model.eval()

    base_edge_index = torch.cat([train_edge_index, val_edge_index, test_edge_index], dim=1)
    base_edge_weight = torch.ones(base_edge_index.size(1), device=device)

    z = model.encode(x, base_edge_index, None)
    add_edge_index, add_edge_weight = model.decode_all(z, base_edge_index, ratio, epoch)

    full_edge_index, full_edge_weight = model.merge_edge(
        base_edge_index,
        base_edge_weight,
        add_edge_index,
        add_edge_weight,
    )

    return full_edge_index, full_edge_weight, z
