import os
import os.path as osp
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.transforms as T

class MyGraphDataset(InMemoryDataset):
    def __init__(self, root, dataset_name, transform=None, pre_transform=None):
        self.dataset_name = dataset_name
        super(MyGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __repr__(self):
        return f"{self.dataset_name}()"

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.dataset_name, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.dataset_name, 'processed')

    @property
    def raw_file_names(self):
        return ['graph.txt', 'feature.txt', 'label.txt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        print(f"正在处理原始数据: {self.raw_dir} ...")
        
        edge_path = self.raw_paths[0]
        if not os.path.exists(edge_path):
            raise FileNotFoundError(f"找不到文件: {edge_path}")
            
        edge_df = pd.read_csv(edge_path, sep=' ', header=None)
        src = torch.tensor(edge_df[0].values, dtype=torch.long)
        dst = torch.tensor(edge_df[1].values, dtype=torch.long)
        weight = torch.tensor(edge_df[2].values, dtype=torch.float)
        edge_index = torch.stack([src, dst], dim=0)

        feat_path = self.raw_paths[1]
        feat_df = pd.read_csv(feat_path, sep='\t', header=None)
        
        group_0_indices = [
            0, 1, 2, 4,       
            29, 30, 32, 33, 34 
        ]
        
        valid_indices = [i for i in group_0_indices if i < len(feat_df.columns)]
        feat_df = feat_df.iloc[:, valid_indices]

        x = torch.tensor(feat_df.values, dtype=torch.float)

        label_path = self.raw_paths[2]
        label_df = pd.read_csv(label_path, sep=' ', header=None)
        y = torch.tensor(label_df.values, dtype=torch.long).squeeze()

        data = Data(x=x, edge_index=edge_index, edge_weight=weight, y=y)

        data_list = [data]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        print(f"Save the processed data to: {self.processed_paths[0]}")
        torch.save((data, slices), self.processed_paths[0])


def get_dataset(dataset_name, root_dir=None):
    datasets = MyGraphDataset(root_dir, dataset_name)
    return datasets


def get_augdata(root, name, test_ratio, edge_add_ratio, device):
    print(f"Loading Custom Dataset: {name} from {root}")
    
    dataset = MyGraphDataset(root=root, dataset_name=name)
    data = dataset[0]
    
    normalize_transform = T.NormalizeFeatures()
    data_full = normalize_transform(data)
    
    class FullData:
        def __init__(self, data, device):
            self.edge_index = data.edge_index.to(device)
            self.edge_weight = data.edge_weight.to(device) if data.edge_weight is not None else torch.ones(data.edge_index.size(1), device=device)
            self.node_labels = data.y.to(device)
            self.node_features = data.x.to(device)
            self.num_features = data.x.size(1)
            self.initial_edges = data.edge_index.size(1) // 2
            self.add_edges = int(self.initial_edges * edge_add_ratio)
    
    fulldata = FullData(data_full, device)
    
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(
            num_val=0.0,
            num_test=test_ratio,
            is_undirected=True,
            add_negative_train_samples=False
        )
    ])
    
    train_data, _, test_data = transform(data)
    
    print(f"Num nodes: {train_data.num_nodes}")
    print(f"Original edges: {fulldata.initial_edges} (before split)")
    print(f"Train edges: {train_data.edge_index.size(1) // 2}")
    print(f"Test edges: {test_data.edge_label_index.size(1) // 2}")
    print(f"Edges per augmentation: {fulldata.add_edges} ({edge_add_ratio*100:.0f}% of initial)")
    print(f"Unique labels: {torch.unique(train_data.y)}")
    
    return train_data, test_data, fulldata
