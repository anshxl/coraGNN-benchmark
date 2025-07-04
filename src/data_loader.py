import os
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.data import Data

def load_cora(root_dir: str = 'data') -> Data:
    """
    Downloads (if needed) the Cora citation dataset, applies feature normalization,
    and returns the single PyG Data object.
    
    Args:
        root_dir (str): Base folder to store datasets (default: "data").
    
    Returns:
        data (torch_geometric.data.Data): Graph object with x, edge_index, train/val/test masks, and y.
    """
    dataset_path = os.path.join(root_dir, 'Cora')

    # NormalizeFeature will row-normalize the feature matrix
    transform = T.NormalizeFeatures()
    dataset = Planetoid(root=dataset_path, name="Cora", transform=transform)
    data = dataset[0]
    data.num_classes = dataset.num_classes
    return data

if __name__ == "__main__":
    # Quick check
    data = load_cora()
    print(data)
    print(f"Num nodes: {data.num_nodes}, Num edges: {data.num_edges}")
    print(f"Train/Val/Test masks: {data.train_mask.sum().item()}/{data.val_mask.sum().item()}/{data.test_mask.sum().item()}")