import os
import psutil
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data
import mlflow
import yaml

from src.data_loader import load_cora

# Read config
def read_config(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Model Definition
class Net(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, 
                dropout, model: str = "gcn", gat_heads: int = 8):
        """
        A unified GNN that can act as GCN, GraphSAGE or GAT.

        Args:
            in_channels:  Size of each input node feature.
            hidden_channels: Size of hidden embedding.
            out_channels: Number of classes (output dim).
            dropout: Dropout probability.
            model: One of {"gcn", "sage", "gat"}.
            gat_heads: Number of attention heads (only used if model=="gat").
        """
        super().__init__()
        self.model = model.lower()
        self.dropout = dropout

        if self.model == "gcn":
            self.conv1 = GCNConv(in_feats, hidden_size)
            self.conv2 = GCNConv(hidden_size, num_classes)
        
        elif self.model == "sage":
            self.conv1 = SAGEConv(in_feats, hidden_size)
            self.conv2 = SAGEConv(hidden_size, num_classes)
        
        elif self.model == "gat":
            self.conv1 = GATConv(in_feats, hidden_size, heads=gat_heads, dropout=dropout)
            self.conv2 = GATConv(hidden_size * gat_heads, num_classes, heads=1, dropout=dropout)
        
        else:
            raise ValueError(f"Model must be one of ['gcn','sage','gat'], got '{model}'")

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x.log_softmax(dim=1)
    
# Training and evaluation functions
def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data)
    pred = out[mask].argmax(dim=1)
    acc = (pred == data.y[mask]).float().mean().item() 
    return acc

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # If using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main(config):
    # Unpack config
    root_dir = config["data"]["root_dir"]
    name = config["data"]["name"]
    model_type = config["model"]["type"]
    hidden_size = config["model"]["hidden_size"]
    dropout = config["model"]["dropout"]
    gat_heads = config["model"]["gat_heads"]
    lr = config["training"]["lr"]
    weight_decay = config["training"]["weight_decay"]
    epochs = config["training"]["epochs"]
    seed = config["training"]["seed"]
    interval = config["training"]["interval"]
    ckpt_path = config["training"]["ckpt_path_template"].format(model=model_type)
    experiment = config["mlflow"]["experiment"]

    # Reproducibility
    set_seed(seed)

    # Load data
    data = load_cora(root_dir=root_dir)

    # Model, optimizer
    model = Net(data.num_node_features, hidden_size, data.num_classes, 
                dropout, model_type, gat_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # MLflow setup
    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        mlflow.log_params({
            "hidden_size": hidden_size,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "seed": seed,
            "model": model_type,
            "gat_heads": gat_heads,
        })

        best_val_acc = 0.0
        for epoch in range(1, epochs + 1):
            # Reset GPU stats, if using CUDA
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            loss = train(model, data, optimizer)
            train_acc = evaluate(model, data, data.train_mask)
            val_acc = evaluate(model, data, data.val_mask)

            # Memory Logging
            # GPU peak memory usage
            if torch.cuda.is_available():
                peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            else:
                peak_mb = 0
            mlflow.log_metric("gpu_peak_mem_mb", peak_mb, step=epoch)

            # CPU memory usage
            rss = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)  # in MB
            mlflow.log_metric("cpu_rss_mb", rss, step=epoch)
            
            mlflow.log_metrics({
                "loss": loss,
                "train_acc": train_acc,
                "val_acc": val_acc
            }, step=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Save model checkpoint
                torch.save(model.state_dict(), ckpt_path)

            if epoch % interval == 0:
                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                      f"Train: {train_acc:.4f} | Val: {val_acc:.4f}")
        
        # Final test evaluation
        model.load_state_dict(torch.load(ckpt_path))
        test_acc = evaluate(model, data, data.test_mask)
        mlflow.log_metric("test_acc", test_acc)
        print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":

    # Read config
    config = read_config("configs/config.yaml")
    main(config)



