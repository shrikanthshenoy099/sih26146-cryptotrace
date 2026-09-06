import sys
sys.path.append(".")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from graph.build_graph import load_cleaned_data, build_id_mappings
from graph.structural_features import build_networkx_graph, compute_wallet_structural_features

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def build_wallet_and_ip_features(df, wallet_to_idx, ip_to_idx):
    G = build_networkx_graph(df)
    wallet_struct = compute_wallet_structural_features(G)

    n_wallets = len(wallet_to_idx)
    wallet_features = torch.zeros((n_wallets, 3), dtype=torch.float)
    for wallet, idx in wallet_to_idx.items():
        feats = wallet_struct.get(wallet, {"in_degree": 0, "out_degree": 0, "pagerank": 0.0})
        wallet_features[idx] = torch.tensor(
            [feats["in_degree"], feats["out_degree"], feats["pagerank"]]
        )

    ip_counts = df["src_ip"].value_counts().to_dict()
    n_ips = len(ip_to_idx)
    ip_features = torch.zeros((n_ips, 1), dtype=torch.float)
    for ip, idx in ip_to_idx.items():
        ip_features[idx] = float(ip_counts.get(ip, 0))

    return wallet_features, ip_features


def load_graph_with_all_features():
    df = load_cleaned_data(path="data/processed/transactions_with_structural_features.csv")
    graph = torch.load("data/processed/transaction_graph.pt", weights_only=False)

    _, wallet_to_idx, tx_to_idx = build_id_mappings(df)
    ip_to_idx = {ip: i for i, ip in enumerate(sorted(set(df["src_ip"]) | set(df["dst_ip"])))}

    wallet_features, ip_features = build_wallet_and_ip_features(df, wallet_to_idx, ip_to_idx)
    graph["wallet"].x = wallet_features
    graph["ip"].x = ip_features

    return graph


def create_masks(graph, test_size=0.2, val_size=0.1, seed=SEED):
    y = graph["transaction"].y.numpy()
    n = len(y)
    indices = np.arange(n)

    train_idx, test_idx = train_test_split(indices, test_size=test_size, stratify=y, random_state=seed)
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_size / (1 - test_size), stratify=y[train_idx], random_state=seed
    )

    for name, idx in [("train_mask", train_idx), ("val_mask", val_idx), ("test_mask", test_idx)]:
        mask = torch.zeros(n, dtype=torch.bool)
        mask[idx] = True
        setattr(graph["transaction"], name, mask)

    return graph


class GraphSAGE(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.lin(x)


def build_model(graph, hidden_channels=32, out_channels=2):
    model = GraphSAGE(hidden_channels, out_channels)
    model = to_hetero(model, graph.metadata(), aggr="sum")
    return model


def train():
    graph = load_graph_with_all_features()
    graph = create_masks(graph)
    graph = graph.to(DEVICE)

    model = build_model(graph).to(DEVICE)

    with torch.no_grad():
        model(graph.x_dict, graph.edge_index_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    y = graph["transaction"].y
    train_mask = graph["transaction"].train_mask
    val_mask = graph["transaction"].val_mask
    test_mask = graph["transaction"].test_mask

    class_counts = torch.bincount(y[train_mask])
    class_weights = (class_counts.sum() / (class_counts.float() * len(class_counts))).to(DEVICE)
    print(f"Class weights [licit, illicit]: {class_weights.tolist()}\n")

    best_val_f1 = 0.0
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        out = model(graph.x_dict, graph.edge_index_dict)
        loss = F.cross_entropy(out["transaction"][train_mask], y[train_mask], weight=class_weights)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                out = model(graph.x_dict, graph.edge_index_dict)
                val_pred = out["transaction"][val_mask].argmax(dim=1).cpu().numpy()
                val_true = y[val_mask].cpu().numpy()
                val_f1 = f1_score(val_true, val_pred, pos_label=1, zero_division=0)
                val_p = precision_score(val_true, val_pred, pos_label=1, zero_division=0)
                val_r = recall_score(val_true, val_pred, pos_label=1, zero_division=0)
                print(f"Epoch {epoch:03d} | loss {loss.item():.4f} | "
                      f"val precision {val_p:.3f} recall {val_r:.3f} f1 {val_f1:.3f}")

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    torch.save(model.state_dict(), "model/gnn_model.pt")

    print(f"\nBest validation F1: {best_val_f1:.3f}")
    model.load_state_dict(torch.load("model/gnn_model.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        out = model(graph.x_dict, graph.edge_index_dict)
        test_pred = out["transaction"][test_mask].argmax(dim=1).cpu().numpy()
        test_true = y[test_mask].cpu().numpy()

    print("\nFinal test set evaluation:")
    print(classification_report(test_true, test_pred, target_names=["licit", "illicit"], zero_division=0))


if __name__ == "__main__":
    train()
