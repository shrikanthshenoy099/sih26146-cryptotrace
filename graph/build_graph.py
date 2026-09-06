import pandas as pd
import ast
import torch
from torch_geometric.data import HeteroData


def load_cleaned_data(path="data/processed/cleaned_transactions.csv"):
    df = pd.read_csv(path)
    # list-columns were saved as strings in CSV - convert back to real lists
    for col in ["input_addresses", "output_addresses", "input_amounts", "output_amounts"]:
        df[col] = df[col].apply(lambda v: ast.literal_eval(v) if isinstance(v, str) else v)
    return df
def build_id_mappings(df):
    all_ips = set(df["src_ip"]) | set(df["dst_ip"])
    ip_to_idx = {ip: i for i, ip in enumerate(sorted(all_ips))}

    all_wallets = set()
    for addrs in df["input_addresses"]:
        all_wallets.update(addrs)
    for addrs in df["output_addresses"]:
        all_wallets.update(addrs)
    wallet_to_idx = {w: i for i, w in enumerate(sorted(all_wallets))}

    tx_to_idx = {txid: i for i, txid in enumerate(df["txid"])}

    return ip_to_idx, wallet_to_idx, tx_to_idx
def build_edges(df, ip_to_idx, wallet_to_idx, tx_to_idx):
    wallet_sends_tx_src, wallet_sends_tx_dst = [], []
    tx_pays_wallet_src, tx_pays_wallet_dst = [], []
    ip_orig_tx_src, ip_orig_tx_dst = [], []

    for _, row in df.iterrows():
        tx_idx = tx_to_idx[row["txid"]]

        for addr in row["input_addresses"]:
            wallet_sends_tx_src.append(wallet_to_idx[addr])
            wallet_sends_tx_dst.append(tx_idx)

        for addr in row["output_addresses"]:
            tx_pays_wallet_src.append(tx_idx)
            tx_pays_wallet_dst.append(wallet_to_idx[addr])

        ip_orig_tx_src.append(ip_to_idx[row["src_ip"]])
        ip_orig_tx_dst.append(tx_idx)

    edges = {
        ("wallet", "sends", "transaction"): torch.tensor([wallet_sends_tx_src, wallet_sends_tx_dst], dtype=torch.long),
        ("transaction", "pays", "wallet"): torch.tensor([tx_pays_wallet_src, tx_pays_wallet_dst], dtype=torch.long),
        ("ip", "originates", "transaction"): torch.tensor([ip_orig_tx_src, ip_orig_tx_dst], dtype=torch.long),
    }
    return edges
def build_transaction_features(df, tx_to_idx):
    n_tx = len(tx_to_idx)
    features = torch.zeros((n_tx, 8), dtype=torch.float)
    labels = torch.zeros(n_tx, dtype=torch.long)

    for _, row in df.iterrows():
        idx = tx_to_idx[row["txid"]]
        total_in = sum(row["input_amounts"])
        total_out = sum(row["output_amounts"])
        features[idx] = torch.tensor([
            total_in,
            total_out,
            row["fee"],
            len(row["input_addresses"]) + len(row["output_addresses"]),
            row.get("avg_in_degree", 0.0),
            row.get("avg_out_degree", 0.0),
            row.get("avg_pagerank", 0.0),
            row.get("avg_ip_diversity", 0.0),
        ])
        labels[idx] = 1 if row["label"] == "illicit" else 0

    return features, labels

    
def build_graph(df):
    ip_to_idx, wallet_to_idx, tx_to_idx = build_id_mappings(df)
    edges = build_edges(df, ip_to_idx, wallet_to_idx, tx_to_idx)
    tx_features, tx_labels = build_transaction_features(df, tx_to_idx)

    data = HeteroData()

    data["ip"].num_nodes = len(ip_to_idx)
    data["wallet"].num_nodes = len(wallet_to_idx)
    data["transaction"].x = tx_features
    data["transaction"].y = tx_labels

    for edge_type, edge_index in edges.items():
        data[edge_type].edge_index = edge_index

    # PyG needs reverse edges for message passing in both directions
    data["transaction", "sent_by", "wallet"].edge_index = edges[("wallet", "sends", "transaction")].flip(0)
    data["wallet", "paid_by", "transaction"].edge_index = edges[("transaction", "pays", "wallet")].flip(0)
    data["transaction", "originated_by", "ip"].edge_index = edges[("ip", "originates", "transaction")].flip(0)

    return data, ip_to_idx, wallet_to_idx, tx_to_idx


if __name__ == "__main__":
    df = load_cleaned_data(path="data/processed/transactions_with_structural_features.csv")
    print(f"Loaded {len(df)} transactions")

    graph, ip_to_idx, wallet_to_idx, tx_to_idx = build_graph(df)

    print("\nGraph summary:")
    print(graph)
    print(f"\nNode counts: {len(ip_to_idx)} IPs, {len(wallet_to_idx)} wallets, {len(tx_to_idx)} transactions")
    print(f"Transaction label distribution: {torch.bincount(graph['transaction'].y)}")

    torch.save(graph, "data/processed/transaction_graph.pt")
    print("\nSaved graph to data/processed/transaction_graph.pt")