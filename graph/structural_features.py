import pandas as pd
import networkx as nx
import ast


def load_cleaned_data(path="data/processed/cleaned_transactions.csv"):
    df = pd.read_csv(path)
    for col in ["input_addresses", "output_addresses", "input_amounts", "output_amounts"]:
        df[col] = df[col].apply(lambda v: ast.literal_eval(v) if isinstance(v, str) else v)
    return df


def build_networkx_graph(df):
    """Builds a simple directed graph: wallet -> wallet, via transactions, for structural analysis."""
    G = nx.DiGraph()

    for _, row in df.iterrows():
        for in_addr in row["input_addresses"]:
            for out_addr in row["output_addresses"]:
                G.add_edge(in_addr, out_addr, txid=row["txid"])

    return G
def compute_wallet_structural_features(G):
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    pagerank = nx.pagerank(G, alpha=0.85)

    features = {}
    for node in G.nodes():
        features[node] = {
            "in_degree": in_degree.get(node, 0),
            "out_degree": out_degree.get(node, 0),
            "pagerank": pagerank.get(node, 0.0),
        }
    return features
def compute_wallet_ip_diversity(df):
    wallet_ips = {}
    for _, row in df.iterrows():
        for addr in row["input_addresses"] + row["output_addresses"]:
            wallet_ips.setdefault(addr, set()).add(row["src_ip"])

    return {wallet: len(ips) for wallet, ips in wallet_ips.items()}
def attach_transaction_structural_features(df, wallet_features, ip_diversity):
    def _avg_feature(addrs, key):
        vals = [wallet_features.get(a, {}).get(key, 0) for a in addrs]
        return sum(vals) / len(vals) if vals else 0.0

    def _avg_ip_diversity(addrs):
        vals = [ip_diversity.get(a, 0) for a in addrs]
        return sum(vals) / len(vals) if vals else 0.0

    df = df.copy()
    all_addrs = df["input_addresses"] + df["output_addresses"]

    df["avg_in_degree"] = all_addrs.apply(lambda addrs: _avg_feature(addrs, "in_degree"))
    df["avg_out_degree"] = all_addrs.apply(lambda addrs: _avg_feature(addrs, "out_degree"))
    df["avg_pagerank"] = all_addrs.apply(lambda addrs: _avg_feature(addrs, "pagerank"))
    df["avg_ip_diversity"] = all_addrs.apply(_avg_ip_diversity)

    return df


if __name__ == "__main__":
    df = load_cleaned_data()
    print(f"Loaded {len(df)} transactions")

    G = build_networkx_graph(df)
    print(f"Wallet graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    wallet_features = compute_wallet_structural_features(G)
    ip_diversity = compute_wallet_ip_diversity(df)

    df = attach_transaction_structural_features(df, wallet_features, ip_diversity)

    print("\nStructural feature summary by label:")
    print(df.groupby("label")[["avg_in_degree", "avg_out_degree", "avg_pagerank", "avg_ip_diversity"]].mean())

    output_path = "data/processed/transactions_with_structural_features.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")