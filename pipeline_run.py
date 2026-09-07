import sys
sys.path.append(".")

import argparse
import torch

from ingestion.parsers import load_transactions
from ingestion.validate import validate_and_clean
from ingestion.geoip_lookup import add_geo_columns
from graph.structural_features import (
    build_networkx_graph,
    compute_wallet_structural_features,
    compute_wallet_ip_diversity,
    attach_transaction_structural_features,
)
from graph.build_graph import build_graph
from model.train import load_graph_with_all_features, DEVICE
from model.risk_scoring import load_trained_model, build_alerts


def run_pipeline(input_path):
    print(f"Step 1/6: Ingesting {input_path} ...")
    df = load_transactions(input_path)
    print(f"  Loaded {len(df)} raw transactions")

    print("Step 2/6: Validating & cleaning ...")
    df = validate_and_clean(df, verbose=True)

    print("Step 3/6: Resolving offline GeoIP ...")
    df = add_geo_columns(df)

    print("Step 4/6: Computing structural graph features ...")
    G = build_networkx_graph(df)
    wallet_features = compute_wallet_structural_features(G)
    ip_diversity = compute_wallet_ip_diversity(df)
    df = attach_transaction_structural_features(df, wallet_features, ip_diversity)
    df.to_csv("data/processed/transactions_with_structural_features.csv", index=False)

    print("Step 5/6: Building the entity graph ...")
    graph, ip_to_idx, wallet_to_idx, tx_to_idx = build_graph(df)
    torch.save(graph, "data/processed/transaction_graph.pt")

    print("Step 6/6: Running the trained model & generating alerts ...")
    graph = load_graph_with_all_features()
    graph = graph.to(DEVICE)
    model = load_trained_model(graph)
    alerts, full_df = build_alerts(df, graph, model)

    alerts.to_csv("data/processed/alerts.csv", index=False)
    full_df.to_csv("data/processed/all_predictions.csv", index=False)

    n_flagged = len(full_df[full_df["predicted_label"] == "illicit"])
    print(f"\nPipeline complete: {len(full_df)} transactions processed, {n_flagged} flagged as illicit.")
    print("Run `streamlit run dashboard/app.py --server.address 127.0.0.1` to view the results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full CryptoTrace pipeline on an input file.")
    parser.add_argument("input_path", help="Path to a CSV/JSON/XML transaction metadata file")
    args = parser.parse_args()
    run_pipeline(args.input_path)