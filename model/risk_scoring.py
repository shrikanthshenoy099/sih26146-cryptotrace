import sys
sys.path.append(".")

import numpy as np
import pandas as pd
import torch

from model.train import load_graph_with_all_features, build_model, DEVICE

FEATURE_NAMES = [
    "total input amount", "total output amount", "fee", "transaction size",
    "avg wallet in-degree", "avg wallet out-degree", "avg wallet pagerank", "avg IP diversity",
]


def load_trained_model(graph):
    model = build_model(graph).to(DEVICE)
    with torch.no_grad():
        model(graph.x_dict, graph.edge_index_dict)  # materialize lazy layers
    model.load_state_dict(torch.load("model/gnn_model.pt", weights_only=True, map_location=DEVICE))
    model.eval()
    return model


def compute_risk_scores(graph, model):
    with torch.no_grad():
        out = model(graph.x_dict, graph.edge_index_dict)
        logits = out["transaction"]
        probs = torch.softmax(logits, dim=1)
        illicit_prob = probs[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
    risk_scores = (illicit_prob * 100).round(2)
    return risk_scores, preds


def explain_transaction(features_row, feature_means, feature_stds, top_n=2):
    """Rule-based explanation: which features deviate most from the dataset average."""
    z_scores = (features_row - feature_means) / (feature_stds + 1e-8)
    order = np.argsort(-np.abs(z_scores))
    reasons = []
    for idx in order[:top_n]:
        name = FEATURE_NAMES[idx]
        z = z_scores[idx]
        direction = "high" if z > 0 else "low"
        reasons.append(f"unusually {direction} {name} (z={z:.1f})")
    return "Flagged due to " + " and ".join(reasons) + "."


def build_alerts(df, graph, model, top_n_alerts=50):
    risk_scores, preds = compute_risk_scores(graph, model)

    features = graph["transaction"].x.cpu().numpy()
    feature_means = features.mean(axis=0)
    feature_stds = features.std(axis=0)

    df = df.reset_index(drop=True).copy()
    df["risk_score"] = risk_scores
    df["predicted_label"] = np.where(preds == 1, "illicit", "licit")
    df["explanation"] = [
        explain_transaction(features[i], feature_means, feature_stds)
        for i in range(len(df))
    ]

    alerts = df[df["predicted_label"] == "illicit"].sort_values("risk_score", ascending=False)
    alert_cols = ["txid", "risk_score", "predicted_label", "explanation", "label"]
    alerts = alerts[alert_cols].head(top_n_alerts)

    return alerts, df


if __name__ == "__main__":
    from graph.build_graph import load_cleaned_data

    df = load_cleaned_data(path="data/processed/transactions_with_structural_features.csv")
    graph = load_graph_with_all_features()
    graph = graph.to(DEVICE)
    model = load_trained_model(graph)

    alerts, full_df = build_alerts(df, graph, model)

    n_flagged = len(full_df[full_df["predicted_label"] == "illicit"])
    print(f"Total flagged as illicit: {n_flagged}")
    print(f"\nTop {len(alerts)} ranked alerts:\n")
    print(alerts.to_string(index=False))

    alerts.to_csv("data/processed/alerts.csv", index=False)
    full_df.to_csv("data/processed/all_predictions.csv", index=False)
    print("\nSaved data/processed/alerts.csv and data/processed/all_predictions.csv")
