import sys
sys.path.append(".")

import ast
import streamlit as st
import pandas as pd
from pyvis.network import Network
import streamlit.components.v1 as components

st.set_page_config(page_title="BitTrace - Bitcoin Transaction Intelligence", layout="wide")


@st.cache_data
def load_data():
    alerts = pd.read_csv("data/processed/alerts.csv")
    full_df = pd.read_csv("data/processed/all_predictions.csv")
    for col in ["input_addresses", "output_addresses"]:
        full_df[col] = full_df[col].apply(lambda v: ast.literal_eval(v) if isinstance(v, str) else v)
    return alerts, full_df


alerts, full_df = load_data()

st.title("BitTrace \u2014 AI-Powered Bitcoin Transaction Intelligence")
st.caption("SIH26146 | Offline monitoring & analysis of Bitcoin transaction traffic | Team SNPSU Warlords")

col1, col2, col3 = st.columns(3)
col1.metric("Total transactions analyzed", len(full_df))
col2.metric("Flagged as illicit", len(full_df[full_df["predicted_label"] == "illicit"]))
top_score = f'{alerts["risk_score"].max():.1f}' if len(alerts) else "N/A"
col3.metric("Highest risk score", top_score)

st.subheader("Ranked Investigative Alerts")
st.dataframe(
    alerts[["txid", "risk_score", "explanation"]],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Investigate a Transaction")
if len(alerts) == 0:
    st.info("No flagged transactions to display.")
else:
    selected_txid = st.selectbox("Select a flagged transaction to inspect", alerts["txid"].tolist())
    row = full_df[full_df["txid"] == selected_txid].iloc[0]

    left, right = st.columns([1, 2])
    with left:
        st.metric("Risk score", f"{row['risk_score']:.1f} / 100")
        st.write(f"**Explanation:** {row['explanation']}")
        st.write(f"**Source IP:** {row['src_ip']}")
        st.write(f"**Country:** {row.get('geo_country', 'N/A')}")
        st.write(f"**ASN:** {row.get('asn', 'N/A')}")
        st.write(f"**Input wallets:** {', '.join(row['input_addresses'])}")
        st.write(f"**Output wallets:** {', '.join(row['output_addresses'])}")

    with right:
        net = Network(height="480px", width="100%", bgcolor="#111111",
                       font_color="white", directed=True, cdn_resources="in_line")
        net.add_node(selected_txid, label="TX", color="#e74c3c", size=28)
        for addr in row["input_addresses"]:
            net.add_node(addr, label=addr[:8], color="#f39c12")
            net.add_edge(addr, selected_txid)
        for addr in row["output_addresses"]:
            net.add_node(addr, label=addr[:8], color="#3498db")
            net.add_edge(selected_txid, addr)
        net.add_node(row["src_ip"], label=row["src_ip"], color="#f1c40f", shape="triangle")
        net.add_edge(row["src_ip"], selected_txid)

        html_content = net.generate_html()
        components.html(html_content, height=500)