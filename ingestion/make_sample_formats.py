import pandas as pd
import json
import xml.etree.ElementTree as ET
import ast

def csv_row_to_native(row):
    """Convert stringified list columns back into real Python lists."""
    row = row.copy()
    for col in ["input_addresses", "output_addresses", "input_amounts", "output_amounts"]:
        if isinstance(row[col], str):
            row[col] = ast.literal_eval(row[col])
    return row

def main():
    df = pd.read_csv("data/raw/synthetic_bitcoin_metadata.csv")
    sample = df.head(20).apply(csv_row_to_native, axis=1)

    # ---- JSON ----
    records = sample.to_dict(orient="records")
    with open("data/raw/sample_transactions.json", "w") as f:
        json.dump(records, f, indent=2)
    print("Wrote data/raw/sample_transactions.json")

    # ---- XML ----
    root = ET.Element("transactions")
    for _, row in sample.iterrows():
        tx = ET.SubElement(root, "transaction")
        for col in df.columns:
            child = ET.SubElement(tx, col)
            child.text = str(row[col])
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write("data/raw/sample_transactions.xml", encoding="utf-8", xml_declaration=True)
    print("Wrote data/raw/sample_transactions.xml")

if __name__ == "__main__":
    main()