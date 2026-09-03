import pandas as pd
import json
import ast
import xml.etree.ElementTree as ET

LIST_COLUMNS = ["input_addresses", "output_addresses", "input_amounts", "output_amounts"]

def _coerce_list_columns(df):
    """XML/CSV store lists as strings like "['a','b']" - convert them back to real lists."""
    for col in LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: ast.literal_eval(v) if isinstance(v, str) and v.strip().startswith("[") else v
            )
    return df
def load_csv(path):
    df = pd.read_csv(path)
    df = _coerce_list_columns(df)
    return df
def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = _coerce_list_columns(df)
    return df
def load_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    records = []
    for tx in root.findall("transaction"):
        record = {child.tag: child.text for child in tx}
        records.append(record)
    df = pd.DataFrame(records)
    df = _coerce_list_columns(df)
    return df
def load_transactions(path):
    """Detects format from file extension and returns a normalized DataFrame."""
    if path.endswith(".csv"):
        return load_csv(path)
    elif path.endswith(".json"):
        return load_json(path)
    elif path.endswith(".xml"):
        return load_xml(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")


if __name__ == "__main__":
    for test_file in [
        "data/raw/synthetic_bitcoin_metadata.csv",
        "data/raw/sample_transactions.json",
        "data/raw/sample_transactions.xml",
    ]:
        df = load_transactions(test_file)
        print(f"{test_file}: loaded {len(df)} rows, columns = {list(df.columns)}")
        print(df.iloc[0][["txid", "input_addresses", "output_amounts"]])
        print("-" * 60)