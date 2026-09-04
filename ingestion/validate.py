import pandas as pd
import ipaddress


def is_valid_ip(ip):
    try:
        ipaddress.ip_address(str(ip))
        return True
    except ValueError:
        return False


def check_required_fields(df):
    mask = df["txid"].notna() & (df["txid"].astype(str).str.strip() != "")
    mask &= df["timestamp"].notna() & (df["timestamp"].astype(str).str.strip() != "")
    return mask


def check_duplicate_txid(df):
    is_duplicate = df["txid"].duplicated(keep="first")
    return ~is_duplicate


def check_valid_ips(df):
    return df["src_ip"].apply(is_valid_ip) & df["dst_ip"].apply(is_valid_ip)


def check_amount_balance(df, tolerance=0.05):
    def _balanced(row):
        try:
            in_amounts = row["input_amounts"]
            out_amounts = row["output_amounts"]
            in_total = sum(in_amounts) if isinstance(in_amounts, list) else float(in_amounts)
            out_total = sum(out_amounts) if isinstance(out_amounts, list) else float(out_amounts)
            fee = float(row["fee"])
            return abs(in_total - (out_total + fee)) <= tolerance * max(in_total, 1e-8)
        except Exception:
            return False
    return df.apply(_balanced, axis=1)


def validate_and_clean(df, verbose=True):
    original_count = len(df)
    checks = {
        "missing required fields": check_required_fields(df),
        "duplicate txid": check_duplicate_txid(df),
        "invalid IP address": check_valid_ips(df),
        "amount imbalance": check_amount_balance(df),
    }

    combined_mask = pd.Series(True, index=df.index)
    for name, mask in checks.items():
        dropped_by_this_check = (~mask & combined_mask).sum()
        if verbose:
            print(f"  [{name}] flagged {dropped_by_this_check} additional row(s)")
        combined_mask &= mask

    cleaned_df = df[combined_mask].reset_index(drop=True)

    if verbose:
        total_dropped = original_count - len(cleaned_df)
        pct = (total_dropped / original_count * 100) if original_count else 0
        print(f"\nValidation summary: {original_count} rows in -> {len(cleaned_df)} rows out "
              f"({total_dropped} dropped, {pct:.2f}%)")

    return cleaned_df


if __name__ == "__main__":
    from parsers import load_transactions
    from geoip_lookup import add_geo_columns

    df = load_transactions("data/raw/synthetic_bitcoin_metadata.csv")
    print(f"Loaded {len(df)} raw rows\n")

    df = validate_and_clean(df)

    print("\nAdding GeoIP columns...")
    df = add_geo_columns(df)

    output_path = "data/processed/cleaned_transactions.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved cleaned dataset to {output_path}")
    print(f"Final label distribution:\n{df['label'].value_counts()}")