import pandas as pd
import numpy as np
import random
import string
from datetime import datetime, timedelta

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

N_TRANSACTIONS = 5000
N_WALLETS = 1200
START_TIME = datetime(2026, 1, 1)
def random_wallet():
    return "1" + "".join(random.choices(string.ascii_letters + string.digits, k=25))

def random_txid():
    return "".join(random.choices("0123456789abcdef", k=64))

def random_ip(pool):
    return random.choice(pool)

# Pools of addresses/IPs reused across transactions so a graph can actually form
NORMAL_WALLETS = [random_wallet() for _ in range(N_WALLETS)]
NORMAL_IPS = [f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,255)}" for _ in range(400)]

# A small deliberately "suspicious" IP range — simulates VPN/proxy exit nodes
SUSPICIOUS_IPS = [f"185.220.10{i%9}.{random.randint(1,254)}" for i in range(30)]
def make_normal_transaction(tx_index, timestamp):
    n_in = random.randint(1, 2)
    n_out = random.randint(1, 3)
    inputs = random.sample(NORMAL_WALLETS, n_in)
    outputs = random.sample(NORMAL_WALLETS, n_out)
    in_amt = [round(random.uniform(0.01, 2.0), 6) for _ in inputs]
    total_in = sum(in_amt)
    fee = round(total_in * random.uniform(0.001, 0.01), 6)
    remaining = round(total_in - fee, 6)

    # Split 'remaining' across outputs using normalized random weights,
    # so the outputs always sum exactly to remaining (no drift).
    weights = [random.uniform(0.5, 1.5) for _ in range(n_out)]
    weight_sum = sum(weights)
    out_amt = [round(remaining * w / weight_sum, 6) for w in weights]

    # Fix any tiny rounding drift by dumping it into the last output
    drift = round(remaining - sum(out_amt), 6)
    out_amt[-1] = round(out_amt[-1] + drift, 6)

    return {
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "src_ip": random_ip(NORMAL_IPS),
        "dst_ip": random_ip(NORMAL_IPS),
        "src_port": random.randint(1024, 65535),
        "dst_port": 8333,
        "txid": random_txid(),
        "input_addresses": inputs,
        "output_addresses": outputs,
        "input_amounts": in_amt,
        "output_amounts": out_amt,
        "fee": fee,
        "script_type": random.choice(["P2PKH", "P2SH", "P2WPKH"]),
        "label": "licit",
    }
def inject_peel_chain(start_time, chain_length=6):
    """One wallet repeatedly sends most of its value forward, peeling off a bit each hop."""
    records = []
    current_wallet = random_wallet()
    balance = round(random.uniform(5, 20), 6)
    t = start_time
    for i in range(chain_length):
        next_wallet = random_wallet()
        peel = round(balance * random.uniform(0.02, 0.08), 6)
        fee = round(balance * 0.001, 6)
        forward_amt = round(balance - peel - fee, 6)
        records.append({
            "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
            "src_ip": random_ip(SUSPICIOUS_IPS),
            "dst_ip": random_ip(SUSPICIOUS_IPS),
            "src_port": random.randint(1024, 65535),
            "dst_port": 8333,
            "txid": random_txid(),
            "input_addresses": [current_wallet],
            "output_addresses": [next_wallet, random_wallet()],  # forward + peel-off
            "input_amounts": [balance],
            "output_amounts": [forward_amt, peel],
            "fee": fee,
            "script_type": random.choice(["P2PKH", "P2SH"]),
            "label": "illicit",
        })
        current_wallet = next_wallet
        balance = forward_amt
        t += timedelta(minutes=random.randint(5, 60))
    return records
def inject_mixing_pattern(start_time, n_intermediate=8):
    """One source fans out to many wallets, which fan back into one destination."""
    records = []
    source = random_wallet()
    destination = random_wallet()
    intermediates = [random_wallet() for _ in range(n_intermediate)]
    total = round(random.uniform(10, 30), 6)
    t = start_time

    # Fan-out: source -> many intermediates
    for w in intermediates:
        amt = round(total / n_intermediate * random.uniform(0.9, 1.1), 6)
        records.append({
            "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
            "src_ip": random_ip(SUSPICIOUS_IPS),
            "dst_ip": random_ip(SUSPICIOUS_IPS),
            "src_port": random.randint(1024, 65535),
            "dst_port": 8333,
            "txid": random_txid(),
            "input_addresses": [source],
            "output_addresses": [w],
            "input_amounts": [amt],
            "output_amounts": [round(amt * 0.99, 6)],
            "fee": round(amt * 0.001, 6),
            "script_type": "P2WPKH",
            "label": "illicit",
        })
        t += timedelta(minutes=random.randint(1, 10))

    # Fan-in: intermediates -> single destination
    for w in intermediates:
        amt = round(total / n_intermediate * random.uniform(0.85, 0.95), 6)
        records.append({
            "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
            "src_ip": random_ip(SUSPICIOUS_IPS),
            "dst_ip": random_ip(SUSPICIOUS_IPS),
            "src_port": random.randint(1024, 65535),
            "dst_port": 8333,
            "txid": random_txid(),
            "input_addresses": [w],
            "output_addresses": [destination],
            "input_amounts": [amt],
            "output_amounts": [round(amt * 0.99, 6)],
            "fee": round(amt * 0.001, 6),
            "script_type": "P2WPKH",
            "label": "illicit",
        })
        t += timedelta(minutes=random.randint(1, 10))

    return records
def inject_ip_hopping_cluster(start_time, n_tx=10):
    """Same small wallet cluster transacts repeatedly from a rotating pool of IPs."""
    records = []
    cluster_wallets = [random_wallet() for _ in range(4)]
    t = start_time
    for _ in range(n_tx):
        src, dst = random.sample(cluster_wallets, 2)
        amt = round(random.uniform(0.1, 3.0), 6)
        records.append({
            "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
            "src_ip": random_ip(SUSPICIOUS_IPS),   # different IP each time
            "dst_ip": random_ip(SUSPICIOUS_IPS),
            "src_port": random.randint(1024, 65535),
            "dst_port": 8333,
            "txid": random_txid(),
            "input_addresses": [src],
            "output_addresses": [dst],
            "input_amounts": [amt],
            "output_amounts": [round(amt * 0.995, 6)],
            "fee": round(amt * 0.005, 6),
            "script_type": "P2SH",
            "label": "illicit",
        })
        t += timedelta(minutes=random.randint(10, 120))
    return records
def generate_dataset():
    all_records = []
    t = START_TIME

    # Bulk of normal traffic
    for i in range(N_TRANSACTIONS):
        t += timedelta(minutes=random.randint(1, 15))
        all_records.append(make_normal_transaction(i, t))

    # Inject illicit patterns scattered through the timeline
    for _ in range(15):
        offset = timedelta(days=random.randint(0, 60))
        all_records.extend(inject_peel_chain(START_TIME + offset))

    for _ in range(8):
        offset = timedelta(days=random.randint(0, 60))
        all_records.extend(inject_mixing_pattern(START_TIME + offset))

    for _ in range(10):
        offset = timedelta(days=random.randint(0, 60))
        all_records.extend(inject_ip_hopping_cluster(START_TIME + offset))

    df = pd.DataFrame(all_records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(f"Generated {len(df)} transactions")
    print(df["label"].value_counts())
    df.to_csv("data/raw/synthetic_bitcoin_metadata.csv", index=False)
    print("Saved to data/raw/synthetic_bitcoin_metadata.csv")