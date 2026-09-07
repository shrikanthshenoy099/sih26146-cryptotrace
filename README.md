# SIH26146 — BitTrace 
**AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic**
Team: SNPSU Warlords | Sponsor: NTRO | Theme: Smart Automation | Category: Software

## Overview
An offline, Linux-based system that ingests bulk Bitcoin network + blockchain
metadata (CSV/JSON/XML), correlates network-layer (IP/port/timing) with
blockchain-layer (wallet/TXID/amount) data, and applies a Graph Neural Network
to detect anomalies, cluster entities, and generate ranked, explainable
investigative leads via a link-analysis dashboard.

## Data Schema

All ingestion parsers (CSV/JSON/XML) must normalize into this exact schema.
Use snake_case field names everywhere — no `srcIP`, no `SourceIP`.

| Field              | Type            | Notes                                      |
|--------------------|-----------------|---------------------------------------------|
| `timestamp`        | ISO datetime str| UTC, format `YYYY-MM-DD HH:MM:SS`           |
| `src_ip`           | string          | IPv4                                        |
| `dst_ip`           | string          | IPv4                                        |
| `src_port`         | int             | 0–65535                                     |
| `dst_port`         | int             | 0–65535                                     |
| `txid`             | string          | Unique — treat as primary key               |
| `input_addresses`  | list[string]    | Can have multiple per transaction           |
| `output_addresses` | list[string]    | Can have multiple per transaction           |
| `input_amounts`    | list[float]     | BTC — same length as input_addresses        |
| `output_amounts`   | list[float]     | BTC — same length as output_addresses       |
| `fee`              | float           | BTC                                         |
| `script_type`      | string enum     | `P2PKH` / `P2SH` / `P2WPKH`                 |
| `geo_country`      | string          | Filled in by GeoIP step, not in raw data    |
| `asn`              | string          | Filled in by GeoIP step, not in raw data    |

### "Clean" data rules (checked in `ingestion/validate.py`)
- Every row must have a non-null `txid` and `timestamp`
- No duplicate `txid` values
- `src_ip` / `dst_ip` must be valid IPv4 strings
- `sum(output_amounts) + fee` should roughly equal `sum(input_amounts)`

## Project Structure
```
sih26146-BitTrace/
├── data/
│   ├── raw/              # synthetic generated files land here
│   └── processed/        # cleaned, validated data
├── ingestion/
│   ├── generate_synthetic.py
│   ├── parsers.py         # CSV/JSON/XML loaders
│   └── validate.py
├── graph/                 # Phase 2 — entity/transaction graph construction
├── model/                 # Phase 3 — GNN training & evaluation
├── backend/                # Phase 5 — FastAPI service
├── dashboard/              # Phase 5 — Streamlit + PyVis dashboard
├── notebooks/               # exploratory work only, not production code
├── requirements.txt
└── README.md
```

## Setup
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python test_imports.py          # verify everything installed correctly
```

## Team Roles
| Member | Role |
|---|---|
| 1 | GNN Model Lead |
| 2 | Data & Graph Engineer |
| 3 | Evaluation & Robustness |
| 4 | Backend Engineer (FastAPI) |
| 5 | Frontend/Visualization Engineer |
| 6 | Research & Pitch Lead |

## Phase Checklist
- [ ] Phase 0 — Understand & set up
- [ ] Phase 1 — Synthetic dataset & ingestion pipeline
- [ ] Phase 2 — Entity/transaction graph construction
- [ ] Phase 3 — AI/ML detection model
- [ ] Phase 4 — Explainability & risk scoring
- [ ] Phase 5 — Backend & dashboard
- [ ] Phase 6 — Offline packaging & integration testing
- [ ] Phase 7 — Documentation & pitch prep
