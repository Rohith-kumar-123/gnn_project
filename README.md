# HR-GNN: Historical- and Relational-aware Graph Learning for Fraud Detection

Detects laundering transactions in **AMLSim HI-Small** with a GNN that knows **how long an account has been active**
and **how trustworthy each transaction link is**, and compares it with a plain GCN.

## Idea

| Module | What it does |
|---|---|
| **A. History + persistence** | Per account: active days, active-day ratio, streak, recency, past amounts, counterparties, burstiness. A gate turns persistence into a trust score `p_v`. |
| **B. Edge reliability** | `r_uv` in [0,1] for every transaction, from both accounts' history, the amount and the edge age. |
| **C. Weighted message passing** | Neighbours count more if persistent (`p_u`), recent (`exp(-λ·dt)`) and reliable (`r_uv`). A second channel keeps the unreliable neighbourhood as a signal. |
| **D. Edge classifier** | Fraud score per transaction from both account embeddings, the transaction features and `r_uv`. |

Loss = focal loss + β · reliability loss (weak supervision, training labels only) + γ · entropy of `r`.

## Folder structure

```
hr-gnn-fraud/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py     settings + the list of model variants (baseline, ablations, full model)
│   ├── data.py       load, sort by time, 60/20/20 chronological split, feature cache
│   ├── features.py   persistence / history features and node snapshots (leak-free)
│   ├── graph.py      per-phase graph tensors
│   ├── model.py      HR-GNN (one class; switches turn modules on/off)
│   ├── train.py      training, checkpoints/resume, model saving, load_model(), predict()
│   └── metrics.py    AP, AUC, precision, recall, F1, alert rate, recall@alert-budget
├── scripts/
│   ├── eda.py              data overview
│   ├── run_experiments.py  train + evaluate all variants x seeds, significance tests
│   └── make_report.py      alert-budget curves, reliability histogram, cold-start table
├── data/    raw/ processed/   (processed/ holds the feature cache)
└── outputs/                   (results, see below)
```

## Run (Colab, data on Google Drive)

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone https://github.com/Rohith-kumar-123/GNN.git /content/GNN
%cd /content/GNN
!pip -q install pyarrow

CSV = "/content/drive/MyDrive/HI-Small_Trans.csv"
OUT = "/content/drive/MyDrive/gnn_outputs"

!python scripts/eda.py --csv "{CSV}"
!python scripts/run_experiments.py --csv "{CSV}" --seeds 5 --out {OUT}
!python scripts/make_report.py --csv "{CSV}" --out {OUT}
```

If Colab disconnects, run the same cells again: finished runs are skipped and an interrupted run continues
from its last checkpoint (saved every 25 epochs). Use `--variants gcn hrgnn` to run only some models.

## What is saved (in `--out`)

| Path | Content |
|---|---|
| `results_summary.csv`, `results_long.csv` | mean ± std table, and one row per model/seed |
| `runs/<model>_s<seed>.json` | validation + test metrics, best epoch, training time |
| `runs/<model>_s<seed>.npz` | test labels, fraud scores, reliability, row numbers |
| `models/<model>_s<seed>.pt` | **final model** (weights, settings, alert threshold) for the dashboard |
| `ckpt/` | temporary crash checkpoints, deleted when a run finishes |

## Using a saved model (dashboard)

```python
from src.config import DataConfig
from src.data import prepare
from src.train import predict

prep = prepare(DataConfig(csv_path=CSV))                 # loads the cached features
df = predict(prep, f"{OUT}/models/hrgnn_s0.pt", phase="test")
alerts = df[df.alert].sort_values("fraud_score", ascending=False)
```
`df` has one row per transaction: timestamp, from/to account, amount, payment format, `fraud_score`,
`alert` (above the validation-chosen threshold), `reliability` and the true `label`.
For a dashboard that only displays results, the saved `.npz` files already contain scores for the test period.

## Models compared

`gcn` (plain GCN baseline) · `gcn_hist` (history only) · `gcn_reliability` (reliability only) ·
`hrgnn_no_decay` · `hrgnn_no_persistence` · `hrgnn_single_channel` · `hrgnn` (full model)

## Evaluation protocol (no leakage)

* Sorted by timestamp, split **60 / 20 / 20 chronologically**.
* Per-transaction history uses only earlier rows; persistence counts whole earlier days.
* The graph used to score a phase contains only transactions up to the end of that phase; no labels at inference.
* Standardisation statistics come from training rows only; the F1 threshold is chosen on validation and frozen for test.
* Metrics: AP (primary), ROC-AUC, precision, recall, F1, false alerts per 1,000 transactions, recall at 0.1 / 0.5 / 1 % alert budget.
* 5 seeds, mean ± std, paired t-test of HR-GNN against every other variant (Holm-adjusted).

## Limitations

* The GNN's account snapshot for *training* edges includes later training days (no labels, never touches val/test).
* The data starts at day 0, so every account looks new at the beginning (left censoring).
* The test period (days 7–17) has a higher fraud rate than training; compare models only within the same split.
* "Suspicious" edges for reliability supervision are weak labels (edges touching training-period fraud accounts).
* The plain GCN uses mean-normalised aggregation, not PyG's symmetric normalisation.
* HI-Small is simulated; validate on another benchmark as future work.
