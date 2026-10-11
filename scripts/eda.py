"""
Quick look at the data: size, fraud rate, time span, fraud per split and per payment format.

    python scripts/eda.py [--csv path]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DataConfig
from src.data import clean_and_sort, chronological_bounds, load_raw


def main(a):
    cfg = DataConfig(csv_path=a.csv)
    df, n_nodes = clean_and_sort(load_raw(cfg), cfg)
    print(f"rows {len(df):,} | accounts {n_nodes:,} | fraud {df.y.sum():,} ({df.y.mean():.4%})")
    print("time span:", df.timestamp.min(), "->", df.timestamp.max(), f"({df.day.max() + 1} days)")

    n_tr, n_va, n = chronological_bounds(len(df), cfg)
    for name, (s, e) in {"train": (0, n_tr), "val": (n_tr, n_va), "test": (n_va, n)}.items():
        y = df.y.values[s:e]
        print(f"{name:5s} rows {e - s:>9,}  fraud {y.sum():>6,}  rate {y.mean():.4%}  days {df.day.values[s]}-{df.day.values[e - 1]}")

    print("\nfraud rate by payment format:")
    print(df.groupby("pay_format").y.agg(["size", "sum", "mean"]).sort_values("mean", ascending=False))

    daily = df.groupby("day").y.agg(["size", "sum"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    daily["size"].plot.bar(ax=ax[0], title="Transactions per day")
    daily["sum"].plot.bar(ax=ax[1], color="firebrick", title="Fraud transactions per day")
    plt.tight_layout()
    Path("outputs").mkdir(exist_ok=True)
    plt.savefig("outputs/eda_daily.png", dpi=150)
    print("saved outputs/eda_daily.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    main(ap.parse_args())
