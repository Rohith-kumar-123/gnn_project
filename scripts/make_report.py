"""
Figures and tables for the paper, built from the saved runs in outputs/runs/.

    python scripts/make_report.py [--models gcn gcn_hist hrgnn]

Produces in outputs/:
    alert_budget.png      recall as a function of the share of transactions reviewed
    reliability_dist.png  r_uv for fraud vs normal test edges (HR-GNN)
    cold_start.csv        AP for accounts with little vs lots of history
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DataConfig
from src.data import prepare


def load(runs, name, seed=0):
    f = runs / f"{name}_s{seed}.npz"
    return dict(np.load(f)) if f.exists() else None


def main(a):
    runs = Path(a.out) / "runs"
    out = Path(a.out)

    # 1) alert-budget curves
    rates = np.logspace(-3.3, -1.3, 25)
    plt.figure(figsize=(6, 4))
    for m in a.models:
        d = load(runs, m)
        if d is None:
            continue
        order = np.argsort(-d["score"])
        hits = np.cumsum(d["y"][order]) / max(d["y"].sum(), 1)
        plt.plot(rates * 100, [hits[max(int(r * len(order)), 1) - 1] for r in rates], label=m)
    plt.xscale("log"); plt.xlabel("% of transactions reviewed (alert rate)"); plt.ylabel("fraud recall")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(out / "alert_budget.png", dpi=150)

    # 2) reliability distribution
    d = load(runs, "hrgnn")
    if d is not None and "r" in d:
        plt.figure(figsize=(6, 4))
        bins = np.linspace(0, 1, 40)
        plt.hist(d["r"][d["y"] == 0], bins=bins, alpha=.6, density=True, label="normal")
        plt.hist(d["r"][d["y"] == 1], bins=bins, alpha=.6, density=True, label="fraud")
        plt.xlabel("edge reliability r"); plt.ylabel("density"); plt.legend(); plt.tight_layout()
        plt.savefig(out / "reliability_dist.png", dpi=150)

    # 3) cold-start vs established accounts (sender history at the time of the transaction)
    prep = prepare(DataConfig(csv_path=a.csv, dev_normal_frac=0.1 if a.dev else None))
    s, e = prep.bounds("test")
    prior_days = np.expm1(prep.hist_raw["s_active_days"].values[s:e])
    cold = prior_days < 1          # sender had no earlier active day
    rows = []
    for m in a.models:
        d = load(runs, m)
        if d is None:
            continue
        for label, mask in [("cold-start (first active day)", cold), ("established", ~cold)]:
            y = d["y"][mask]
            rows.append({"model": m, "group": label, "n": int(mask.sum()), "fraud": int(y.sum()),
                         "AP": average_precision_score(y, d["score"][mask]) if y.sum() else np.nan})
    pd.DataFrame(rows).to_csv(out / "cold_start.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["gcn", "gcn_hist", "hrgnn"])
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--dev", action="store_true")
    main(ap.parse_args())
