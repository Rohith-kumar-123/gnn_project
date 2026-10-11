"""
Train and evaluate the GNN variants over several seeds.

    python scripts/run_experiments.py --csv /content/drive/MyDrive/HI-Small_Trans.csv --out /content/drive/MyDrive/gnn_outputs
    python scripts/run_experiments.py --csv ... --variants gcn hrgnn --seeds 3

Safe to stop and restart: finished runs are skipped, an interrupted run resumes from
its last checkpoint (<out>/ckpt), and every final model is saved to <out>/models.
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import VARIANTS, DataConfig, TrainConfig
from src.data import describe_split, prepare
from src.train import run_gnn

SHOW = ["ap", "auc", "precision", "recall", "f1", "false_alerts_per_1000", "recall@0.5%"]


def compare_with_full_model(df):
    """Paired t-test on AP: full HR-GNN vs every other variant, Holm-adjusted."""
    if "hrgnn" not in set(df["model"]):
        return
    ap = lambda m: df[df.model == m].sort_values("seed")["ap"].values
    rows = []
    for m in df["model"].unique():
        if m != "hrgnn" and len(ap(m)) == len(ap("hrgnn")) >= 3:
            t, p = stats.ttest_rel(ap("hrgnn"), ap(m))
            rows.append((m, ap("hrgnn").mean() - ap(m).mean(), p))
    rows.sort(key=lambda r: r[2])
    print("\nHR-GNN vs each variant (paired t-test on AP, Holm-adjusted):")
    prev = 0.0
    for i, (m, d, p) in enumerate(rows):
        adj = min(1.0, max(prev, (len(rows) - i) * p))
        prev = adj
        print(f"  vs {m:22s} diff {d:+.4f}   p {p:.4f}   adjusted p {adj:.4f}")


def main(a):
    dcfg = DataConfig(csv_path=a.csv, dev_normal_frac=0.1 if a.dev else None)
    tcfg = TrainConfig(device=a.device)
    if a.epochs:
        tcfg = replace(tcfg, epochs=a.epochs)
    out = Path(a.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(a.ckpt_dir) if a.ckpt_dir else out / "ckpt"
    model_dir = out / "models"

    prep = prepare(dcfg)
    print("[split]", json.dumps(describe_split(prep), indent=1))

    rows = []
    for name in (a.variants or list(VARIANTS)):
        for seed in range(a.seeds):
            f_json = out / "runs" / f"{name}_s{seed}.json"
            if f_json.exists():                                  # finished earlier -> skip
                res = json.loads(f_json.read_text())
                print(f"== {name} | seed {seed} (already done, skipped)")
            else:
                print(f"== {name} | seed {seed}")
                res, arrays = run_gnn(prep, name, VARIANTS[name], tcfg, seed,
                                      ckpt_dir=ckpt_dir, model_dir=model_dir)
                np.savez_compressed(out / "runs" / f"{name}_s{seed}.npz", **arrays)
                f_json.write_text(json.dumps(res, indent=1))
            rows.append({"model": name, "seed": seed, **res["test"],
                         "train_seconds": res.get("train_seconds", np.nan)})

    df = pd.DataFrame(rows)
    df.to_csv(out / "results_long.csv", index=False)
    g = df.groupby("model", sort=False)
    summary = pd.DataFrame({m: g[m].mean().round(4).astype(str) + " ± " + g[m].std().fillna(0).round(4).astype(str)
                            for m in SHOW})
    summary.to_csv(out / "results_summary.csv")
    print("\n=== TEST RESULTS (mean ± std over seeds) ===")
    print(summary.to_string())
    compare_with_full_model(df)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="local CSV (default: download from Hugging Face)")
    ap.add_argument("--variants", nargs="*", help=f"subset of {list(VARIANTS)} (default: all)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=None, help="max epochs (default from config.py)")
    ap.add_argument("--dev", action="store_true", help="10%% of normal rows, debugging only")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--ckpt_dir", default=None, help="default: <out>/ckpt")
    main(ap.parse_args())
