"""
Load AMLSim HI-Small, sort it by time, split it chronologically and build features.

    prep = prepare(DataConfig())

`prep` holds everything the models and baselines need. The expensive part
(feature engineering on ~5M rows) is cached to data/processed/ so it only runs once.
"""
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig
from . import features as F

RAW_COLS = ["timestamp", "from_bank", "src_acct", "to_bank", "dst_acct",
            "amt_recv", "cur_recv", "amt_paid", "cur_paid", "pay_format", "is_fraud"]


@dataclass
class Prepared:
    df: pd.DataFrame            # time-sorted transactions (raw columns + src, dst, t_sec, day, y)
    n_nodes: int
    n_train: int                # rows [0, n_train)            -> train
    n_val_end: int              # rows [n_train, n_val_end)    -> validation
    n_total: int                # rows [n_val_end, n_total)    -> test
    tx_cols: list
    hist_cols: list
    edge_tx: np.ndarray         # [E, n_tx]   standardised
    edge_hist: np.ndarray       # [E, n_hist] standardised
    snapshots: dict             # phase name -> node matrix [N, len(NODE_COLS)]
    hist_raw: pd.DataFrame      # un-standardised history features (for analysis / tree models)

    def bounds(self, phase: str):
        """(start, end) rows of the TARGET edges of a phase. The graph is rows [0, end)."""
        return {"train": (0, self.n_train),
                "val": (self.n_train, self.n_val_end),
                "test": (self.n_val_end, self.n_total)}[phase]


# -----------------------------------------------------------------------------
def load_raw(cfg: DataConfig) -> pd.DataFrame:
    if cfg.csv_path:
        df = pd.read_csv(cfg.csv_path)
    else:
        from datasets import load_dataset
        df = load_dataset(cfg.hf_name, split="train").to_pandas()
    # The file has two columns called "Account" (sender, receiver) -> rename by position.
    assert df.shape[1] == 11, f"expected 11 columns, got {df.shape[1]}: {list(df.columns)}"
    df.columns = RAW_COLS
    return df


def clean_and_sort(df: pd.DataFrame, cfg: DataConfig):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M")
    df["y"] = df["is_fraud"].astype(int)

    if cfg.dev_normal_frac is not None:   # quick-debug subsample (NOT for reporting)
        rng = np.random.default_rng(0)
        keep = (df["y"] == 1) | (rng.random(len(df)) < cfg.dev_normal_frac)
        df = df[keep]

    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    codes, uniques = pd.factorize(pd.concat([df["src_acct"], df["dst_acct"]], ignore_index=True))
    n = len(df)
    df["src"], df["dst"] = codes[:n], codes[n:]
    df["t_sec"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    df["day"] = (df["t_sec"] // 86400).astype(int)
    return df, len(uniques)


def chronological_bounds(n: int, cfg: DataConfig):
    n_train = int(n * cfg.train_frac)
    n_val_end = int(n * (cfg.train_frac + cfg.val_frac))
    return n_train, n_val_end, n


# -----------------------------------------------------------------------------
def prepare(cfg: DataConfig, force: bool = False) -> Prepared:
    tag = f"dev{cfg.dev_normal_frac}" if cfg.dev_normal_frac else "full"
    cache = Path(cfg.cache_dir) / f"prepared_{tag}.pkl"
    if cache.exists() and not force:
        print(f"[data] loading cached features from {cache}")
        with open(cache, "rb") as f:
            return pickle.load(f)

    print("[data] loading raw data ...")
    df, n_nodes = clean_and_sort(load_raw(cfg), cfg)
    n_train, n_val_end, n_total = chronological_bounds(len(df), cfg)
    print(f"[data] {len(df):,} transactions, {n_nodes:,} accounts, fraud rate {df['y'].mean():.4%}")

    print("[data] building transaction + history features ...")
    tx = F.transaction_features(df)
    hist = F.history_features(df, cfg.dormant_days)

    mu_tx, sd_tx = F.fit_standardiser(tx, n_train)
    mu_h, sd_h = F.fit_standardiser(hist, n_train)

    print("[data] building node snapshots (train / val / test) ...")
    snaps = {}
    for phase, end in [("train", n_train), ("val", n_val_end), ("test", n_total)]:
        snaps[phase] = F.node_snapshot(df.iloc[:end], n_nodes, cfg.dormant_days)

    prep = Prepared(
        df=df, n_nodes=n_nodes, n_train=n_train, n_val_end=n_val_end, n_total=n_total,
        tx_cols=list(tx.columns), hist_cols=list(hist.columns),
        edge_tx=F.apply_standardiser(tx, mu_tx, sd_tx),
        edge_hist=F.apply_standardiser(hist, mu_h, sd_h),
        snapshots=snaps, hist_raw=hist,
    )
    Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(prep, f, protocol=4)
    print(f"[data] cached to {cache}")
    return prep


def describe_split(prep: Prepared) -> dict:
    out = {}
    for ph in ["train", "val", "test"]:
        a, b = prep.bounds(ph)
        y = prep.df["y"].values[a:b]
        out[ph] = {"rows": int(b - a), "fraud": int(y.sum()), "fraud_rate": float(y.mean()),
                   "days": f"{prep.df['day'].values[a]}-{prep.df['day'].values[b - 1]}"}
    return out
