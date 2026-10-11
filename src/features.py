"""
Feature engineering for the HR-GNN project.

Two kinds of features are built, and neither ever looks at labels:

1. EDGE features (one row per transaction)
     - tx features   : amount, currency mismatch, payment format, hour of day ...
     - hist features : what we knew about the sender / receiver BEFORE this
                       transaction (activity persistence, past amounts, past
                       counterparties, burstiness ...)

2. NODE snapshots (one row per account)
     - a summary of everything an account did in the graph visible at a given
       moment (end of train / end of validation / end of test).

Leakage rules
-------------
* Per-edge history uses strictly earlier rows (cumulative sums minus the current row)
  or strictly earlier DAYS (persistence is counted in whole days).
* A node snapshot for phase X only uses rows up to the end of phase X.
* Standardisation statistics are fitted on the training rows only.

The dataframe passed in must already be sorted by time (see data.py).
"""
import numpy as np
import pandas as pd

PAY_FORMATS_PREFIX = "fmt_"

# --- node snapshot column groups (order matters, models slice by these) --------
STRUCT_COLS = ["log_in_deg", "log_out_deg"]
PERSIST_COLS = ["active_days", "active_ratio", "streak", "recency", "age_days", "dormant"]
BEHAV_COLS = ["log_n_out", "log_n_in", "mean_amt_out", "mean_amt_in", "std_amt",
              "log_uniq_cp", "in_out_ratio", "log_recent_1d", "new_cp_frac"]
NODE_COLS = STRUCT_COLS + PERSIST_COLS + BEHAV_COLS


# =============================================================================
# 1. Transaction-level features
# =============================================================================
def transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features describing the transaction itself (no history)."""
    out = pd.DataFrame(index=df.index)
    out["log_amt_paid"] = np.log1p(df["amt_paid"])
    out["log_amt_recv"] = np.log1p(df["amt_recv"])
    out["currency_mismatch"] = (df["cur_recv"] != df["cur_paid"]).astype(float)
    out["is_self_loop"] = (df["src"] == df["dst"]).astype(float)
    out["cross_bank"] = (df["from_bank"] != df["to_bank"]).astype(float)
    hour = (df["t_sec"] % 86400) / 3600.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    fmt = pd.get_dummies(df["pay_format"], prefix=PAY_FORMATS_PREFIX.rstrip("_")).astype(float)
    return pd.concat([out, fmt], axis=1)


# =============================================================================
# 2. Persistence (active-day) table
# =============================================================================
def persistence_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (account, active day). Every number is about the days BEFORE that
    day, so joining it back to a transaction never leaks the same day's events.

        active_days_prior : how many distinct earlier days the account was active
        recency_days      : days since the previous active day (NaN if first day)
        age_days          : days since the account was first seen
        streak_prior      : length of the consecutive-day run that ended yesterday
        run_len           : consecutive-day run length INCLUDING this day (used by snapshots)
    """
    events = pd.concat([
        df[["src", "day"]].rename(columns={"src": "acct"}),
        df[["dst", "day"]].rename(columns={"dst": "acct"}),
    ], ignore_index=True)
    days = events.drop_duplicates().sort_values(["acct", "day"]).reset_index(drop=True)
    g = days.groupby("acct")

    days["active_days_prior"] = g.cumcount()
    prev_day = g["day"].shift(1)
    days["recency_days"] = days["day"] - prev_day
    days["first_day"] = g["day"].transform("min")
    days["age_days"] = days["day"] - days["first_day"]

    # consecutive-day runs: a new run starts whenever the gap to the previous day is not 1
    new_run = (days["recency_days"] != 1)
    run_id = new_run.groupby(days["acct"]).cumsum()
    days["run_len"] = days.groupby([days["acct"], run_id]).cumcount() + 1
    prev_run_len = days.groupby("acct")["run_len"].shift(1)
    days["streak_prior"] = np.where(days["recency_days"] == 1, prev_run_len, 0.0)

    # share of the days since first appearance on which the account was active
    days["active_ratio"] = np.where(days["age_days"] > 0,
                                    days["active_days_prior"] / days["age_days"].clip(lower=1), 0.0)
    return days


def _attach_persistence(df, pt, key, prefix, dormant_days):
    m = df[[key, "day"]].merge(pt, left_on=[key, "day"], right_on=["acct", "day"], how="left")
    ix = df.index
    rec = m["recency_days"].fillna(0).values
    return pd.DataFrame({
        f"{prefix}_active_days": np.log1p(m["active_days_prior"].values),
        f"{prefix}_active_ratio": m["active_ratio"].values,
        f"{prefix}_streak": np.log1p(m["streak_prior"].values),
        f"{prefix}_recency": np.log1p(rec),
        f"{prefix}_age_days": np.log1p(m["age_days"].values),
        f"{prefix}_dormant_return": (rec >= dormant_days).astype(float),
        f"{prefix}_first_seen_today": (m["active_days_prior"].values == 0).astype(float),
    }, index=ix)


# =============================================================================
# 3. Per-edge behavioural history (strictly earlier transactions only)
# =============================================================================
def _role_history(df, key, prefix):
    """History of the account playing `key` ('src' or 'dst') in each transaction."""
    la = np.log1p(df["amt_paid"])
    by = df[key]

    n_prior = la.groupby(by).cumcount()                      # earlier rows of this account
    n = n_prior.replace(0, np.nan)
    prev_sum = la.groupby(by).cumsum() - la
    prev_sq = (la ** 2).groupby(by).cumsum() - la ** 2
    mean = prev_sum / n
    std = np.sqrt((prev_sq / n - mean ** 2).clip(lower=0))
    amt_z = ((la - mean) / (std + 0.5)).clip(-5, 5).fillna(0)   # is this amount unusual for them?

    # number of different counterparties seen before
    new_pair = (df.groupby(["src", "dst"]).cumcount() == 0).astype(int)
    uniq_prior = new_pair.groupby(by).cumsum() - new_pair

    # timing: gap since the previous transaction, and burstiness of past gaps
    gap = df["t_sec"].groupby(by).diff()                      # NaN on the first transaction
    gz = gap.fillna(0)
    gc = gap.notna().groupby(by).cumsum().replace(0, np.nan)
    gm = gz.groupby(by).cumsum() / gc
    gs = np.sqrt(((gz ** 2).groupby(by).cumsum() / gc - gm ** 2).clip(lower=0))
    burstiness = ((gs - gm) / (gs + gm)).fillna(0)            # -1 regular ... +1 very bursty

    return pd.DataFrame({
        f"{prefix}_log_n_prior": np.log1p(n_prior),
        f"{prefix}_first_tx": (n_prior == 0).astype(float),
        f"{prefix}_amt_z": amt_z,
        f"{prefix}_log_uniq_cp": np.log1p(uniq_prior),
        f"{prefix}_since_last": np.log1p(gz / 3600.0),
        f"{prefix}_burstiness": burstiness,
    }, index=df.index)


def history_features(df: pd.DataFrame, dormant_days: int = 3) -> pd.DataFrame:
    """All 'what did we know before this transaction' features (Module A inputs)."""
    pt = persistence_table(df)
    pair_n = df.groupby(["src", "dst"]).cumcount()
    pair = pd.DataFrame({"pair_log_prior": np.log1p(pair_n),
                         "pair_is_new": (pair_n == 0).astype(float)}, index=df.index)
    return pd.concat([
        _attach_persistence(df, pt, "src", "s", dormant_days),
        _attach_persistence(df, pt, "dst", "r", dormant_days),
        _role_history(df, "src", "s"),
        _role_history(df, "dst", "r"),
        pair,
    ], axis=1)


# =============================================================================
# 4. Standardisation (fit on training rows only)
# =============================================================================
def fit_standardiser(x: pd.DataFrame, n_train: int):
    mu = x.iloc[:n_train].mean()
    sd = x.iloc[:n_train].std().replace(0, 1.0).fillna(1.0)
    return mu, sd


def apply_standardiser(x: pd.DataFrame, mu, sd, clip=5.0) -> np.ndarray:
    return ((x - mu) / sd).clip(-clip, clip).fillna(0).values.astype(np.float32)


# =============================================================================
# 5. Node snapshots
# =============================================================================
def _zscore(a: np.ndarray, active: np.ndarray) -> np.ndarray:
    mu, sd = a[active].mean(), a[active].std()
    out = (a - mu) / (sd if sd > 0 else 1.0)
    out[~active] = 0.0
    return np.clip(out, -5, 5)


def node_snapshot(g: pd.DataFrame, n_nodes: int, dormant_days: int = 3) -> np.ndarray:
    """
    Summarise every account using only the rows in `g` (the graph visible so far).
    Returns a float32 matrix [n_nodes, len(NODE_COLS)], z-scored per snapshot so the
    scale does not drift as more history accumulates.
    """
    src, dst = g["src"].values, g["dst"].values
    la = np.log1p(g["amt_paid"].values)
    now_day, now_t = g["day"].max(), g["t_sec"].max()

    def cnt(idx, w=None):
        return np.bincount(idx, weights=w, minlength=n_nodes)

    n_out, n_in = cnt(src), cnt(dst)
    n_all = n_out + n_in
    active = n_all > 0

    # distinct counterparties
    pairs = np.unique(src.astype(np.int64) * n_nodes + dst.astype(np.int64))
    uniq = cnt(pairs // n_nodes) + cnt(pairs % n_nodes)

    s_out, s_in = cnt(src, la), cnt(dst, la)
    sum_all = s_out + s_in
    sq_all = cnt(src, la ** 2) + cnt(dst, la ** 2)
    mean_all = sum_all / np.maximum(n_all, 1)
    std_all = np.sqrt(np.maximum(sq_all / np.maximum(n_all, 1) - mean_all ** 2, 0))

    recent = g["t_sec"].values >= now_t - 86400
    n_recent = cnt(src[recent]) + cnt(dst[recent])

    # persistence at snapshot time (latest active-day row of every account)
    pt = persistence_table(g)
    last = pt.groupby("acct").tail(1)
    a = last["acct"].values

    def put(values):
        arr = np.zeros(n_nodes)
        arr[a] = values
        return arr

    active_days = put(last["active_days_prior"].values + 1)
    age = put(now_day - last["first_day"].values + 1)
    last_day = put(last["day"].values)
    recency = np.where(active, now_day - last_day, 0)

    cols = {
        "log_in_deg": np.log1p(n_in), "log_out_deg": np.log1p(n_out),
        "active_days": np.log1p(active_days),
        "active_ratio": active_days / np.maximum(age, 1),
        "streak": np.log1p(put(last["run_len"].values)),
        "recency": np.log1p(recency),
        "age_days": np.log1p(age),
        "dormant": (recency >= dormant_days).astype(float),
        "log_n_out": np.log1p(n_out), "log_n_in": np.log1p(n_in),
        "mean_amt_out": s_out / np.maximum(n_out, 1),
        "mean_amt_in": s_in / np.maximum(n_in, 1),
        "std_amt": std_all,
        "log_uniq_cp": np.log1p(uniq),
        "in_out_ratio": (n_in + 1) / (n_out + 1),
        "log_recent_1d": np.log1p(n_recent),
        "new_cp_frac": uniq / np.maximum(n_all, 1),
    }
    cols["in_out_ratio"] = np.log(cols["in_out_ratio"])
    mat = np.stack([_zscore(np.asarray(cols[c], dtype=np.float64), active) for c in NODE_COLS], axis=1)
    return mat.astype(np.float32)
