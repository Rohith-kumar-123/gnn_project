"""
Turns the prepared dataframe into tensors for each phase.

Time-respecting setup (rows are sorted by time):

    phase  | message-passing graph | edges we classify (targets)
    -------+-----------------------+----------------------------
    train  | rows [0, n_train)     | rows [0, n_train)
    val    | rows [0, n_val_end)   | rows [n_train, n_val_end)
    test   | rows [0, n_total)     | rows [n_val_end, n_total)

So when we score a test transaction, the graph contains only transactions up to the
end of the test period and no labels are ever used at inference time.
"""
from dataclasses import dataclass

import numpy as np
import torch

from .config import ModelConfig
from . import features as F


@dataclass
class Phase:
    name: str
    node_x: torch.Tensor                 # [N, f]  inputs of the node encoder
    node_persist: torch.Tensor | None    # [N, k]  persistence columns (only if use_persistence)
    src: torch.Tensor                    # [E]
    dst: torch.Tensor                    # [E]
    edge_x: torch.Tensor                 # [E, Fe] view of the shared edge-feature matrix
    dt_days: torch.Tensor                # [E]     age of every edge relative to "now"
    y: torch.Tensor                      # [E]
    start: int                           # first target row
    end: int                             # last target row + 1


def make_phases(prep, mcfg: ModelConfig, device) -> tuple[dict, torch.Tensor]:
    """Returns ({'train': Phase, 'val': Phase, 'test': Phase}, weak_suspicious_train)."""
    df = prep.df
    n_nodes = prep.n_nodes

    # edge features: transaction features always, history features only for Module A variants
    edge_np = np.concatenate([prep.edge_tx, prep.edge_hist], axis=1) if mcfg.use_hist else prep.edge_tx
    edge_x = torch.from_numpy(edge_np).to(device)
    src_all = torch.from_numpy(df["src"].to_numpy().copy()).long().to(device)
    dst_all = torch.from_numpy(df["dst"].to_numpy().copy()).long().to(device)
    y_all = torch.from_numpy(df["y"].to_numpy().copy()).float().to(device)
    t_all = torch.from_numpy(df["t_sec"].to_numpy().copy()).float().to(device)

    ix = {c: i for i, c in enumerate(F.NODE_COLS)}
    struct_ix = [ix[c] for c in F.STRUCT_COLS]
    persist_ix = [ix[c] for c in F.PERSIST_COLS]
    hist_ix = persist_ix + [ix[c] for c in F.BEHAV_COLS]

    phases = {}
    for name in ["train", "val", "test"]:
        start, end = prep.bounds(name)
        snap = torch.from_numpy(prep.snapshots[name]).to(device)
        node_x = snap[:, struct_ix + hist_ix] if mcfg.use_hist else snap[:, struct_ix]
        node_persist = snap[:, persist_ix] if mcfg.use_persistence else None
        dt = (t_all[end - 1] - t_all[:end]) / 86400.0
        phases[name] = Phase(name, node_x, node_persist, src_all[:end], dst_all[:end],
                             edge_x[:end], dt, y_all[:end], start, end)

    # Weak supervision for reliability: an edge is "suspicious" if either endpoint
    # was involved in a laundering transaction in the TRAINING period.
    tr = phases["train"]
    fraud_nodes = torch.unique(torch.cat([tr.src[tr.y == 1], tr.dst[tr.y == 1]]))
    is_fraud_node = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    is_fraud_node[fraud_nodes] = True
    weak = (is_fraud_node[tr.src] | is_fraud_node[tr.dst]).float()
    return phases, weak
