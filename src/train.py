"""
Train, evaluate, save and reuse one GNN variant.

    run_gnn(...)   trains one variant/seed, with crash-safe checkpoints and final-model saving
    load_model()   rebuilds a saved model
    predict()      scores every transaction of a phase -> a DataFrame ready for a dashboard

Loss = focal(edge classification)
       + beta  * reliability loss  (weak supervision, training labels only)
       + gamma * entropy of r      (makes reliability decisive instead of "mushy 0.5")
"""
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from .config import ModelConfig, TrainConfig
from .graph import make_phases
from .metrics import best_f1_threshold, evaluate
from .model import HRGNN, focal_loss
from . import features as FE


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _save(path: Path, state: dict):
    """Write to a temp file, then rename: a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def score_phase(model, ph):
    """Fraud scores (and reliability) for the target edges of a phase."""
    model.eval()
    idx = torch.arange(ph.start, ph.end, device=ph.src.device)
    logits, r, _ = model(ph, idx)
    score = torch.sigmoid(logits).cpu().numpy()
    r_t = r[idx].cpu().numpy() if r is not None else None
    return score, r_t, ph.y[idx].cpu().numpy()


def sample_batch(y_train: torch.Tensor, neg_ratio: int, device):
    """All fraud edges + `neg_ratio` random normal edges per fraud edge."""
    pos = torch.nonzero(y_train == 1).squeeze(-1)
    neg = torch.nonzero(y_train == 0).squeeze(-1)
    if neg_ratio > 0:
        k = min(len(neg), neg_ratio * max(len(pos), 1))
        neg = neg[torch.randperm(len(neg), device=device)[:k]]
    return torch.cat([pos, neg])


# -----------------------------------------------------------------------------
def run_gnn(prep, name: str, mcfg: ModelConfig, tcfg: TrainConfig, seed: int, verbose=True,
            ckpt_dir=None, model_dir=None, ckpt_every: int = 25):
    """
    ckpt_dir  : progress is saved here every `ckpt_every` epochs; a rerun resumes from it
                (deleted when the run finishes)
    model_dir : the final best-validation model is saved here (kept, used by predict())
    """
    set_seed(seed)
    device = get_device(tcfg.device)
    phases, weak = make_phases(prep, mcfg, device)
    tr, va, te = phases["train"], phases["val"], phases["test"]

    n_persist = len(FE.PERSIST_COLS)
    model = HRGNN(mcfg, tr.node_x.size(1), n_persist, tr.edge_x.size(1)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)

    best_ap, best_state, best_epoch, bad = -1.0, None, 0, 0
    start_epoch, elapsed_before = 1, 0.0
    ckpt_path = Path(ckpt_dir) / f"{name}_s{seed}.pt" if ckpt_dir else None

    if ckpt_path is not None and ckpt_path.exists():          # resume an interrupted run
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        best_ap, best_state = ck["best_ap"], ck["best_state"]
        best_epoch, bad = ck["best_epoch"], ck["bad"]
        start_epoch, elapsed_before = ck["epoch"] + 1, ck["elapsed"]
        torch.set_rng_state(ck["rng_cpu"].cpu())
        if ck.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(ck["rng_cuda"].cpu())
        print(f"  [{name} s{seed}] RESUMING from checkpoint at epoch {ck['epoch']} (best val AP {best_ap:.4f})")

    t0 = time.time()
    for epoch in range(start_epoch, tcfg.epochs + 1):
        model.train()
        idx = sample_batch(tr.y, tcfg.neg_ratio, device)
        logits, r, _ = model(tr, idx)
        loss = focal_loss(logits, tr.y[idx], tcfg.focal_gamma, tcfg.focal_alpha)

        if r is not None:
            # weak supervision: edges next to known-fraud accounts should get LOW reliability
            target = 1.0 - weak
            w = torch.where(weak == 1, 0.5 / weak.mean().clamp_min(1e-6),
                            0.5 / (1 - weak.mean()).clamp_min(1e-6))   # balance both groups
            rel_loss = F.binary_cross_entropy(r.clamp(1e-6, 1 - 1e-6), target, weight=w)
            entropy = -(r * torch.log(r + 1e-6) + (1 - r) * torch.log(1 - r + 1e-6)).mean()
            loss = loss + tcfg.beta_rel * rel_loss + tcfg.gamma_entropy * entropy

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        # validation (graph = train + val edges, targets = val edges)
        v_score, _, v_y = score_phase(model, va)
        v_ap = average_precision_score(v_y, v_score) if v_y.sum() else 0.0
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  [{name} s{seed}] epoch {epoch:3d}  loss {loss.item():.4f}  val AP {v_ap:.4f}")

        stop = False
        if v_ap > best_ap:
            best_ap, best_epoch, bad = v_ap, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            stop = bad >= tcfg.patience

        if ckpt_path is not None and (epoch % ckpt_every == 0) and not stop:
            _save(ckpt_path, {
                "model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                "best_ap": best_ap, "best_state": best_state, "best_epoch": best_epoch, "bad": bad,
                "elapsed": elapsed_before + time.time() - t0,
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None})
        if stop:
            break

    train_time = elapsed_before + time.time() - t0
    model.load_state_dict(best_state)

    v_score, _, v_y = score_phase(model, va)
    t_score, t_r, t_y = score_phase(model, te)
    thr = best_f1_threshold(v_y, v_score)               # chosen on validation, frozen for test
    result = {
        "model": name, "seed": seed, "best_epoch": best_epoch, "n_params": n_params,
        "train_seconds": train_time,
        "val": evaluate(v_y, v_score, thr),
        "test": evaluate(t_y, t_score, thr),
    }
    if mcfg.use_time_decay:
        result["lambda_per_day"] = float(F.softplus(model.raw_lambda).item())

    if model_dir is not None:                           # keep the final model for the dashboard
        _save(Path(model_dir) / f"{name}_s{seed}.pt", {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "model_cfg": asdict(mcfg), "threshold": thr, "best_epoch": best_epoch,
            "dims": {"node_dim": tr.node_x.size(1), "persist_dim": n_persist, "edge_dim": tr.edge_x.size(1)}})
    if ckpt_path is not None and ckpt_path.exists():    # finished: crash checkpoint no longer needed
        ckpt_path.unlink()

    arrays = {"y": t_y, "score": t_score, "row": np.arange(te.start, te.end)}
    if t_r is not None:
        arrays["r"] = t_r
    return result, arrays


# -----------------------------------------------------------------------------
# Using a saved model (dashboard / inference)
# -----------------------------------------------------------------------------
def load_model(path, device="cpu"):
    """Rebuild a saved model. Returns (model, checkpoint_dict)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model = HRGNN(ModelConfig(**ck["model_cfg"]), **ck["dims"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def predict(prep, model_path, phase: str = "test", device: str = "auto") -> pd.DataFrame:
    """
    Score every transaction of a phase ('train' / 'val' / 'test') with a saved model.
    Returns one row per transaction: who paid whom, when, how much, the fraud score,
    whether it crosses the alert threshold, the edge reliability (HR-GNN) and the true label.
    """
    dev = get_device(device)
    model, ck = load_model(model_path, dev)
    phases, _ = make_phases(prep, ModelConfig(**ck["model_cfg"]), dev)
    ph = phases[phase]
    score, r, y = score_phase(model, ph)
    d = prep.df.iloc[ph.start:ph.end]
    out = pd.DataFrame({
        "row": np.arange(ph.start, ph.end), "timestamp": d["timestamp"].values,
        "from_account": d["src_acct"].values, "to_account": d["dst_acct"].values,
        "amount_paid": d["amt_paid"].values, "payment_format": d["pay_format"].values,
        "fraud_score": score, "alert": score >= ck["threshold"], "label": y.astype(int)})
    if r is not None:
        out["reliability"] = r
    return out
