"""
Metrics for extremely imbalanced fraud detection.

AP (average precision) is the main metric. ROC-AUC is reported too, but with ~0.1%
fraud it looks flattering, so we also report what an analyst actually experiences:
how many false alerts per 1,000 transactions, and how much fraud is caught if only a
fixed share of transactions can be reviewed (the "alert budget").
"""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ALERT_BUDGETS = (0.001, 0.005, 0.01)   # review 0.1%, 0.5%, 1% of transactions


def best_f1_threshold(y, score) -> float:
    """Threshold that maximises F1. Call this on VALIDATION data only."""
    if y.sum() == 0:
        return float(np.quantile(score, 0.999))
    p, r, thr = precision_recall_curve(y, score)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(thr[np.argmax(f1)])


def recall_at_budget(y, score, rate: float) -> float:
    k = max(1, int(np.ceil(rate * len(y))))
    top = np.argpartition(-score, k - 1)[:k]
    return float(y[top].sum() / max(y.sum(), 1))


def evaluate(y, score, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    pred = score >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    out = {
        "ap": float(average_precision_score(y, score)) if y.sum() else float("nan"),
        "auc": float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan"),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "threshold": float(threshold),
        "alerts_per_1000": float(pred.mean() * 1000),
        "false_alerts_per_1000": float(fp / len(y) * 1000),
    }
    for rate in ALERT_BUDGETS:
        out[f"recall@{rate:.1%}"] = recall_at_budget(y, score, rate)
    return out
