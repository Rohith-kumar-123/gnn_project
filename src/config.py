"""
All settings live here, so every experiment can be reproduced from this one file.

The model variants at the bottom define the baseline, the ablations and the full
HR-GNN. Each variant is just a different combination of on/off switches.
"""
from dataclasses import dataclass, replace


@dataclass
class DataConfig:
    hf_name: str = "bbfizp/AMLSim-HI-Small"   # Hugging Face dataset id
    csv_path: str | None = None               # use a local CSV instead of Hugging Face
    cache_dir: str = "data/processed"
    train_frac: float = 0.6                   # chronological 60 / 20 / 20 split
    val_frac: float = 0.2
    # DEBUG ONLY: keep this fraction of normal transactions (all fraud is kept).
    # It changes the fraud rate, so never report results from a dev run.
    dev_normal_frac: float | None = None
    dormant_days: int = 3                     # "dormant, then active again" threshold


@dataclass
class ModelConfig:
    hidden: int = 32
    n_layers: int = 2
    dropout: float = 0.2
    use_hist: bool = False         # Module A: history features (node snapshot + per-edge history)
    use_persistence: bool = False  # persistence gate p_v used as a trust prior in aggregation
    use_reliability: bool = False  # Module B: learned edge reliability r_uv
    use_time_decay: bool = False   # exp(-lambda * age of the edge) in aggregation
    dual_channel: bool = False     # separate channels for reliable / unreliable edges


@dataclass
class TrainConfig:
    epochs: int = 1000
    lr: float = 3e-3
    weight_decay: float = 1e-5
    patience: int = 40             # early stopping on validation AP
    focal_gamma: float = 2.0
    focal_alpha: float = 0.5       # 0.5 = no extra class weighting (negative sampling already balances)
    neg_ratio: int = 50            # normal edges sampled per fraud edge each epoch (0 = use all)
    beta_rel: float = 0.1          # weight of the reliability (weak supervision) loss
    gamma_entropy: float = 0.01    # weight of the entropy term on r
    device: str = "auto"           # "auto" -> cuda if available, else cpu


# ----------------------------------------------------------------------------
# Experiment variants (baseline, ablations, full model)
# ----------------------------------------------------------------------------
FULL = ModelConfig(use_hist=True, use_persistence=True, use_reliability=True,
                   use_time_decay=True, dual_channel=True)

VARIANTS: dict[str, ModelConfig] = {
    # plain GCN baseline: structure-only node inputs, no history, uniform edge weights
    "gcn": ModelConfig(),
    # isolates Module A (history features only)
    "gcn_hist": ModelConfig(use_hist=True),
    # isolates Module B (edge reliability only, no history)
    "gcn_reliability": ModelConfig(use_reliability=True),
    # ablations of the full model
    "hrgnn_no_decay": replace(FULL, use_time_decay=False),
    "hrgnn_no_persistence": replace(FULL, use_persistence=False),
    "hrgnn_single_channel": replace(FULL, dual_channel=False),
    # full proposed model
    "hrgnn": FULL,
}
