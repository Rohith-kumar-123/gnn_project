"""
HR-GNN: Historical- and Relational-aware GNN (pure PyTorch, no PyG needed).

One class covers the plain GCN baseline, every ablation, and the full model;
the switches in ModelConfig decide which parts are active.

    A  node encoder       node features  -> h_v           (history embedding)
       persistence gate   persistence    -> p_v in [0,1]  (trust prior)
    B  edge reliability   r_uv = sigmoid(MLP([h_u, h_v, e_uv, p_u, p_v, |p_u-p_v|, dt]))
    C  message passing    h_v' = relu(W_self h_v + sum_u alpha_uv * r_uv * W h_u)
                          alpha_uv = normalise( p_u * exp(-lambda * dt_uv) )
    D  edge classifier    y_uv = sigmoid(MLP([z_u, z_v, e_uv, r_uv]))

Messages flow along the money direction AND against it (separate weights), because
a fraud ring is visible from both the sender side and the receiver side.
With dual_channel=True, a second channel aggregates with weights (1 - r), so the
"unreliable neighbourhood" is kept as a signal instead of being thrown away.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def scatter_sum(values: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = values.new_zeros((n,) + tuple(values.shape[1:]))
    return out.index_add_(0, index, values)


def normalise_by_target(a: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Make the weights of all messages arriving at the same node sum to 1."""
    return a / scatter_sum(a, index, n)[index].clamp_min(1e-8)


class EdgeReliability(nn.Module):
    """
    r_uv = sigmoid(MLP([m_u, m_v, edge features])).
    The first linear layer is split into a sender part, a receiver part and an edge part.
    This is mathematically the same as one MLP on the concatenation, but the node
    projections are computed once per node instead of once per edge (big memory saving).
    """

    def __init__(self, node_dim: int, edge_dim: int, extra_dim: int, hidden: int = 32):
        super().__init__()
        self.w_src = nn.Linear(node_dim, hidden)
        self.w_dst = nn.Linear(node_dim, hidden, bias=False)
        self.w_edge = nn.Linear(edge_dim, hidden, bias=False)
        self.w_extra = nn.Linear(extra_dim, hidden, bias=False)
        self.out = nn.Linear(hidden, 1)

    def forward(self, m, src, dst, edge_x, extra):
        h = self.w_src(m)[src] + self.w_dst(m)[dst] + self.w_edge(edge_x) + self.w_extra(extra)
        return torch.sigmoid(self.out(F.relu(h))).squeeze(-1)


class ReliabilityConv(nn.Module):
    """One message-passing layer. `channels` is a list of (alpha_fwd, alpha_bwd) edge weights."""

    def __init__(self, dim: int, n_channels: int):
        super().__init__()
        self.self_lin = nn.Linear(dim, dim)
        self.fwd = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(n_channels)])
        self.bwd = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(n_channels)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, src, dst, channels):
        n = x.size(0)
        out = self.self_lin(x)
        for lin_f, lin_b, (w_f, w_b) in zip(self.fwd, self.bwd, channels):
            out = out + scatter_sum(lin_f(x)[src] * w_f[:, None], dst, n)   # sender -> receiver
            out = out + scatter_sum(lin_b(x)[dst] * w_b[:, None], src, n)   # receiver -> sender
        return F.relu(self.norm(out))


class HRGNN(nn.Module):
    def __init__(self, cfg: ModelConfig, node_dim: int, persist_dim: int, edge_dim: int):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden

        # Module A
        self.node_enc = nn.Sequential(nn.Linear(node_dim, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
        if cfg.use_persistence:
            assert cfg.use_hist, "persistence gate needs the history features (use_hist=True)"
            self.gate = nn.Sequential(nn.Linear(persist_dim, 16), nn.ReLU(), nn.Linear(16, 1))

        # time decay: lambda = softplus(raw) per day, learned (starts at ~0.13 / day)
        if cfg.use_time_decay:
            self.raw_lambda = nn.Parameter(torch.tensor(-2.0))

        # Module B
        if cfg.use_reliability:
            node_in = H + (1 if cfg.use_persistence else 0)
            extra = 1 + (1 if cfg.use_persistence else 0)       # dt  (+ |p_u - p_v|)
            self.rel = EdgeReliability(node_in, edge_dim, extra)

        # Module C
        n_ch = 2 if (cfg.use_reliability and cfg.dual_channel) else 1
        self.convs = nn.ModuleList([ReliabilityConv(H, n_ch) for _ in range(cfg.n_layers)])
        self.drop = nn.Dropout(cfg.dropout)

        # Module D
        head_in = 2 * H + edge_dim + (1 if cfg.use_reliability else 0)
        self.head = nn.Sequential(nn.Linear(head_in, 64), nn.ReLU(), nn.Dropout(cfg.dropout),
                                  nn.Linear(64, 1))

    # ------------------------------------------------------------------
    def forward(self, ph, idx: torch.Tensor):
        """
        ph  : Phase (graph tensors)
        idx : indices (into the phase's edge list) of the edges to classify
        returns (logits for idx, reliability r for ALL graph edges or None, persistence p or None)
        """
        cfg = self.cfg
        n = ph.node_x.size(0)
        src, dst = ph.src, ph.dst

        h = self.node_enc(ph.node_x)
        p = torch.sigmoid(self.gate(ph.node_persist)).squeeze(-1) if cfg.use_persistence else None

        # --- Module B: edge reliability ---------------------------------
        r = None
        if cfg.use_reliability:
            m = h if p is None else torch.cat([h, p[:, None]], dim=1)
            extra = [ph.dt_days[:, None] / 10.0]
            if p is not None:
                extra.append((p[src] - p[dst]).abs()[:, None])
            r = self.rel(m, src, dst, ph.edge_x, torch.cat(extra, dim=1))

        # --- Module C: aggregation weights ------------------------------
        base = torch.ones_like(ph.dt_days)
        if cfg.use_time_decay:
            base = torch.exp(-F.softplus(self.raw_lambda) * ph.dt_days)
        a_fwd = base * (p[src] if p is not None else 1.0)      # trust the sender
        a_bwd = base * (p[dst] if p is not None else 1.0)      # trust the receiver
        alpha_f = normalise_by_target(a_fwd, dst, n)
        alpha_b = normalise_by_target(a_bwd, src, n)

        if r is None:
            channels = [(alpha_f, alpha_b)]
        elif cfg.dual_channel:
            channels = [(alpha_f * r, alpha_b * r), (alpha_f * (1 - r), alpha_b * (1 - r))]
        else:
            channels = [(alpha_f * r, alpha_b * r)]

        z = h
        for conv in self.convs:
            z = self.drop(conv(z, src, dst, channels))

        # --- Module D: edge classifier ----------------------------------
        s, d = src[idx], dst[idx]
        parts = [z[s], z[d], ph.edge_x[idx]]
        if r is not None:
            parts.append(r[idx][:, None])
        logits = self.head(torch.cat(parts, dim=1)).squeeze(-1)
        return logits, r, p


def focal_loss(logits, y, gamma: float, alpha: float):
    """Focal loss: down-weights easy (mostly normal) examples."""
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    pt = torch.exp(-bce)
    a = alpha * y + (1 - alpha) * (1 - y)
    return (a * (1 - pt) ** gamma * bce).mean()
