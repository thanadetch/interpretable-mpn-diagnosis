"""a77 — Gated-attention bag-rep AUGMENTED with explicit grading-aligned
distribution features (the one untried hybrid).

Rationale: every standalone grading-aligned readout (coverage/quantiles/spread/
moments, a52-a76) UNDERFITS or ties because a single linear projection + tiny
head lacks the baseline's capacity; the baseline gated attention is the
frontier but pools features, possibly discarding the *diffuse-density shape*
the pathologist uses. This model keeps the baseline UNCHANGED and CONCATENATES
a small vector of explicit diffuse-density statistics of the fibrosis
projection s_i = <f_i, v> (v warm-started from the seed=2-train prototype axis)
before the head:

    g = [ mean(s), std(s), q10(s), q50(s), q90(s), coverage=mean(sigmoid((s-tau)/beta)) ]

    y = clamp( Linear([ h_attn (128) ; g (6) ]) , 0, 3 )

Capacity ~= baseline (197K) + v(1280) + 6 head weights, so no capacity blow-up.
The head is initialised so g carries ZERO weight at start (a77 == baseline at
init) and must EARN any contribution -> no capacity confound vs the ablation.

Ablation a78: use_grading_feats=False -> exactly the baseline gated attention,
isolating 'do explicit diffuse-density distribution features add over the
learned attention pooling?'. No ||h|| weighting. forward uses ONLY 'features'.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AXIS_PATH = _REPO_ROOT / "data" / "prototypes_virchow2_reti_train_seed2.pt"


def _load_axis(input_dim: int) -> torch.Tensor:
    try:
        blob = torch.load(_AXIS_PATH, map_location="cpu", weights_only=False)
        ax = blob["axis"].float().view(-1)
        if ax.numel() == input_dim:
            return ax / (ax.norm() + 1e-8)
    except Exception:
        pass
    g = torch.randn(input_dim)
    return g / (g.norm() + 1e-8)


def _soft_quantile(s: torch.Tensor, level: float, rank_tau: float = 0.05, tau: float = 0.05) -> torch.Tensor:
    # differentiable: soft within-bag rank in [0,1], softmax-weight patches near `level`
    N = s.shape[0]
    if N == 1:
        return s.reshape(())  # single patch: quantile == the value (0-dim scalar)
    diff = s.unsqueeze(1) - s.unsqueeze(0)          # [N,N]
    soft_rank = torch.sigmoid(diff / rank_tau).mean(dim=1)  # [N] in (0,1)
    w = F.softmax(-((soft_rank - level) ** 2) / tau, dim=0)  # [N]
    return (w * s).sum()


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        use_grading_feats: bool = True,
    ) -> None:
        super().__init__()
        self.use_grading_feats = use_grading_feats

        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)

        # warm-started fibrosis direction (trainable) + soft-coverage threshold
        self.v = nn.Parameter(_load_axis(input_dim))
        self.tau = nn.Parameter(torch.zeros(1))
        self.log_beta = nn.Parameter(torch.zeros(1))

        n_g = 3 if use_grading_feats else 0
        self.head = nn.Linear(hidden_dim + n_g, num_classes)
        # init so grading feats start at ZERO weight (a77 == baseline at init)
        with torch.no_grad():
            if use_grading_feats:
                self.head.weight[:, hidden_dim:].zero_()

    def _grading_feats(self, features: torch.Tensor) -> torch.Tensor:
        s = features @ self.v  # [N] projection onto fibrosis axis
        mean = s.mean()
        std = s.std(unbiased=False) if s.numel() > 1 else torch.zeros((), device=s.device)
        beta = F.softplus(self.log_beta) + 1e-3
        cov = torch.sigmoid((s - self.tau) / beta).mean()
        # stable diffuse-density triplet (no O(N^2) soft-rank gradient, which
        # de-stabilised the earlier 6-feature version -> degenerate val 0.000)
        return torch.stack([mean, std, cov]).view(1, -1)  # [1,3]

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)            # [N, hidden]
        V = self.attention_V(h)
        U = self.attention_U(h)
        attn = F.softmax(self.attention_W(V * U).squeeze(-1), dim=0)  # [N]
        h_att = torch.mm(attn.unsqueeze(0), h)   # [1, hidden]

        if self.use_grading_feats:
            g = self._grading_feats(features)    # [1, 6]
            rep = torch.cat([h_att, g], dim=1)   # [1, hidden+6]
        else:
            rep = h_att
        logits = self.head(rep)  # [1, num_classes] — raw; trainer rounds+clips at eval (matches baseline)
        if return_attention:
            return logits, attn, None
        return logits, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    use_grading_feats=True,
)
