"""a157 — Fibrosis-Coverage-Fraction (soft-occupancy along the train fibrosis axis).

FULLY SELF-CONTAINED (runs on `--data_root data`, no external file / no pre-run script;
normal imports only). Grading-faithful by construction: the reticulin grade is read as the
*fraction of the marrow space* whose fibrosis-axis score exceeds a learned threshold — a
DIFFUSE, bag-wide occupancy measure where every patch contributes a soft membership, rather
than a155's softmax attention that spikes on the few highest-scoring patches.

  v        = (μ_G3 − μ_G0)/‖·‖    computed IN-MEMORY from TRAIN ROIs only (leakage-safe)
             (reused from a155._train_axis — normal import, no file)
  s_i      = ⟨feat_i, v⟩                                    # absolute fibrosis-axis score (cross-bag comparable)
  cov      = mean_i σ(α·(s_i − τ))                          # soft fraction of "fibrotic" patches (α,τ learnable)
  y        = w·cov + b                                      # monotone map occupancy → ordinal grade (w,b learnable)

Why this shape: the pathology prior says grade = overall/holistic density of the fibre meshwork
across the marrow, NOT a property of a few standout patches. A coverage FRACTION (occupancy) is
the literal expression of that — it counts how much of the bag is fibrotic. τ/α are initialised
from the per-grade train anchors (sensible scale), then all four scalars + v are learnable. Low
DOF (avoids the val-overfit seen with higher-capacity heads on this 10-patient val cohort).
Permutation- and bag-size-invariant (mean over patches), deterministic at inference.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .a155_axis_attention_density import _train_axis, _current_seed


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, warm_start=True):
        super().__init__()
        assert num_classes == 1
        self.feat_dim = input_dim
        v_init = None
        tau_init, alpha_init = 0.0, 1.0
        if warm_start:
            got = _train_axis(input_dim, _current_seed())
            if got is None:
                raise ValueError(f"[a157] unknown input_dim {input_dim} (no backbone mapping for the fibrosis axis).")
            axis, anchors = got
            v_init = axis.float().view(-1).clone()
            amin, amax = float(anchors.min()), float(anchors.max())
            span = max(amax - amin, 1e-6)
            tau_init = 0.5 * (amin + amax)        # threshold at the G0..G3 midpoint
            alpha_init = 4.0 / span               # sigmoid spans the anchor range
        if v_init is None:
            v_init = F.normalize(torch.randn(input_dim), dim=0)
        self.v = nn.Parameter(v_init)
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))
        # alpha = softplus(alpha_raw) > 0  (keeps the soft-threshold monotone increasing)
        a = max(float(alpha_init), 1e-3)
        self.alpha_raw = nn.Parameter(torch.tensor(math.log(math.expm1(a))))
        self.w = nn.Parameter(torch.tensor(3.0))   # coverage 0..1 -> grade 0..3
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        D = self.feat_dim
        feats = features[:, :D]
        s = feats @ self.v                                   # [N] absolute axis score
        alpha = F.softplus(self.alpha_raw)
        memb = torch.sigmoid(alpha * (s - self.tau))         # [N] soft per-patch fibrotic membership
        cov = memb.mean()                                    # diffuse occupancy fraction in (0,1)
        y = (self.w * cov + self.b).view(-1)
        if return_attention:
            return y, memb, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, warm_start=True)
