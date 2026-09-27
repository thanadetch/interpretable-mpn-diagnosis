"""a138 — Spatial Reticulin-Network Structure Aggregator (SRNSA).

The first aggregator in this project that uses the PATCH SPATIAL LAYOUT (every prior
module was permutation-invariant). Motivated directly by the WHO/EUMNET myelofibrosis
grading criteria, where the grade is defined by the NETWORK STRUCTURE of the reticulin
meshwork (MF-0 scattered linear → MF-3 dense continuous network with coarse bundles),
NOT merely by the amount of fibre. So we measure both the AMOUNT (diffuse density along
the fibrosis axis, the a96/a135 signal) AND the SPATIAL CONTINUITY (how spatially
clustered the high-fibrosis patches are on the tile grid).

Input: data_struct bag = [virchow2 1280 | row | col] (1282-d). forward reads ONLY
'features'; the trailing 2 cols are the per-patch (row,col) grid coordinates.

Mechanism:
  s_i   = <feat_i, v>                       # per-patch fibrosis score (v = per-fold train axis)
  density = mean_i s_i                       # AMOUNT (the a96/a135 diffuse-projection signal)
  s_z   = zscore_i(s_i)                      # per-bag standardised
  w_ij  = 1[ |row_i-row_j| + |col_i-col_j| == 1 ]   # 4-neighbour grid adjacency from rc
  moran = (s_z^T W s_z) / sum(W)             # avg neighbour-product = spatial autocorrelation
                                             #   (Moran's I; high ⇒ continuous network)
  y     = w_s*density + w_m*moran + c_s      # ordinal readout
(w_s,c_s) warm-calibrated from per-grade anchors (like a135); w_m init 0 ⇒ a138 starts
EXACTLY as the density-only readout and LEARNS the structure correction.

Ablation a139 = use_structure=False (drops the moran term) ⇒ pure diffuse-density readout
(≈ a96). a138 vs a139 isolates the SOLE question: does spatial network structure add
grade signal over fibre amount? Permutation-invariant ONLY through rc (deterministic),
bag-size-invariant, deterministic at eval.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

_DATA = Path(__file__).resolve().parents[3] / "data"
_BB_BY_DIM = {1280: "virchow2", 1536: "uni2", 768: "titan"}


def _current_seed(default: int = 2) -> int:
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--seed" and i + 1 < len(argv):
            try: return int(argv[i + 1])
            except ValueError: pass
        if a.startswith("--seed="):
            try: return int(a.split("=", 1)[1])
            except ValueError: pass
    return default


def _proto(input_dim: int):
    bb = _BB_BY_DIM.get(input_dim)
    if bb is None: return None
    p = _DATA / f"prototypes_{bb}_reti_train_seed{_current_seed()}.pt"
    return p if p.exists() else None


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, use_structure=True, warm_start=True):
        super().__init__()
        assert num_classes == 1
        self.feat_dim = input_dim
        self.use_structure = use_structure
        v_init = None; w_s_init, c_s_init = 1.0, 1.5
        if warm_start:
            p = _proto(input_dim)
            if p is not None:
                try:
                    blob = torch.load(p, map_location="cpu", weights_only=False)
                    v0 = blob["axis"].float().view(-1)
                    if v0.numel() == input_dim:
                        v0 = v0 / v0.norm().clamp(min=1e-8); v_init = v0.clone()
                        pr = blob["prototypes"]
                        anch = torch.tensor([float(pr[g].float().view(-1) @ v0) for g in range(4)])
                        amin, amax = float(anch.min()), float(anch.max())
                        span = max(amax - amin, 1e-6)
                        w_s_init = 3.0 / span; c_s_init = -(3.0 / span) * amin
                except Exception: pass
        if v_init is None:
            v_init = F.normalize(torch.randn(input_dim), dim=0)
        self.v = nn.Parameter(v_init)
        self.w_s = nn.Parameter(torch.tensor(float(w_s_init)))
        self.c_s = nn.Parameter(torch.tensor(float(c_s_init)))
        self.w_m = nn.Parameter(torch.tensor(0.0))   # structure weight; init 0 ⇒ starts as density-only

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        D = self.feat_dim
        feats = features[:, :D]
        s = feats @ self.v                                  # [N] per-patch fibrosis score
        density = s.mean()
        moran = s.new_zeros(())
        if self.use_structure and features.shape[1] >= D + 2 and s.shape[0] >= 2:
            rc = features[:, D:D + 2]                        # [N,2] (row,col)
            sd = s.std().clamp(min=1e-6)
            sz = (s - s.mean()) / sd                         # [N] standardised
            # 4-neighbour grid adjacency from integer rc
            d1 = (rc[:, 0:1] - rc[:, 0:1].t()).abs()
            d2 = (rc[:, 1:2] - rc[:, 1:2].t()).abs()
            W = ((d1 + d2) == 1).float()                     # [N,N] adjacency (no self)
            Wsum = W.sum()
            if Wsum > 0:
                moran = (sz.unsqueeze(0) @ (W @ sz.unsqueeze(1))).squeeze() / Wsum  # avg neighbour product
        y = (self.w_s * density + self.w_m * moran + self.c_s).view(-1)
        if return_attention:
            return y, None, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, use_structure=True, warm_start=True)
