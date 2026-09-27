"""a150 — OPAL: Ordinal Per-patch Anchored Latent-CDF pooling (workflow-designed).

Genuinely-new ordinal mechanism, distinct from a96 (which pools THEN projects). OPAL applies
a per-patch soft ORDINAL membership FIRST, then pools the per-patch grade-CDFs — so it is
sensitive to the FOCAL-vs-DIFFUSE shape of the within-bag fibrosis distribution (a few intense
patches vs many moderate ones), which the workflow argued lives at titan's boundary errors.

  s_i   = ⟨feat_i, v⟩                                  # per-patch fibrosis score (warm axis)
  a_0..3 = ⟨prototype_g, v⟩ (sorted)                   # per-fold grade anchors along the axis
  t_k   = (a_{k-1}+a_k)/2 , k=1,2,3                    # ordinal thresholds between consecutive grades
  P_i(grade≥k) = σ((s_i − t_k)/β)                       # per-patch soft cumulative membership
  C_k   = mean_i P_i(grade≥k)                           # bag-level grade CDF (pool the per-patch CDFs)
  y     = w_s·(Σ_{k=1}^{3} C_k) + c_s                   # cumulative-link expected grade ∈ ~[0,3]

β (sharpness), w_s, c_s, v learnable; β init→ moderate. CDF-then-pool (OPAL) vs pool-then-CDF
(ablation a151) isolates whether the per-patch ordinal nonlinearity adds focal/diffuse signal
over the mean. Reads plain `data` (needs only feats + per-fold prototype). Deterministic,
permutation-/bag-size-invariant.
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
    def __init__(self, input_dim=1280, num_classes=1, per_patch_cdf=True, warm_start=True):
        super().__init__()
        assert num_classes == 1
        self.feat_dim = input_dim
        self.per_patch_cdf = per_patch_cdf
        v_init = None; thr = torch.tensor([0.5, 1.5, 2.5])
        if warm_start:
            p = _proto(input_dim)
            if p is not None:
                try:
                    blob = torch.load(p, map_location="cpu", weights_only=False)
                    v0 = blob["axis"].float().view(-1)
                    if v0.numel() == input_dim:
                        v0 = v0 / v0.norm().clamp(min=1e-8); v_init = v0.clone()
                        pr = blob["prototypes"]
                        anch = torch.tensor(sorted(float(pr[g].float().view(-1) @ v0) for g in range(4)))
                        thr = torch.tensor([(anch[k - 1] + anch[k]) / 2 for k in (1, 2, 3)])
                except Exception: pass
        if v_init is None:
            v_init = F.normalize(torch.randn(input_dim), dim=0)
        self.v = nn.Parameter(v_init)
        self.thr = nn.Parameter(thr.float())               # 3 ordinal thresholds
        self.beta_raw = nn.Parameter(torch.tensor(-1.0))   # β = softplus(beta_raw)+eps (init ~0.31)
        self.w_s = nn.Parameter(torch.tensor(1.0))
        self.c_s = nn.Parameter(torch.tensor(0.0))

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        D = self.feat_dim
        feats = features[:, :D]
        s = feats @ self.v                                  # [N] per-patch fibrosis score
        beta = F.softplus(self.beta_raw) + 1e-3
        if self.per_patch_cdf:
            # OPAL: per-patch CDF then pool  -> [N,3] -> mean over patches -> [3]
            P = torch.sigmoid((s.unsqueeze(1) - self.thr.unsqueeze(0)) / beta)  # [N,3]
            C = P.mean(dim=0)                               # [3] bag grade-CDF
        else:
            # ablation a151: pool then CDF (mean density -> CDF)
            mu = s.mean()
            C = torch.sigmoid((mu - self.thr) / beta)       # [3]
        y = (self.w_s * C.sum() + self.c_s).view(-1)
        if return_attention:
            return y, None, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, per_patch_cdf=True, warm_start=True)
