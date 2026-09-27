"""a141 — Uniformity-Weighted Diffuse Density (UWDD).

Grading-principle-motivated, low-DOF, single-backbone aggregator. Directly tests a
WHO/EUMNET reticulin-grading intuition NOT yet probed: the grade is set not only by
the AMOUNT of fibre (mean density along the fibrosis axis — the a96 signal) but by how
UNIFORM/CONTINUOUS the meshwork is (MF-0 patchy/scattered → MF-3 a uniformly dense,
continuous network). On the patch grid this is the DISPERSION of the per-patch fibrosis
score: low grades = a few hot patches in a quiet background (high dispersion relative to
mean); high grades = uniformly high (the whole bag is dense). We add a single dispersion
term to the diffuse-density readout and let training decide its sign/weight.

  s_i     = <feat_i, v>                  # per-patch fibrosis score (v = per-fold warm axis)
  mu      = mean_i s_i                    # AMOUNT (the a96/a135 diffuse-density signal)
  sigma   = std_i s_i                     # within-bag dispersion of the fibrosis score
  y       = w_s*mu + w_d*sigma + c_s       # ordinal readout (w_d init 0 ⇒ starts = a96)

(w_s,c_s) warm-calibrated from per-grade anchors (like a135); w_d init 0 ⇒ a141 starts
EXACTLY as the density-only readout (a96) and LEARNS whether uniformity adds ordinal
signal beyond mean density. 4 params total — far below the val-overfit regime that sank
the higher-capacity heads. Ablation a142 = use_uniformity=False (drops sigma) ⇒ pure
density readout (≈ a96). a141 vs a142 isolates the SOLE question: does meshwork uniformity
add grade signal over fibre amount? Permutation- & bag-size-invariant; deterministic.

Reads ONLY 'features' (plain `data` root, D-dim). RAW scalar (round+clip downstream).
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
    def __init__(self, input_dim=1280, num_classes=1, use_uniformity=True, warm_start=True):
        super().__init__()
        assert num_classes == 1
        self.feat_dim = input_dim
        self.use_uniformity = use_uniformity
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
        self.w_d = nn.Parameter(torch.tensor(0.0))   # uniformity weight; init 0 ⇒ starts as density-only

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        D = self.feat_dim
        feats = features[:, :D]
        s = feats @ self.v                                  # [N] per-patch fibrosis score
        mu = s.mean()
        disp = s.new_zeros(())
        if self.use_uniformity and s.shape[0] >= 2:
            disp = s.std()                                  # within-bag dispersion of fibrosis score
        y = (self.w_s * mu + self.w_d * disp + self.c_s).view(-1)
        if return_attention:
            return y, None, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, use_uniformity=True, warm_start=True)
