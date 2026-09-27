"""a152 — Fibrosis-Attention Diffuse Density (the POSITIVE grading-story reweight).

User intuition: keep it simple, just reweight patches to fit the grading story. The flagship
(a135/a148) does the NEGATIVE form (suppress bone/fat distractors). a152 does the POSITIVE
dual: directly UP-WEIGHT patches that score high on the fibrosis concept, then pool. This is
the most literal "attend to fibrosis" story.

  z_fib_i = (fib_i − bag_mean) / GLOBAL_std            # hybrid-normalized fibrosis concept
  w_i     = softmax(λ · z_fib_i)                        # λ=softplus(lam_raw); λ→0 ⇒ uniform mean (=a96)
  z       = Σ w_i·feat_i                                # fibrosis-attended diffuse pool
  y       = w_s·⟨z, v⟩ + c_s                            # project on warm fibrosis axis → grade

CAVEAT (honest): the grading principle is DIFFUSE density (not spotlighting a few patches), so
a large λ would CONTRADICT it (spotlight = the top-k failure mode a11/a12). λ is learnable and
init small; if the model wants λ→0 (uniform) that itself says "fibrosis-attention adds nothing
over diffuse mean". Reads data_distract bag = [feat | fib | bone | adipose]. Warm axis per-fold.
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
_FIB_STD = 0.0281


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
    def __init__(self, input_dim=1280, num_classes=1, warm_start=True):
        super().__init__()
        assert num_classes == 1
        self.feat_dim = input_dim
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
                        amin, amax = float(anch.min()), float(anch.max()); span = max(amax - amin, 1e-6)
                        w_s_init = 3.0 / span; c_s_init = -(3.0 / span) * amin
                except Exception: pass
        if v_init is None:
            v_init = F.normalize(torch.randn(input_dim), dim=0)
        self.v = nn.Parameter(v_init)
        self.w_s = nn.Parameter(torch.tensor(float(w_s_init)))
        self.c_s = nn.Parameter(torch.tensor(float(c_s_init)))
        self.lam_raw = nn.Parameter(torch.tensor(-1.0))   # λ=softplus(lam_raw), init ~0.31 (gentle)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        D = self.feat_dim
        feats = features[:, :D]
        if features.shape[1] >= D + 1:
            fib = features[:, D]
            zf = (fib - fib.mean()) / _FIB_STD
            w = F.softmax(F.softplus(self.lam_raw) * zf, dim=0)
        else:
            w = torch.full((feats.shape[0],), 1.0 / feats.shape[0], device=feats.device)
        z = torch.mv(feats.t(), w)
        y = (self.w_s * torch.dot(z, self.v) + self.c_s).view(-1)
        if return_attention:
            return y, w, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, warm_start=True)
