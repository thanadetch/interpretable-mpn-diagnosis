"""a146 — Selectivity-Disentangled Occupancy + Diffuse Density (SDOP, workflow-designed).

Empirical test of whether a GENUINELY-NONLINEAR, provably-NOT-mean-reducible per-patch
functional can add grade signal beyond diffuse density (i.e. does mean-sufficiency extend
from linear/moment functionals to nonlinear ones?). Grading rationale: grade ~ the EXTENT
of genuine fibrotic tissue = fraction of patches that are strongly fibrosis-selective AND
fibrosis-DOMINANT over the two distractors a pathologist ignores (bone trabeculae, fat).

  density  = mean_i ⟨feat_i, v⟩                                   # diffuse fibrosis density (a96 signal)
  z_*      = (concept − bag_mean) / GLOBAL_std                    # hybrid norm (per-bag center + global scale)
  sel_i    = softplus(z_fib_i)·σ(z_fib_i − max(z_bone_i, z_adip_i))   # fibrosis-selective contrast
  occ      = mean_i σ((sel_i − τ)/β)                              # NONLINEAR occupancy (not a fn of mean_i feat)
  y        = w_s·density + w_o·occ + c_s                           # w_o init 0 ⇒ starts = pure density (a96)

The max() and product make `occ` provably NOT recoverable from per-bag feature means, so if
w_o stays ~0 / hurts, mean-sufficiency holds even for nonlinear functionals (strong result).
Ablation a147 = use_occupancy=False ⇒ pure diffuse density (≈ a96). Reads data_distract bag
= [feat | fibrosis | bone | adipose] (1283/1539/771-d). Warm fibrosis axis per-fold.
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
# GLOBAL std of TITAN zero-shot concept scores (all 1330 bags); hybrid norm uses per-bag mean + these stds
_FIB_STD, _BONE_STD, _ADI_STD = 0.0281, 0.0193, 0.0179


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
    def __init__(self, input_dim=1280, num_classes=1, use_occupancy=True, warm_start=True):
        super().__init__()
        assert num_classes == 1
        self.feat_dim = input_dim
        self.use_occupancy = use_occupancy
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
        self.w_o = nn.Parameter(torch.tensor(0.0))        # occupancy weight; init 0 ⇒ starts = pure density
        self.tau = nn.Parameter(torch.tensor(0.0))        # occupancy threshold
        self.beta_raw = nn.Parameter(torch.tensor(0.0))   # sharpness (β = softplus+eps)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        D = self.feat_dim
        feats = features[:, :D]
        density = (feats @ self.v).mean()
        occ = density.new_zeros(())
        if self.use_occupancy and features.shape[1] >= D + 3:
            fib = features[:, D]; bone = features[:, D + 1]; adi = features[:, D + 2]
            zf = (fib - fib.mean()) / _FIB_STD
            zb = (bone - bone.mean()) / _BONE_STD
            za = (adi - adi.mean()) / _ADI_STD
            sel = F.softplus(zf) * torch.sigmoid(zf - torch.maximum(zb, za))   # fibrosis-selective contrast
            beta = F.softplus(self.beta_raw) + 1e-3
            occ = torch.sigmoid((sel - self.tau) / beta).mean()                # nonlinear occupancy
        y = (self.w_s * density + self.w_o * occ + self.c_s).view(-1)
        if return_attention:
            return y, None, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, use_occupancy=True, warm_start=True)
