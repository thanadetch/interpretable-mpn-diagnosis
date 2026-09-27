"""a148 — Multi-Distractor-Avoiding Diffuse Density (bone AND adipose).

Extends the faithful flagship (a135/a145 = bone-avoiding diffuse density) to suppress BOTH
non-fibrosis distractors the pathologist ignores: bone trabeculae AND fat (open adipose).
Motivated by the faithfulness audit: baseline gated-attention correlates with bone (+0.30)
AND adipose (+0.23) — both should be avoided. Hybrid concept-norm (per-bag center + global
scale), the normalization that won round-31.

  zf,zb,za = hybrid-normalized fibrosis / bone / adipose concept scores
  w = softmax( −softplus(λ_b)·relu(zb − zf) − softplus(λ_a)·relu(za − zf) )   # down-weight bone- OR fat-dominant patches
  z = Σ wᵢ·featᵢ ;  y = w_s·⟨z, v⟩ + c_s                                       # diffuse pool → fibrosis axis → grade

Tests the faithfulness-vs-grade TRADEOFF: adipose correlates +0.60 with grade, so suppressing
fat-dominant patches MAY cost grade signal even as it improves faithfulness. Ablation a149 =
use_adipose=False (λ_a dropped) = bone-only (= a145). a148 vs a149 isolates adding fat-avoidance.
Reads data_distract bag = [feat | fibrosis | bone | adipose]. Low-DOF, warm fibrosis axis per-fold.
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
    def __init__(self, input_dim=1280, num_classes=1, use_adipose=True, warm_start=True):
        super().__init__()
        assert num_classes == 1
        self.feat_dim = input_dim
        self.use_adipose = use_adipose
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
        self.lam_b = nn.Parameter(torch.tensor(0.0))   # bone-avoidance strength
        self.lam_a = nn.Parameter(torch.tensor(0.0))   # adipose-avoidance strength

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        D = self.feat_dim
        feats = features[:, :D]
        if features.shape[1] >= D + 2:
            fib = features[:, D]; bone = features[:, D + 1]
            zf = (fib - fib.mean()) / _FIB_STD
            zb = (bone - bone.mean()) / _BONE_STD
            logit = -F.softplus(self.lam_b) * F.relu(zb - zf)
            if self.use_adipose and features.shape[1] >= D + 3:
                adi = features[:, D + 2]
                za = (adi - adi.mean()) / _ADI_STD
                logit = logit - F.softplus(self.lam_a) * F.relu(za - zf)
            w = F.softmax(logit, dim=0)
        else:
            w = torch.full((feats.shape[0],), 1.0 / feats.shape[0], device=feats.device)
        z = torch.mv(feats.t(), w)
        y = (self.w_s * torch.dot(z, self.v) + self.c_s).view(-1)
        if return_attention:
            return y, w, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, use_adipose=True, warm_start=True)
