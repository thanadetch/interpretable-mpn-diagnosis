"""a135 — PER-FOLD, leakage-safe version of a131 (calibrated bone-aware diffuse pool).

a131 hardcoded the seed=2 prototype file for BOTH the warm fibrosis axis and the
affine calibration, so it is only honest at seed=2. a135 reads `--seed` from sys.argv
(like a115) and loads `prototypes_virchow2_reti_train_seed{seed}.pt` — each fold uses
ONLY its own train split's axis + per-grade anchors -> NO leakage. This lets us measure
the TRUE 5-fold MEAN test-QWK of the grading-aligned, low-DOF aggregator (~1283 params)
against the baseline's noisy 0.794 mean.

Mechanism (1282-d data_bonefib bag = [virchow2 1280 | bone | fibrosis]):
  bone-aware diffuse weights w = softmax(-softplus(lam)*relu(bone_z - fib_z))
  z = sum_i w_i feats_i ; s = <z, v> ; y = w_s*s + c_s
  v warm = per-fold train fibrosis axis; (w_s,c_s) calibrated from per-fold anchors.
Learnable: v(1280)+lam(1)+w_s(1)+c_s(1). forward reads ONLY 'features'. RAW logit.

Ablation a136 = bone_aware=False (per-fold calibrated diffuse projection ~ a96).
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
_BONE_MEAN, _BONE_STD = -0.0486, 0.0193
_FIB_MEAN, _FIB_STD = 0.0045, 0.0281


def _current_seed(default: int = 2) -> int:
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--seed" and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except ValueError:
                pass
        if a.startswith("--seed="):
            try:
                return int(a.split("=", 1)[1])
            except ValueError:
                pass
    return default


def _proto_path(input_dim: int) -> Optional[Path]:
    bb = _BB_BY_DIM.get(input_dim)
    if bb is None:
        return None
    p = _DATA / f"prototypes_{bb}_reti_train_seed{_current_seed()}.pt"
    return p if p.exists() else None


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, bone_aware=True, warm_start=True):
        super().__init__()
        assert num_classes == 1
        self.feat_dim = input_dim
        self.bone_aware = bone_aware
        v_init = None
        w_s_init, c_s_init = 1.0, 1.5
        if warm_start:
            p = _proto_path(input_dim)
            if p is not None:
                try:
                    blob = torch.load(p, map_location="cpu", weights_only=False)
                    v0 = blob["axis"].float().view(-1)
                    if v0.numel() == input_dim:
                        v0 = v0 / v0.norm().clamp(min=1e-8)
                        v_init = v0.clone()
                        protos = blob["prototypes"]
                        anchors = torch.tensor([float(protos[g].float().view(-1) @ v0) for g in range(4)])
                        a_min, a_max = float(anchors.min()), float(anchors.max())
                        span = max(a_max - a_min, 1e-6)
                        w_s_init = 3.0 / span
                        c_s_init = -(3.0 / span) * a_min
                except Exception:
                    pass
        if v_init is None:
            v_init = F.normalize(torch.randn(input_dim), dim=0)
        self.v = nn.Parameter(v_init)
        self.lam_raw = nn.Parameter(torch.tensor(0.0))
        self.w_s = nn.Parameter(torch.tensor(float(w_s_init)))
        self.c_s = nn.Parameter(torch.tensor(float(c_s_init)))

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        D = self.feat_dim
        feats = features[:, :D]
        if self.bone_aware and features.shape[1] >= D + 2:
            bone = (features[:, D] - _BONE_MEAN) / _BONE_STD
            fib = (features[:, D + 1] - _FIB_MEAN) / _FIB_STD
            w = F.softmax(-F.softplus(self.lam_raw) * F.relu(bone - fib), dim=0)
        else:
            w = torch.full((feats.shape[0],), 1.0 / feats.shape[0], device=feats.device)
        z = torch.mv(feats.t(), w)
        s = torch.dot(z, self.v)
        y = (self.w_s * s + self.c_s).view(-1)
        if return_attention:
            return y, w, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, bone_aware=True, warm_start=True)
