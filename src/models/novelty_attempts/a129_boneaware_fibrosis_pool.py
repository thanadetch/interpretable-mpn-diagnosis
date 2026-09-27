"""a129 — bone-aware DIFFUSE fibrosis-density pooling: the most grading-faithful
aggregator = the pathologist's reading algorithm in one module.

Combines the two pillars of the grading prior that no single prior module did:
  (A) DIFFUSE density along the learned fibrosis direction (a95/a96), and
  (B) AVOID bone trabeculae (a122).

Mechanism on a 1282-d bag = [virchow2 1280 | bone | fibrosis] (data_bonefib):
  1) bone-aware DIFFUSE weights: w_i = softmax_i(-softplus(lam_bone) * relu(bone_z_i - fib_z_i))
     -> uniform (pure diffuse mean) when lam_bone=0; down-weights BONE-SPECIFIC
        patches (high bone, low fibrosis) as lam_bone grows. Stays diffuse (soft,
        whole-bag), never spikes on a few patches.
  2) z = sum_i w_i * feats_i                        # bone-aware diffuse bag descriptor
  3) s = <z, v>                                     # fibrosis-axis projection
                                                    #   v warm-started at train axis (seed=2)
  4) y = w_s * s + c_s                              # interpretable affine readout -> grade

Learnable: v (1280) + lam_bone (1) + w_s (1) + c_s (1) = 1283. Interpretable, diffuse,
bone-avoiding, fibrosis-aligned, no ||h|| weighting. forward reads ONLY 'features'.
RAW logit. Permutation- & bag-size-invariant. Deterministic at eval.

Ablation a130 = lam_bone frozen at 0 (= a96 diffuse fibrosis-projection, no bone term):
isolates the bone-aware pooling as the active ingredient.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

_AXIS_PATH = Path(__file__).resolve().parents[3] / "data" / "prototypes_virchow2_reti_train_seed2.pt"
_BONE_MEAN, _BONE_STD = -0.0486, 0.0193
_FIB_MEAN, _FIB_STD = 0.0045, 0.0281


def _axis(input_dim: int) -> Optional[torch.Tensor]:
    try:
        v = torch.load(_AXIS_PATH, map_location="cpu", weights_only=False)["axis"].float().view(-1)
        if v.numel() == input_dim:
            return v / v.norm().clamp(min=1e-8)
    except Exception:
        pass
    return None


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, bone_aware=True, warm_start=True):
        super().__init__()
        assert num_classes == 1
        self.feat_dim = input_dim
        self.bone_aware = bone_aware
        axis = _axis(input_dim) if warm_start else None
        v_init = axis.clone() if axis is not None else F.normalize(torch.randn(input_dim), dim=0)
        self.v = nn.Parameter(v_init)
        # bone-avoidance strength (>=0 via softplus); init ~0 (lam_raw large-negative => uniform)
        self.lam_raw = nn.Parameter(torch.tensor(0.0))   # softplus(0)=0.69 -> mild bone avoidance at init
        # interpretable affine readout (warm to roughly map fibrosis score -> [0,3])
        self.w_s = nn.Parameter(torch.tensor(1.0))
        self.c_s = nn.Parameter(torch.tensor(1.5))

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        D = self.feat_dim
        feats = features[:, :D]
        if self.bone_aware and features.shape[1] >= D + 2:
            bone = (features[:, D] - _BONE_MEAN) / _BONE_STD
            fib = (features[:, D + 1] - _FIB_MEAN) / _FIB_STD
            bone_specific = F.relu(bone - fib)
            w = F.softmax(-F.softplus(self.lam_raw) * bone_specific, dim=0)   # bone-aware diffuse weights
        else:
            w = torch.full((feats.shape[0],), 1.0 / feats.shape[0], device=feats.device)
        z = torch.mv(feats.t(), w)            # [D] bone-aware diffuse descriptor
        s = torch.dot(z, self.v)              # fibrosis-axis projection
        y = (self.w_s * s + self.c_s).view(-1)
        if return_attention:
            return y, w, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, bone_aware=True, warm_start=True)
