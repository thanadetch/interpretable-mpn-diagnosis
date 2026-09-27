"""a143 — Per-Bag-Normalized Bone-Aware Diffuse Density.

Same grading-aligned mechanism as a135 (bone-aware diffuse fibrosis-density pool) but the
bone/fib concept scores are z-normalized PER BAG (using each ROI's own patch mean/std)
instead of with the GLOBAL dataset constants (_BONE_MEAN/_STD, _FIB_MEAN/_STD).

Motivation (user): ROIs differ in magnification and stain intensity, so the absolute
concept-score scale is NOT comparable across ROIs — normalizing bone/fib within each ROI
adapts the bone-avoidance to that ROI's own context. BONUS: per-bag stats are fully
leakage-safe (no train/val/test global statistics enter the model at all; the only
warm-start is the per-fold prototype axis, already leakage-safe).

  bone_z = (bone − μ_b) / σ_b ,  fib_z = (fib − μ_f) / σ_f
     norm_mode="per_bag": μ,σ = THIS bag's patch mean/std (σ clamped to a floor)
     norm_mode="global" : μ,σ = global dataset constants (= the a135 mechanism)  [ablation a144]
  w = softmax(−softplus(λ)·relu(bone_z − fib_z))        # down-weight bone-specific patches
  z = Σ wᵢ·featᵢ                                          # bone-avoiding diffuse pool
  y = w_s·⟨z, v⟩ + c_s                                    # project on warm fibrosis axis → grade

a143 (per_bag) vs a144 (global) isolates EXACTLY the effect of per-ROI vs global concept
normalization on the bone-aware diffuse-density readout. Low-DOF (axis + λ + w_s + c_s),
permutation-/bag-size-invariant, deterministic. Reads data_bonefib bag = [feat|bone|fib].
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
    def __init__(self, input_dim=1280, num_classes=1, norm_mode="per_bag", warm_start=True):
        super().__init__()
        assert num_classes == 1
        self.feat_dim = input_dim
        self.norm_mode = norm_mode          # "per_bag" | "global"
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
        self.lam_raw = nn.Parameter(torch.tensor(0.0))
        self.w_s = nn.Parameter(torch.tensor(float(w_s_init)))
        self.c_s = nn.Parameter(torch.tensor(float(c_s_init)))

    def _zscore(self, x):
        if self.norm_mode == "per_bag":
            mu = x.mean()
            sd = x.std().clamp(min=1e-3)        # floor: near-uniform bags don't blow up
            return (x - mu) / sd
        # global
        return x  # caller applies global constants

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        D = self.feat_dim
        feats = features[:, :D]
        if features.shape[1] >= D + 2:
            bone_raw = features[:, D]
            fib_raw = features[:, D + 1]
            if self.norm_mode == "per_bag":
                bone = self._zscore(bone_raw)
                fib = self._zscore(fib_raw)
            elif self.norm_mode == "hybrid":
                # per-bag CENTER (removes per-ROI stain/mag offset) + GLOBAL scale
                # (preserves absolute bone magnitude → keeps faithfulness on bone-heavy ROIs)
                bone = (bone_raw - bone_raw.mean()) / _BONE_STD
                fib = (fib_raw - fib_raw.mean()) / _FIB_STD
            else:  # global
                bone = (bone_raw - _BONE_MEAN) / _BONE_STD
                fib = (fib_raw - _FIB_MEAN) / _FIB_STD
            w = F.softmax(-F.softplus(self.lam_raw) * F.relu(bone - fib), dim=0)
        else:
            w = torch.full((feats.shape[0],), 1.0 / feats.shape[0], device=feats.device)
        z = torch.mv(feats.t(), w)
        y = (self.w_s * torch.dot(z, self.v) + self.c_s).view(-1)
        if return_attention:
            return y, w, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, norm_mode="per_bag", warm_start=True)
