"""a101 — Minimal-DOF frozen-axis diffuse mean-projection (robustness floor probe).

Every richer mechanism val-overfits the hard seed=2 fold (a91/a93/a99 lift val,
tank test). This is the opposite extreme: the MOST-regularised possible
grading model — diffuse mean-pool, project onto the FROZEN train-only fibrosis
axis (NOT trainable), and a 2-parameter affine readout. Only w (scale) and b
(offset) learn; the direction cannot drift to fit a fold. If even this generalises
no better than baseline paired, the cohort floor is confirmed; if its minimal
variance gives a robust paired edge, that is itself informative.

    z = mean_i f_i ; s = <z, v_frozen> ; y = w*s + b   (2 trainable params)

Ablation a102: trainable axis (v becomes a Parameter, warm-started) — isolates
'does freezing the direction (zero direction-DOF) reduce the val<->test gap vs
letting it train?'. forward uses ONLY 'features'. RAW logit (trainer rounds/clips).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AXIS_PATH = _REPO_ROOT / "data" / "prototypes_virchow2_reti_train_seed2.pt"


def _axis(input_dim: int) -> torch.Tensor:
    try:
        v = torch.load(_AXIS_PATH, map_location="cpu", weights_only=False)["axis"].float().view(-1)
        if v.numel() == input_dim:
            return v / v.norm().clamp(min=1e-8)
    except Exception:
        pass
    g = torch.randn(input_dim)
    return g / g.norm().clamp(min=1e-8)


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        train_axis: bool = False,
    ) -> None:
        super().__init__()
        axis = _axis(input_dim)
        if train_axis:
            self.v = nn.Parameter(axis)
            self.register_buffer("_vbuf", torch.empty(0))
        else:
            self.register_buffer("v", axis)  # frozen direction
        self.w = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(1.5))

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        z = features.mean(dim=0)            # [D] diffuse pool
        s = (z * self.v).sum()              # scalar projection onto fibrosis axis
        y = (self.w * s + self.b).view(1, 1)
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, train_axis=False)
