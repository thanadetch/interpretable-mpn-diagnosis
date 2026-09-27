"""a105 — Sorted-projection-CURVE pooling at baseline capacity.

a70 (3 soft-quantiles) underfit (val 0.717) because it had NO bottleneck (~1.3K
params). This gives the full-distribution-shape idea a FAIR encoder: the
baseline bottleneck (164K) feeds a learned per-patch projection, whose sorted
within-bag curve (resampled to a fixed M points, so bag-size-invariant) is the
bag descriptor. Reading the WHOLE sorted density curve (not 3 sampled quantiles,
not mean/var moments) is the genuinely-distinct ingredient.

    h_i = Dropout(ReLU(Linear(1280->128) f_i))         # baseline encoder
    s_i = <h_i, w>                                      # per-patch fibrosis score
    curve = interp(sort(s), M)  in R^M                  # sorted density curve (M=16)
    y = Linear(M -> 1)(curve)                           # RAW logit

Permutation-invariant (sort), bag-size-invariant (resample to fixed M),
deterministic. Ablation a106: M=1 -> curve = mean(s) -> reads only central
density (collapses the curve), isolating 'does the full sorted SHAPE beat the
central density?'. forward uses ONLY 'features'.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_AXIS_PATH = Path(__file__).resolve().parents[3] / "data" / "prototypes_virchow2_reti_train_seed2.pt"


def _axis(input_dim: int) -> Optional[torch.Tensor]:
    try:
        v = torch.load(_AXIS_PATH, map_location="cpu", weights_only=False)["axis"].float().view(-1)
        if v.numel() == input_dim:
            return v / v.norm().clamp(min=1e-8)
    except Exception:
        pass
    return None


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        n_points: int = 16,            # M sorted-curve resample points (a106 ablation = 1)
    ) -> None:
        super().__init__()
        self.n_points = int(n_points)
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # per-patch projection direction (warm-started from bottleneck response to axis)
        self.w = nn.Parameter(torch.randn(hidden_dim) / (hidden_dim ** 0.5))
        ax = _axis(input_dim)
        if ax is not None:
            with torch.no_grad():
                resp = self.bottleneck[0].weight.detach() @ ax
                self.w.copy_(resp / resp.norm().clamp(min=1e-8))
        self.head = nn.Linear(self.n_points, num_classes)
        with torch.no_grad():
            self.head.bias.fill_(1.5)

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)               # [N, hidden]
        s = h @ self.w                              # [N] per-patch score
        if self.n_points == 1:
            curve = s.mean().view(1, 1)             # central density (ablation)
        else:
            s_sorted = torch.sort(s).values         # [N] ascending
            curve = F.interpolate(
                s_sorted.view(1, 1, -1), size=self.n_points, mode="linear", align_corners=True
            ).view(1, self.n_points)                # [1, M] bag-size-invariant sorted curve
        logits = self.head(curve)                   # [1, num_classes] RAW
        return logits, None, None


KWARGS = dict(input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, n_points=16)
