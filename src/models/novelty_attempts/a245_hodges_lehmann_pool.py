"""a245 - Coordinate-wise MEDIAN pooling (robust feature location).

    H = Dropout(ReLU(Linear(input_dim, 128)))(features)   # [N, 128]
    z = median(H, dim=0)                                   # [128] coord-wise median
    y = Linear(128, 1)(z).view(-1)                         # (1,)

Summarise the bag by the coordinate-wise MEDIAN of the encoded patch vectors -- a robust location
(breakdown 0.5) that a minority of bone/fat/artefact/off-tissue patches cannot drag, unlike mean pooling
(breakdown 0). Lower read-variance directly attacks the val-selection variance behind the val<->test
-0.95 anti-correlation, WITHOUT collapsing to the mean; it has NO learnable robustness/shape knob (nothing
to "go inert") and applies the IDENTICAL read to all backbones (sidesteps the disjoint-helper trap).

NOTE: originally specified as the Hodges-Lehmann estimator (median of all N^2 Walsh pairwise averages),
but that O(N^2 * 128) median is computationally impractical on the Apple-MPS trainer (~20 min/epoch,
torch.median over ~12.7k rows per bag). The plain coordinate-wise median retains the same robustness,
no-inert-knob and backbone-invariance properties at O(N log N). Self-contained, permutation/size-invariant,
deterministic in eval(), n=1 safe (median of 1 point = the point).
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        H = self.bottleneck(features)                  # [N, hidden]
        if H.size(0) == 1:
            z = H[0]
        else:
            z = torch.median(H, dim=0).values          # [hidden] coordinate-wise median
        y = self.head(z).view(-1)
        if return_attention:
            return y, None, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
