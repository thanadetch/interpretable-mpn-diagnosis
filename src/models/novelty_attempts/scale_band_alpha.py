"""scale_band_alpha — every magnification band owns a FREE learnable entmax order.

WHAT THIS IS
------------
The ROIs are grouped by their measured physical scale, and each group gets its own alpha, learned
end to end with nothing tying the groups together:

    band 0   um/px  < 0.45     ~25-40x, one patch covers ~60-100 um     alpha_0 = 1 + sigmoid(G * t_0)
    band 1   0.45 <= < 0.70    ~20x,    ~100-160 um                     alpha_1 = 1 + sigmoid(G * t_1)
    band 2   um/px >= 0.70     ~4-12x,  ~160-600 um                     alpha_2 = 1 + sigmoid(G * t_2)
    band 3   scale bar unreadable (46% of ROIs)                         alpha_3 = 1 + sigmoid(G * t_3)

Four parameters, one per band, all initialised to zero so every band starts at exactly alpha =
1.5 — identical to ASGAP — and any separation between the bands afterwards is something the data
produced. `G = 3` amplifies the pre-sigmoid term; without it a bare scalar receives too small a
gradient to move on 30 training patients, the failure a215's "learnable" alpha ran into.

WHY THIS RATHER THAN THE EARLIER TWO
------------------------------------
`scale_gated_alpha` (a409) blends five fixed orders, so its effective alpha is an average and is
dragged toward the centre by construction: given the full 1.0-2.0 range it only used 1.478-1.518.
`scale_direct_alpha` (a411) frees the range but forces a single monotone function of log scale, so
it cannot express a non-monotone pattern. Here the bands are independent: each may land anywhere
in (1, 2), in any order.

`alpha_report()` returns the four learned values, which is the direct answer to "what alpha does
each magnification want" — readable without reference to accuracy.

CONTROL
-------
`shuffle=True` keeps the four free parameters and the band sizes but assigns bands to ROIs at
random. The model can still learn four different orders; it just cannot align them with the real
magnification. If the shuffled control matches, the grouping carries no information and the
freedom to use several orders is doing the work — which is what the earlier a407/a408 pair
already suggested.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .a215_adaptive_sparse_gated_attention_pooling import entmax_bisect
from .scale_gated_alpha import LOG_REF, LOG_SCALE, _ScaleFeatureBank

GAIN = 3.0
N_BANDS = 4
# band edges in standardised-log-scale units, matching um/px 0.45 and 0.70
_EDGES = [(np.log10(v) - LOG_REF) / LOG_SCALE for v in (0.45, 0.70)]


class Model(nn.Module):
    def __init__(self, input_dim: int = 1280, num_classes: int = 1, hidden_dim: int = 128,
                 dropout: float = 0.5, shuffle: bool = False, seed: int = 0):
        super().__init__()
        self.shuffle, self.seed = bool(shuffle), int(seed)
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.theta = nn.Parameter(torch.zeros(N_BANDS))     # all bands start at alpha = 1.5
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.bank = _ScaleFeatureBank()
        self._rng = np.random.default_rng(self.seed)

    def _band(self, s: float, u: float) -> int:
        if self.shuffle:
            return int(self._rng.integers(0, N_BANDS))
        if u > 0.5:
            return 3
        return int(np.searchsorted(_EDGES, s, side="right"))

    def alpha_of(self, band: int) -> torch.Tensor:
        return 1.0 + torch.sigmoid(GAIN * self.theta[band])

    @torch.no_grad()
    def alpha_report(self) -> dict:
        names = ("<0.45 (~25-40x)", "0.45-0.70 (~20x)", ">=0.70 (~4-12x)", "unknown")
        return {n: round(float(self.alpha_of(b)), 4) for b, n in enumerate(names)}

    def forward(self, features, return_attention: bool = False, metrics=None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        if features.dim() == 3:
            features = features.squeeze(0)
        s, u = self.bank.feat(features)
        alpha = self.alpha_of(self._band(s, u))
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = entmax_bisect(e, alpha)
        y = self.classifier(torch.mv(h.t(), a)).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, shuffle=False)
