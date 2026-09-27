"""a296 — Smooth BOUNDED output head (y = 3*sigmoid(raw)) on the baseline pool, probing the OUTPUT axis
(seed-2 fish #17; best-of-N, seed-2 only, NOT robust).

All prior perturbations moved the POOLING / HEAD-CAPACITY / FEATURE axes and confirmed the baseline is a
local optimum there. The one remaining surface is the OUTPUT representation. The trainer hard-clamps
predictions to [0,3] AFTER the module; a296 instead emits a SMOOTH bounded prediction y = 3*sigmoid(s)
inside the module, so gradients stay informative near the G0 (0) and G3 (3) boundaries instead of being
killed by the hard clamp. uni2's residual errors concentrate at the extreme grades (only 2 patients each
for G0/G3), so smoother boundary calibration is the most targeted output-side idea. Everything else =
baseline EXACTLY (softmax gated pool + single Linear scorer). DISTINCT: changes only the output squashing.
Honest expectation: boundary smoothing rarely beats the hard clamp on QWK and may slightly compress the
dynamic range; recorded per the never-stop directive. Self-contained, perm/size-invariant, deterministic
eval, MPS-safe.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = F.softmax(e, dim=0)                                # baseline softmax pool (uni2's optimum)
        z = torch.mv(h.t(), a)
        s = self.classifier(z).view(-1)
        y = 3.0 * torch.sigmoid(s)                            # smooth bounded output in [0,3]
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
