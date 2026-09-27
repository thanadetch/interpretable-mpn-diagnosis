"""a170 — Per-bag ADAPTIVE temperature gated attention (input-dependent T).

LIGHT, self-contained. a158 added ONE GLOBAL scalar temperature T and it learned T->0.98 (inert):
a single fold-wide T has no reason to move because the test-optimal point is the standard softmax.
a170 asks a different question: should the temperature DEPEND ON THE BAG? Pathology rationale —
a sparse/peaked ROI (attention naturally concentrated) might want a different diffuseness than an
ROI whose fibre signal is already spread out. So T is a function of the bag's own attention
dispersion instead of a constant.

    e_i        = W(V(h_i) * U(h_i))                       # gated attention energies (per patch)
    a0         = softmax(e)                               # base (T=1) attention
    H          = -Σ a0 log a0 / log N   in [0,1]          # normalised attention entropy (bag descriptor)
    T_bag      = softplus(g0 + g1 * (H - 0.5))            # per-bag temperature
    a_i        = softmax(e_i / T_bag) ; z = Σ a_i h_i ; y = classifier(z)

g0 = inverse-softplus(1) and g1 = 0 at init -> T_bag = 1 for EVERY bag = exact ABMIL/a158
baseline. The optimiser earns a non-trivial, bag-conditioned temperature only if it generalises.
Bag-descriptor H is permutation- and size-invariant (normalised by log N); deterministic at
inference. 2 extra DOF over the baseline. No axis, no external file.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

_INV_SOFTPLUS_1 = math.log(math.expm1(1.0))  # g0 init so softplus(g0)=1


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        # T_bag = softplus(g0 + g1 * (H - 0.5)); init g0->T=1, g1=0 -> exact baseline
        self.g0 = nn.Parameter(torch.tensor(_INV_SOFTPLUS_1))
        self.g1 = nn.Parameter(torch.tensor(0.0))

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        n = e.shape[0]
        if n >= 2:
            a0 = F.softmax(e, dim=0)
            ent = -(a0 * (a0.clamp(min=1e-12)).log()).sum()
            H = ent / math.log(n)            # normalised entropy in [0,1]
        else:
            H = e.new_tensor(0.5)            # single-patch bag -> neutral -> T=1
        T = F.softplus(self.g0 + self.g1 * (H - 0.5)).clamp(min=1e-2)
        attn = F.softmax(e / T, dim=0)
        z = torch.mm(attn.unsqueeze(0), h).squeeze(0)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, attn, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
