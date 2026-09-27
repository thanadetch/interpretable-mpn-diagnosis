"""a313 — Softmax pool (UNCHANGED) + bag-diffuseness density side-channel to the head.

KEY INSIGHT (from a311/a312): every prior failure CRASHED uni2 because it MODIFIED THE POOLING WEIGHTS
(entmax/coverage/blend all deviate from softmax's shape, which uni2 needs). So a313 does NOT touch the
pooling weights at all — the bag descriptor is the exact baseline softmax gated-attention mean
(z_mean = sum_i softmax(e)_i h_i), preserving uni2's optimum. Instead it hands the head an extra
grading-principle-aligned scalar: the *diffuseness* of the attention distribution.

  s = H(a) / log(N)  in [0,1],   a = softmax(e),   H(a) = -sum_i a_i log a_i

s=1 means the attention is spread bag-wide (diffuse reticulin meshwork read), s->0 means it spikes on a
few patches. This is exactly the holistic-density vs few-standout-patches distinction the pathologist
uses, computed as a size-invariant scalar. The head sees [z_mean ; s] -> grade.

WHY THIS IS uni2-SAFE BY CONSTRUCTION (the honest argument): the main H-dim channel is the untouched
softmax pool, so the worst case on uni2 is that the head learns ~0 weight on s and recovers the baseline
(it CANNOT crash uni2 the way weight-modifying pools did). The hope (not a claim) is that the diffuse
backbones titan/virchow2 gain a little from the explicit diffuseness signal. DISCLOSURE: still a single
drop-in; the side scalar adds H+1 -> 1 head params and could mildly overfit the 214-ROI val cohort, so
GATE2 is not expected — but unlike every prior candidate this one is uni2-safe by construction, which is
the property the wall demands. No ||h|| weighting. Self-contained, concept-free, permutation/size-
invariant, deterministic eval, MPS-safe, no new deps.
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
        self.classifier = nn.Linear(hidden_dim + 1, num_classes)   # +1 for the diffuseness scalar

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)   # [N]
        a = F.softmax(e, dim=0)                                                        # UNCHANGED softmax pool
        z_mean = torch.mv(h.t(), a)                                                    # [H]
        N = a.shape[0]
        # size-invariant diffuseness scalar in [0,1]: normalized attention entropy
        ent = -(a * torch.log(a.clamp(min=1e-12))).sum()
        denom = torch.log(torch.tensor(float(N), device=a.device)).clamp(min=1e-6)
        s = (ent / denom).clamp(0.0, 1.0).view(1)                                      # [1]
        z = torch.cat([z_mean, s], dim=0)                                              # [H+1]
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
