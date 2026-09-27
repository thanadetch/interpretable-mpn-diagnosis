"""a187 — DSMIL dual-stream pooling. NEW family: critical-instance + masked attention to it.

Dual-stream MIL (Li et al. 2021), a strong MIL architecture never tried here:
  stream 1 (instance): score every patch with a per-instance head; pick the CRITICAL instance (argmax).
  stream 2 (bag): every patch attends to the critical instance (similarity in a learned query space);
                  pool values by that attention.
  prediction fuses the bag-stream output and the critical-instance score.
This explicitly anchors the bag read on its single most-salient patch and measures everything relative
to it -- a fundamentally different inductive bias from independent gated attention. Concept-free,
self-contained, permutation-invariant (argmax tie-break by index = deterministic), eval-deterministic.
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
        self.inst_head = nn.Linear(hidden_dim, num_classes)     # per-instance score (stream 1)
        self.q = nn.Linear(hidden_dim, hidden_dim)              # query
        self.v = nn.Linear(hidden_dim, hidden_dim)              # value
        self.bag_head = nn.Linear(hidden_dim, num_classes)      # bag-stream head (stream 2)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                           # [N,D]
        inst = self.inst_head(h).squeeze(-1)                    # [N] per-instance scalar score
        m = torch.argmax(inst)                                  # critical instance index
        Q = self.q(h)                                           # [N,D]
        V = self.v(h)                                           # [N,D]
        qm = Q[m]                                               # [D] critical query
        A = F.softmax((Q @ qm) / (h.shape[1] ** 0.5), dim=0)    # [N] attention to critical instance
        b = torch.mv(V.t(), A)                                  # [D] bag embedding
        y_bag = self.bag_head(b).view(-1)
        y_max = inst[m].view(-1)                                # critical-instance score
        y = 0.5 * (y_bag + y_max)                               # DSMIL fusion
        if return_attention:
            return y, A, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
