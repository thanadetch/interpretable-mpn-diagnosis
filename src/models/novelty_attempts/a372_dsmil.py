"""a372 — DSMIL (Li, Li & Eliceiri, CVPR 2021), "Dual-stream MIL Network for WSI
Classification with Self-supervised Contrastive Learning".

Written from the paper's formulation (github.com/binli123/dsmil-wsi, `dsmil.py`), adapted to the
ordinal-regression head this project uses. One of the standard published MIL baselines that this
codebase has never actually run on the grading task.

MECHANISM (two streams over the same instance embeddings)

  stream 1 — INSTANCE. A shared linear instance classifier scores every patch; the highest-scoring
             patch is the CRITICAL INSTANCE x_m. Its score c_m is the first stream's output.
  stream 2 — BAG. Every patch emits a query q_i and a value v_i; the critical instance's query
             q_m is the anchor. The attention on patch i is the softmax over the scaled inner
             product <q_i, q_m>, i.e. every patch is measured by its similarity TO THE CRITICAL
             INSTANCE rather than to a learned global query (this is DSMIL's distinguishing
             move versus ABMIL). b = sum_i a_i v_i, and c_b = W_b b.

  output    y = (c_m + c_b) / 2      (the paper's average of the two streams)

DEVIATIONS (honest)
  - REGRESSION HEAD: num_classes=1 and the ordinal-regression target of this project, instead of
    the paper's binary/multi-class cross-entropy. The two-stream structure is unchanged.
  - The paper's non-local `distance` variant and its SimCLR feature pre-training are out of scope;
    features here come from frozen pathology foundation models, which is the closer analogue of
    the paper's self-supervised embedder.
  - `nonlinear` instance classifier option kept at the paper's default (linear).

Permutation-invariant, bag-size-invariant, deterministic at eval.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dim: int = 1280, num_classes: int = 1,
                 hidden_dim: int = 128, dropout: float = 0.5, passing_v: bool = False):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        # stream 1: instance classifier
        self.inst_classifier = nn.Linear(hidden_dim, num_classes)
        # stream 2: query / value projections
        self.q = nn.Sequential(nn.Linear(hidden_dim, 128), nn.Tanh())
        self.v = (nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
                  if passing_v else nn.Identity())
        self.bag_classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention: bool = False, metrics=None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        if features.dim() == 3:
            features = features.squeeze(0)
        h = self.bottleneck(features)                      # [N, H]

        c = self.inst_classifier(h)                        # [N, C] instance scores
        m_idx = torch.argmax(c[:, 0])                      # critical instance (ordinal score)
        c_max = c[m_idx]                                   # [C]

        Q = self.q(h)                                      # [N, 128]
        V = self.v(h)                                      # [N, H]
        q_m = Q[m_idx]                                     # [128] anchor query
        e = (Q @ q_m) / (Q.shape[1] ** 0.5)                # similarity TO the critical instance
        a = torch.softmax(e, dim=0)                        # [N]
        b = torch.mv(V.t(), a)                             # [H]
        c_bag = self.bag_classifier(b)                     # [C]

        y = (0.5 * (c_max + c_bag)).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
