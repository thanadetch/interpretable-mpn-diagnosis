"""a457 — every patch predicts a DISTRIBUTION over the four grades; attention pools their votes.

THE IDEA
    ABMIL maps each patch to an unbounded scalar through a linear head and pools those scalars
    with attention. Algebraically that already IS "predict per patch, then average by attention"
    -- verified on this cohort to 1e-6 -- so a per-patch scalar readout is not a new model.

    What IS different: let the patch head emit a SOFTMAX over {G0, G1, G2, G3} instead of a free
    scalar, and read each patch's grade as the expectation of that distribution.

        p_i    = softmax(W_g h_i)                  [N, 4]   this patch's grade distribution
        y_i    = sum_g g * p_i(g)                  [N]      bounded to [0, 3] by construction
        y      = sum_i a_i y_i                     scalar   attention-weighted, ordinal
        mass(g)= sum_i a_i p_i(g)                  [4]      "how much of the field reads as g"

    Two things follow that ABMIL does not give:

    1. NO SINGLE PATCH CAN SHOUT. A linear head lets one patch emit 7.0 and drag the bag; the
       softmax expectation caps every patch at [0, 3]. That matches this project's grading
       principle -- grade is the diffuse density of the meshwork across the field, not the
       verdict of a few standout patches -- which is exactly the prior that ruled out top-k and
       norm-ranked pooling.
    2. mass(g) IS A READABLE EXPLANATION. "58% of the attention mass reads as G2" is a sentence
       for a pathologist; an attention heatmap is not.

    Ordinality is kept: the loss stays SmoothL1 on the scalar, so this does NOT reintroduce the
    multi-class cross-entropy that gave every categorical baseline in this project G0 recall = 0%.

HARD VOTE (inference-only, reported by the analysis script, never trained against)
    g_i = argmax_g p_i(g);  vote(g) = sum_{i: g_i = g} a_i;  grade = argmax_g vote(g)
    Measured post-hoc on ABMIL/ASGAP checkpoints this readout lost 4 of 6 cells and collapsed G1
    (down to -18.4 points) because argmax discards the ordinal scale. It is computed here only so
    the trained-from-scratch version of the same rule can be checked against its own model.

PARAMETERS
    Identical to ABMIL except the head: Linear(128 -> 4) instead of Linear(128 -> 1), i.e. +387
    parameters on 197,250. No capacity confound worth arguing about.

PREDICTION, STATED BEFORE THE RUN
    Ties or slightly loses to ABMIL, because bounding every patch to [0, 3] removes headroom the
    linear head uses for the 80 G3 ROIs in the test split. If it WINS, the capping helped, and
    that is a positive result aligned with the pathology prior rather than a tuning artefact.

Gate: val QWK AND test QWK above ABMIL on the same backbone (seed 2, no augmentation).
Permutation- and size-invariant, deterministic at eval, MPS-safe, no new deps.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim: int = 1280, num_classes: int = 1, hidden_dim: int = 128,
                 dropout: float = 0.5, n_grades: int = 4):
        super().__init__()
        if num_classes != 1:
            raise ValueError("a457 is a scalar-regression head; use --formulation regression")
        self.n_grades = n_grades
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.grade_head = nn.Linear(hidden_dim, n_grades)
        self.register_buffer("levels", torch.arange(n_grades, dtype=torch.float32))

    def patch_terms(self, features: torch.Tensor):
        """Returns (attention [N], per-patch grade distribution [N, G], per-patch grade [N])."""
        if features.dim() == 3:
            features = features.squeeze(0)
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = F.softmax(e, dim=0)
        p = F.softmax(self.grade_head(h), dim=-1)
        y_i = p @ self.levels.to(p.dtype)
        return a, p, y_i

    def forward(self, features, return_attention: bool = False, metrics=None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        a, _, y_i = self.patch_terms(features)
        y = torch.dot(a, y_i).view(1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
