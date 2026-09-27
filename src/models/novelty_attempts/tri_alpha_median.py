"""tri_alpha_median — three attention branches at DIFFERENT entmax orders, combined by the median.

THE OBSERVATION THIS IS BUILT ON
--------------------------------
Under 5-fold CV (out-of-fold over all 1,330 ROIs) the best sparsity level is not the same for
every backbone:

    Virchow2   softmax  (alpha = 1)    950 / 1330   >  entmax-1.5   914
    UNI2-h     entmax-1.5              992 / 1330   >  softmax      908
    TITAN      entmax-1.5              912 / 1330   >  softmax      887

One fixed alpha therefore cannot be right everywhere, and picking it per backbone on the test
split would be selection on the test set. Three branches at alpha = 1.0 (softmax), 1.5 and 2.0
(sparsemax) span the whole sparsity spectrum; taking the MEDIAN of their three predictions keeps
whichever branch sits in the middle and discards the one that disagrees most, so a backbone whose
optimum lies at either extreme is not dragged toward the centre the way a mean would drag it.

WHAT IS NEW HERE, VERIFIED AGAINST THE 424 MODULES ON DISK
----------------------------------------------------------
    a302  three branches combined by the median, but all three use SOFTMAX (identical alpha;
          they differ only by random init) -- no sparsity diversity at all
    a303  two branches, softmax + entmax-1.5, combined by the MEAN
    a289  four alphas [1.1, 1.25, 1.4, 1.55] -- all on the dense side, never reaches sparsemax,
          and combined by a consensus (mean) head
    a272  two alphas concatenated before a single head, not voted

So the combination "alpha spanning the full 1 -> 2 range" x "median combine" is untried.

THE CONTROL THAT DECIDES IT
---------------------------
`alphas=(1.5, 1.5, 1.5)` gives three branches with the SAME architecture, the SAME parameter
count and the SAME median combine, differing only in that the sparsity diversity is removed.
Ensembling by itself almost always buys a little; that control is what separates "the alpha
diversity did something" from "any three-branch ensemble would have done this". Report both or
report neither.

The median of three scalars is differentiable -- it selects the middle branch and the gradient
flows through it, so the branches stay trainable and only the outlier branch is ignored per bag.

Permutation- and bag-size-invariant, deterministic at eval, no instance-instance mixing.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .a215_adaptive_sparse_gated_attention_pooling import entmax_bisect


class _Branch(nn.Module):
    """One gated-attention MIL head whose pooling sparsity is fixed at `alpha`."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, alpha: float):
        super().__init__()
        self.alpha = float(alpha)
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, 1)

    def weights(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        # entmax_bisect is singular at alpha -> 1, where the map is exactly softmax
        a = F.softmax(e, dim=0) if self.alpha <= 1.0 + 1e-6 else entmax_bisect(e, self.alpha)
        return h, a

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, a = self.weights(features)
        return self.classifier(torch.mv(h.t(), a)).view(()), a


class Model(nn.Module):
    def __init__(self, input_dim: int = 1280, num_classes: int = 1, hidden_dim: int = 128,
                 dropout: float = 0.5, alphas: Sequence[float] = (1.0, 1.5, 2.0)):
        super().__init__()
        if num_classes != 1:
            raise ValueError("tri_alpha_median is a regression head (num_classes=1).")
        self.alphas = tuple(float(a) for a in alphas)
        self.branches = nn.ModuleList(
            [_Branch(input_dim, hidden_dim, dropout, a) for a in self.alphas])

    def branch_outputs(self, features: torch.Tensor) -> list[float]:
        """Per-branch predictions, for checking whether the branches actually disagree."""
        if features.dim() == 3:
            features = features.squeeze(0)
        with torch.no_grad():
            return [float(b(features)[0]) for b in self.branches]

    def forward(self, features, return_attention: bool = False, metrics=None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        if features.dim() == 3:
            features = features.squeeze(0)
        ys, maps = [], []
        for b in self.branches:
            y, a = b(features)
            ys.append(y)
            maps.append(a)
        y = torch.median(torch.stack(ys)).view(1)
        if return_attention:
            # the attention of the branch the median actually selected
            k = int(torch.argsort(torch.stack(ys))[len(ys) // 2])
            return y, maps[k], None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, alphas=(1.0, 1.5, 2.0))
