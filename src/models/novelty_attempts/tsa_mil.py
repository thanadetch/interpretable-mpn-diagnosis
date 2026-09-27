"""tsa_mil — Threshold-Specific Attention MIL (TSA; base module for a400-a404).

THE ASSUMPTION BEING CHALLENGED
-------------------------------
Every attention MIL — ABMIL, ASGAP, CLAM, DSMIL, TransMIL — computes ONE attention map per
bag and reads every class decision out of the SAME pooled vector. That is correct when the
classes are unordered categories competing for the same evidence. It is wrong for a graded
quantity whose successive thresholds are defined by DIFFERENT tissue:

    MF-0 -> MF-1   "occasional linear reticulin"        evidence = does ANY fibre exist at all
    MF-1 -> MF-2   "diffuse, dense, extensive intersections"  evidence = crossings and density
    MF-2 -> MF-3   "coarse bundles, often with collagen"      evidence = thick bundle morphology

A single attention map must therefore serve three different queries at once. Under a bag of ~44
patches and 30 training cases it resolves that conflict by favouring the most abundant, highest-
contrast evidence (the MF-3 end), which is exactly the failure this project measures everywhere:
G1 recall is the weakest cell for every backbone and every aggregator tried (55-78).

MECHANISM
---------
Decompose the ordinal target into its cumulative thresholds and give EACH threshold its own
attention over the same shared instance embeddings:

    h_i   = Bottleneck(x_i)                                        # shared encoder [N, H]
    for k = 1..K:                                                  # K = 3 thresholds
        e^k_i = W_k( tanh(V_k h_i) (.) sigmoid(U_k h_i) )          # threshold-specific gate
        a^k   = entmax_1.5(e^k)                                    # sparse; G1 is sparse evidence
        z^k   = sum_i a^k_i h_i                                    # threshold-specific descriptor
        p_k   = sigmoid( w_k . z^k + b_k )                         # P(grade >= k | its own view)
    q_k   = prod_{j<=k} p_j                                        # chain: q is non-increasing in k
    y     = q_1 + q_2 + q_3                                        # expected grade in (0, 3)

The chain parametrisation makes P(>=1) >= P(>=2) >= P(>=3) hold STRUCTURALLY, so the three
independent attention maps can never produce an incoherent ordinal answer. `y` is a proper
expected grade, so the existing regression loss and QWK metric apply unchanged.

WHY THIS IS A MECHANISM AND NOT A READOUT TWEAK
-----------------------------------------------
Every ordinal module already in this repo (a47 cumulative-link, a54 coverage cascade, a89/a90
calibrated ordinal, a150 ordinal-CDF, a246 survival curve, a251 consensus CDF) pools ONCE and
then applies an ordinal head to that single vector. TSA changes WHERE the ordinal structure
enters: it is in the ATTENTION, upstream of pooling, so different thresholds may look at
different patches. That is the entire claim, and a401 is built to falsify it.

CONTROLS (the claim is only meaningful with these)
--------------------------------------------------
a401 `share_attn=True`   ONE attention shared by all three thresholds, everything else identical
                         - same chain, same parameter budget in the heads. Isolates "does the
                         attention have to be threshold-specific, or is the ordinal chain alone
                         doing the work?" THIS IS THE DECIDING ROW.
a402 `monotone=False`    three attentions, but y = sum_k p_k with no chain - isolates whether
                         the structural monotonicity matters or only the extra capacity.
a404 `K=1`               collapses to a single attention with a sigmoid readout, i.e. the
                         fieldless baseline shape, as a sanity floor.

MECHANISM VERIFICATION (independent of accuracy)
------------------------------------------------
`attention_maps()` returns all K maps so the claim can be checked directly: if the maps for
k=1 and k=3 are near-identical the mechanism is INERT regardless of what the metrics say -
the same test that exposed ASGAP's learnable alpha as an inert parameter. Report the pairwise
agreement, not just QWK.

GUARDS carried over from this session's measured failures:
  * no instance-instance mixing -> the ctx-family G1 collapse (75->47, 55->12) cannot recur
  * entmax pooling retained     -> keeps the ASGAP G1/G2 advantage over softmax
  * shared bottleneck           -> the three branches cannot drift into separate encoders on 30 cases

Permutation-invariant, bag-size-invariant, deterministic at eval, MPS-safe.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .a215_adaptive_sparse_gated_attention_pooling import entmax_bisect

POOLS = ("entmax", "softmax")


class _GatedAttention(nn.Module):
    """Ilse-style gated attention producing one logit per instance."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.W = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.W(self.V(h) * self.U(h)).squeeze(-1)


class Model(nn.Module):
    def __init__(self, input_dim: int = 1280, num_classes: int = 1, hidden_dim: int = 128,
                 dropout: float = 0.5, thresholds: int = 3, share_attn: bool = False,
                 monotone: bool = True, pool: str = "entmax", alpha: float = 1.5):
        super().__init__()
        if pool not in POOLS:
            raise ValueError(f"pool must be one of {POOLS}")
        self.K = int(thresholds)
        self.share_attn = bool(share_attn)
        self.monotone = bool(monotone)
        self.pool = pool
        self.alpha = float(alpha)
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        n_attn = 1 if share_attn else self.K
        self.attn = nn.ModuleList([_GatedAttention(hidden_dim) for _ in range(n_attn)])
        # one scalar head per threshold (always separate - the thresholds are different questions)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(self.K)])

    def _pool(self, e: torch.Tensor) -> torch.Tensor:
        return torch.softmax(e, dim=0) if self.pool == "softmax" else entmax_bisect(e, self.alpha)

    def attention_maps(self, features: torch.Tensor) -> List[torch.Tensor]:
        """All K attention maps, for verifying the thresholds really look at different patches."""
        if features.dim() == 3:
            features = features.squeeze(0)
        h = self.bottleneck(features)
        return [self._pool(self.attn[0 if self.share_attn else k](h)) for k in range(self.K)]

    def forward(self, features, return_attention: bool = False, metrics=None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        if features.dim() == 3:
            features = features.squeeze(0)
        h = self.bottleneck(features)                                   # [N, H]

        maps, probs = [], []
        for k in range(self.K):
            a = self._pool(self.attn[0 if self.share_attn else k](h))   # [N]
            z = torch.mv(h.t(), a)                                      # [H]
            probs.append(torch.sigmoid(self.heads[k](z)).view(()))
            maps.append(a)

        if self.monotone:
            y, running = 0.0, torch.ones((), device=h.device, dtype=h.dtype)
            for p in probs:                                             # q_k = prod_{j<=k} p_j
                running = running * p
                y = y + running
        else:
            y = sum(probs)

        y = y.view(1)
        if return_attention:
            return y, maps[0], None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, thresholds=3, share_attn=False,
              monotone=True, pool="entmax")
