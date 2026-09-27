"""a258 - Modern-Hopfield / energy associative-retrieval pooling.

THE WALL (verbatim context). No candidate has ever cleared the all-backbone gate, because one FIXED setting
cannot fit all three backbones across the val<->test QWK anti-correlation (~-0.95) on a tiny 214-ROI cohort.
The barrier is COHORT-driven, mechanism/capacity-independent; we are NOT promised any architecture breaks it.
a258 is a COMPLETENESS-COVERAGE contribution: a categorically-new bag REPRESENTATION family (associative
energy retrieval / modern Hopfield, Ramsauer 2020), not another attention-shape/temperature/sparsity/de-
concentration tweak (that entire lane -- entmax, size-temp, James-Stein, entropy-gated rank-cap, redundancy,
variance-reg, robust-MAD, content-gate, mean-blend -- is exhausted and failed).

MECHANISM (what makes a258 a distinct FAMILY). Rather than computing a relevance distribution over patches
and pooling a first-moment weighted mean, a258 forms the bag descriptor as the (near) fixed point of a
learned associative-energy dynamics seeded by the diffuse bag mean:
  1. Bottleneck:  h = Dropout(ReLU(Linear(input_dim, hidden_dim)))(features)      -> [N, D_h].
  2. Bag prior:   m = h.mean(dim=0)                                                -> [D_h].
       Permutation- and size-invariant; n=1 safe (m = h_0, no std, no /(N-1), no count division).
  3. Interaction: a raw learnable matrix Wp [D_h, D_h] init ~ randn * 1e-3 (shallow energy at init, so the
       dynamics start near "just relax onto the mean"). Symmetrized EACH forward: W = 0.5 * (Wp + Wp.t()),
       a valid symmetric Hopfield interaction. Gradients flow to all D_h^2 entries (no lone inert scalar).
  4. State init: q = m.
  5. Energy-gradient retrieval, iterate n_iters (fixed = 5) times:
         q = tanh( W @ q + m ).
       Each step blends the learned interaction dynamics (W @ q) with the diffuse bag-mean prior (+ m), so
       the recurrence is anchored to the bag and cannot run away; tanh keeps q bounded (a soft energy basin).
       After a few steps q settles near a fixed point of the associative dynamics -- the retrieval state.
  6. Descriptor: z = q  (a [D_h] bag descriptor -- still a genuine POOLED bag vector the head reads, NOT a
       median/quantile/histogram-only read that previously collapsed to ~random).
  7. Regress:   y = classifier(z).view(-1),  classifier = Linear(hidden_dim, num_classes).

WHY THIS IS A NEW REPRESENTATION (not a re-skinned attention pool). The bag enters ONLY through its mean m;
the patch-distinguishing work is done by the learned recurrent energy dynamics W operating on that summary,
producing a non-linear (tanh, multi-step) readout of the bag rather than a convex combination of patches.
There is no per-patch softmax relevance distribution and no temperature/sparsity knob to tune. This is the
associative-retrieval lane, categorically distinct from the de-concentration / attention-shape lane.

HONEST NOTES / APPROXIMATIONS (stated plainly).
  - This is an APPROXIMATE modern-Hopfield: in the strict Ramsauer 2020 formulation the update retrieves a
    softmax-weighted combination of STORED PATTERNS (the patches themselves). a258 instead runs a symmetric
    classical-Hopfield-style fixed-point iteration q = tanh(W q + m) on the bag MEAN with a learned coupling
    W, seeded and re-anchored by m. So it is "modern-Hopfield-INSPIRED energy retrieval," not the exact
    softmax-over-patterns retrieval; the patches contribute only through m and through W's learned structure.
    This is a deliberate capacity choice: the small cohort historically punishes capacity >> the 197K
    baseline, so the bag is summarized to one mean vector and the (D_h^2 = 16K) coupling does the modelling.
  - n_iters = 5 is a fixed unrolled depth, not an iterate-to-convergence guarantee; q is a NEAR fixed point.
    Determinism in eval() follows from the fixed iteration count and absence of randomness (dropout is the
    only stochastic layer and is disabled in eval()).
  - The +m re-injection each step makes the dynamics non-autonomous (a "clamped-input" Hopfield); this is
    what keeps the trajectory bounded and bag-anchored, at the cost of not being a pure autonomous energy
    descent. Stated for honesty: convergence is empirical, not proven for arbitrary W.
  - There is no per-patch saliency in this design, so return_attention has nothing meaningful to return;
    forward returns attn = None always.

CONSTRAINTS satisfied. Self-contained nn.Module; torch / nn / F only; no external files/deps. MPS-safe ops
only: matmul (W @ q), elementwise (tanh, +), mean -- NO linalg.solve/eigh, NO cdist, NO torch.median, NO
sort/scatter even. Permutation-invariant and bag-size-invariant: the bag enters only via m = mean(h) and the
fixed-depth W dynamics; no ordering, no (N-1), no count division beyond the mean. Deterministic in eval().
n=1 safe: m = h_0 is well-defined, the recurrence needs no std / no division by count. Learnable content
lives in bottleneck / Wp (D_h^2 entries) / classifier -- VECTORS/MATRICES/heads with non-flat gradients; no
lone learnable shape-scalar to go inert.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, n_iters=5):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        # Raw interaction parameter, init near zero -> shallow energy at start (dynamics ~ relax onto mean).
        # Symmetrized each forward so the energy is a valid symmetric Hopfield energy; D_h^2 learnable entries.
        self.Wp = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 1e-3)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.n_iters = int(n_iters)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)               # [N, D_h]
        m = h.mean(dim=0)                            # [D_h]  permutation- & size-invariant; n=1 safe
        W = 0.5 * (self.Wp + self.Wp.t())            # [D_h, D_h] symmetric Hopfield interaction
        q = m                                        # [D_h]  init state = bag-mean prior
        for _ in range(self.n_iters):                # fixed-depth associative energy retrieval
            q = torch.tanh(W @ q + m)                # blend learned dynamics with the diffuse prior each step
        z = q                                        # [D_h]  near-fixed-point bag descriptor (the pooled vector)
        y = self.classifier(z).view(-1)              # shape (1,)
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
