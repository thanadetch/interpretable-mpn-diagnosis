"""a266 - DeepSets multi-statistic learned readout. NEW family: a single pooled bag vector built by
CONCATENATING THREE complementary smooth permutation-invariant statistics of a learned per-patch
embedding, read by one head. (NOT a single attention-shaped first-moment mean, NOT VLAD/Fisher, NOT
covariance/spectral, NOT OT/Sinkhorn, NOT Hopfield/Perceiver, NOT plain mean+std.)

WHY THIS IS A CATEGORICALLY-DISTINCT FAMILY. The exhausted lane is the single attention-weighted mean:
one relevance weight vector, all the tried variation living in HOW that one weight is shaped (entmax,
size-temp, James-Stein, entropy-gated rank-cap, variance-reg, robust-MAD, content-gate, mean-blend).
The mean+std multi-stat pools (PerPatchScorePooling / MeanStdPool) only ever combined TWO statistics
(a first moment and a spread). a266 is a DeepSets-style multi-statistic readout that fuses THREE
complementary smooth aggregations into one descriptor:
  (1) an importance-WEIGHTED MEAN  (central tendency, attention-style first moment),
  (2) a temperature-modulated SMOOTH-MAX via log-sum-exp  (a soft tail / extremum statistic that the
      weighted mean cannot represent -- it tracks the strongest-responding patch per feature),
  (3) an importance-WEIGHTED centred SECOND MOMENT  (heterogeneity / spread around stat1).
The smooth-max (logsumexp) term is the genuinely new ingredient versus mean+std: it is an extremum-
sensitive statistic, not a spread, so the head sees central tendency AND tail AND dispersion together.
This matters for fibrosis grading where a small patch of dense reticulin (a tail/extremum signal) can
matter independently of the bag's average and its variance.

MECHANISM (matmul / elementwise / softmax / logsumexp only; no [N,K,D] tensor, no pairwise products).
  1. h = bottleneck(features) = Dropout(ReLU(Linear(input_dim, H)))                          -> [N,H]
  2. learned per-patch importance  w = softmax(score(h).squeeze(-1)),  score = Linear(H,1)   -> [N]
  3. stat1 (importance-weighted mean):   stat1 = Σ_i w_i * h_i                                -> [H]
  4. stat2 (bag-size-invariant smooth-max):
         stat2 = tau * ( logsumexp(h / tau, dim=0) - log N ),  tau = exp(log_tau).clamp(1e-2,50)
     The "- log N" makes it bag-size-invariant: for a constant feature it reduces EXACTLY to that
     constant for any N (logsumexp of N equal values c/tau is log N + c/tau, so tau*(...)-tau*logN = c).
     As tau -> 0 it approaches the hard per-feature max; as tau grows it approaches the (unweighted)
     mean. tau is a VECTOR-free scalar knob BUT it is NOT the only learnable content (the heads and
     bottleneck carry the load), and it enters multiplicatively around a non-trivial logsumexp, so it
     does not go inert the way a lone attention-shape scalar would.
  5. stat3 (importance-weighted centred second moment / heterogeneity):
         centred = h - stat1 ;  stat3 = sqrt( Σ_i w_i * centred_i^2 + eps )                  -> [H]
  6. z = cat([stat1, stat2, stat3])                                                          -> [3H]
  7. y = readout(z).view(-1),  readout = Linear(3H, num_classes)                             -> (1,)

HONEST NOTES / APPROXIMATIONS.
  - tau IS a single learnable scalar (log_tau). To avoid the documented "lone learnable shape-scalar goes
    inert" failure mode, the SUBSTANTIVE learnable content lives in the vectors/heads (bottleneck, score,
    readout); tau only modulates the softness of ONE of three statistics, around a real logsumexp whose
    gradient w.r.t. tau is non-trivial. It is not a global gain on the pooled vector and not the sole
    knob deciding relevance. If tau drifts inert, the other two statistics still fully determine the
    descriptor, so the model degrades gracefully rather than collapsing.
  - logsumexp is computed with the numerically-stable F.logsumexp (max-shift internally), so it is safe
    for any feature magnitude and for N=1 (logsumexp over a single element equals that element; with the
    "- log 1 = 0" correction stat2 reduces to h itself, i.e. the lone patch's embedding, which is the
    correct limit).
  - stat3 is a population (1/Σw) importance-weighted second moment with NO Bessel /(N-1) correction, so
    it is well-defined and finite at N=1 (centred = 0 -> stat3 = sqrt(eps), a small constant); there is
    no division by patch count or by (N-1) anywhere.
  - Bag-size invariance: stat1 uses w that sums to 1 over patches (softmax) -> independent of N; stat2 is
    explicitly "- log N" corrected; stat3 uses the same normalised w. No raw 1/N or count-dependent term
    survives, so doubling the bag with identical patches leaves z unchanged.
  - This is concept-free: no zero-shot text prompts, no bone/fibrosis labels, no norm-weighting (||h|| is
    never used as a relevance signal; w comes from the learned `score` head only).

CONSTRAINTS satisfied. Self-contained nn.Module; torch / torch.nn / torch.nn.functional only; no external
files, no new deps. MPS-safe ops ONLY: Linear, ReLU, Dropout, softmax, logsumexp, exp, clamp, sqrt,
elementwise mul/sub, cat -- NO linalg.solve, NO linalg.eigh, NO linalg.svd, NO cdist, NO torch.median.
Permutation-invariant: every per-patch contribution enters through a sum/softmax/logsumexp over patches,
so reordering patches leaves stat1, stat2, stat3, z, y unchanged. Deterministic in eval() (Dropout off;
all other ops deterministic). n=1 SAFE (limits documented above; no /N, no /(N-1), eps-guarded sqrt).
Capacity is modest: H=128 bottleneck, descriptor dim 3H=384, readout Linear from it; learnable params are
the bottleneck, score head, readout head (vectors/matrices) plus a single log_tau scalar.
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
        self.score = nn.Linear(hidden_dim, 1)                  # per-patch importance logits -> [N,1]
        self.log_tau = nn.Parameter(torch.zeros(()))           # tau = exp(log_tau) (smooth-max softness)
        self.readout = nn.Linear(3 * hidden_dim, num_classes)  # reads the 3-statistic descriptor
        self.eps = 1e-6

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                                  # [N,H]
        N = h.shape[0]
        w = F.softmax(self.score(h).squeeze(-1), dim=0)                # [N] importance, Σ_i w_i = 1

        # (1) importance-weighted mean (central tendency, first moment)
        stat1 = (w.unsqueeze(-1) * h).sum(dim=0)                       # [H]

        # (2) bag-size-invariant temperature-modulated smooth-max per feature
        tau = torch.exp(self.log_tau).clamp(1e-2, 50.0)               # scalar > 0
        logN = torch.log(torch.tensor(float(N), device=h.device, dtype=h.dtype))
        stat2 = tau * (torch.logsumexp(h / tau, dim=0) - logN)        # [H]

        # (3) importance-weighted centred second moment (heterogeneity / spread)
        centred = h - stat1.unsqueeze(0)                              # [N,H]
        stat3 = torch.sqrt((w.unsqueeze(-1) * centred * centred).sum(dim=0) + self.eps)  # [H]

        z = torch.cat([stat1, stat2, stat3], dim=0)                  # [3H]
        y = self.readout(z).view(-1)                                 # shape (1,)
        if return_attention:
            return y, w, None                                        # per-patch importance as attention
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
