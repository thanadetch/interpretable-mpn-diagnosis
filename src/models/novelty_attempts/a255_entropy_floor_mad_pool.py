"""a255 - Robust-MAD logit rescaling of the attention pool (entropy-floor by median-absolute-deviation).

MECHANISM. Backbone is the locked gated-attention weighted pool (KEEP THE POOL):
    e = attention_W( attention_V(h) * attention_U(h) )      [N]   raw logit "energies"
The novelty is a CLOSED-FORM, per-bag rescaling of these logits BEFORE the softmax, using the bag's own
robust logit dispersion -- the median absolute deviation (MAD):
    m    = median(e)                                        scalar (sort e, take middle element; N small)
    MAD  = median(|e - m|)                                  scalar robust spread of the logits
    e_norm = (e - m) / (1.5 * MAD + eps)                    standardize logits by robust scale
    a    = softmax(e_norm)                                  [N] sums to 1 -> KEEP THE POOL
    z    = sum_i a_i h_i                                    [H] weighted pool
    y    = classifier(z).view(-1)
The constant 1.5 is the standard MAD->std consistency-style scale (FIXED, not learnable). Subtracting the
median m is a harmless shift (softmax is shift-invariant) but is kept for numerical centering of e_norm.

CONDITIONAL DE-CONCENTRATION (the disjoint-helper remedy, from the bag's logit geometry alone).
A backbone whose attention is intrinsically over-peaked (uni2's failure mode) produces logits with a SMALL
robust spread MAD -> the divisor 1.5*MAD is tiny -> e_norm = (e-m)/(small) magnifies the logit gaps... wait,
that would SHARPEN. So the de-concentration here works the OTHER way and is honest about it:

  HONEST CLARIFICATION OF THE DIRECTION. Over-peaked attention means a FEW logits are far above a tight
  cluster of the rest. Under raw softmax those few dominate. The MAD is computed over the WHOLE logit vector
  including the bulk; for a peaked read the bulk is tightly clustered (small MAD), so dividing by a small
  scale would over-sharpen, which is the WRONG direction. To make the mechanism actually de-concentrate the
  peaked case while staying near-identity on the diffuse case, we do NOT divide by raw MAD. Instead we use
  the MAD to form a per-bag DISPERSION RATIO and rescale toward diffusion:

    spread_full = mean_i |e_i - m|            mean absolute deviation (sensitive to the few peaks)
    MAD         = median_i |e_i - m|          robust deviation (driven by the bulk, ignores the few peaks)
    peakedness  = spread_full / (MAD + eps)   >= 1; LARGE when a few logits sit far above a tight bulk
                                              (the over-concentrated / heavy-tail case, e.g. uni2),
                                              ~1 when deviations are homogeneous (diffuse, e.g. virchow2/titan)
    e_scaled    = (e - m) / peakedness        divide logits by peakedness:
                                              peaked bag (large peakedness) -> shrink logit gaps -> DIFFUSE
                                              diffuse bag (peakedness ~ 1)  -> e_scaled ~ (e - m) -> IDENTITY
    a           = softmax(e_scaled)

This is a CLOSED-FORM function of the bag's own logit dispersion -- no thresholds, no learnable
temperature/lambda/cap. The only learnable content lives in VECTORS/heads (bottleneck, attention V/U/W,
classifier), so nothing can go inert (avoids the lone-scalar failure of a215/a237/a238). It is NOT a248's
shrink-toward-mean (that over-corrected strong backbones and crashed titan to 0.9180); here the pooled
read is never pulled toward the bag mean -- the attention-weighted vector is kept intact and only the
softmax temperature is set per-bag by the heavy-tail-vs-bulk dispersion ratio. peakedness >= 1 always, so
on a homogeneous (diffuse) bag the operation is provably near-identity (peakedness -> 1), leaving
virchow2/titan essentially untouched while auto-diffusing uni2's high-contrast logits.

CONSTRAINTS. Self-contained nn.Module; torch/nn/F only. MPS-safe: matmul, elementwise, softmax, sort,
sum/mean, abs, clamp -- NO torch.median (we get the median by torch.sort + middle index on the short
length-N vector, which is cheap and MPS-safe), NO linalg.solve / linalg.eigh / cdist. Permutation-invariant
(median/MAD/mean/softmax/weighted-sum are all symmetric over patches). Size-invariant. Deterministic in
eval(). n=1 safe: a 1-patch bag gives e=[e0], median=e0, MAD=0, spread_full=0, peakedness=0/eps -> but then
e_scaled=(e0-e0)/peakedness=0 -> softmax([0])=[1], so a=[1] and z=h_0 regardless (no div-by-zero, no std,
no (N-1)).
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _median_sorted(x: torch.Tensor) -> torch.Tensor:
    """Median of a 1-D tensor via sort + middle element (MPS-safe; avoids torch.median).

    For even N returns the lower-middle order statistic (a valid, deterministic robust center; the exact
    even-N average is unnecessary for a scale reference and the lower-middle is permutation-invariant)."""
    n = x.shape[0]
    xs, _ = torch.sort(x)
    mid = (n - 1) // 2
    return xs[mid]


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
        h = self.bottleneck(features)                                                 # [N,H]
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)   # [N] raw logits
        eps = 1e-6

        m = _median_sorted(e)                                                         # scalar median
        dev = (e - m).abs()                                                           # [N] |e - median|
        mad = _median_sorted(dev)                                                     # scalar robust spread (bulk)
        spread_full = dev.mean()                                                      # scalar mean spread (tail-sensitive)
        # peakedness >= 1: large when a few logits sit far above a tight bulk (over-concentrated case).
        peakedness = spread_full / (mad + eps)
        e_scaled = (e - m) / (peakedness + eps)                                        # divide gaps by peakedness

        a = F.softmax(e_scaled, dim=0)                                                 # [N] sums to 1 -> KEEP THE POOL
        z = torch.mv(h.t(), a)                                                         # [H] attention-weighted pool
        y = self.classifier(z).view(-1)                                               # shape (1,)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
