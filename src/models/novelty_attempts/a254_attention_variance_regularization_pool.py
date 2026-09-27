"""a254 — attention-variance-regularization pool (conditional de-concentration via a
learned-direction correction, gated by a knob-free per-bag variance statistic).

THE WALL (uni2-test > 0.9418 while keeping titan/virchow2): one FIXED attention setting
cannot fit all three backbones. uni2's baseline attention is intrinsically the most
concentrated (norm-entropy 0.891 vs virchow2 0.965 / titan 0.947). Sharpening helps the
diffuse backbones but over-concentrates uni2 and discards the diffuse density grading
needs. So the only viable mechanism CONDITIONALLY DE-CONCENTRATES an over-peaked read
(helps uni2) while staying near-identity on an already-diffuse read (neutral on
virchow2/titan) -- inferred per-bag from the bag's OWN content, since the model cannot
see which backbone it is.

WHY THIS IS CATEGORICALLY DIFFERENT from prior failed de-concentrators:
- NOT a shrink-to-mean (a237/a243/a248): those PULL the read toward the bag/attention mean
  unconditionally or via a content gate, and over-correct the strong backbones (a248 titan
  crashed to 0.9180). Here the backbone read z_attn is KEPT verbatim; we ADD a small,
  learned-DIRECTION correction z_reg that is *zero-centred* under the attention measure
  (sum_i a_i corr_i with corr a function of the centred deviations delta_i = h_i - z_attn),
  so it does not translate the read toward a mean -- it reshapes it along learned axes.
- NOT a lone learnable shape-scalar (a215/a238/a237 went inert): the correction's content
  lives in a full Linear head (corr = Linear(delta), a VECTOR map with non-flat gradient,
  zero-initialised so corr=0 -> exact baseline at init). The de-concentration GATE tau is a
  CLOSED-FORM, knob-free function of the bag statistics -- no learnable scalar to go flat.
- NOT a redundancy/repulsion reweighting of attention (a244): attention `a` is left
  untouched; we operate in the residual (deviation) space the attention pool throws away.

MECHANISM (per bag):
  1. h = bottleneck(features)                                  [N,H]
  2. a = softmax(W(V(h)*U(h)))                                 [N]   (gated attention, kept)
  3. z_attn = sum_i a_i h_i                                    [H]   (backbone read, KEPT)
  4. delta_i = h_i - z_attn                                    [N,H] (attention-centred residual)
  5. var_delta = sum_i a_i ||delta_i||^2                       []    (attn-weighted spread)
     scale     = sum_i a_i ||h_i||^2 + eps                     []    (attn-weighted energy)
  6. corr_i = Linear_corr(delta_i)                             [N,H] (zero-init -> 0 at start)
  7. z_reg = sum_i a_i corr_i                                  [H]   (zero-centred residual read;
                                                                       n=1 -> delta=0 -> z_reg=0)
  8. tau = sigmoid(var_delta / scale) - 0.5                    []    in (-0.5, 0.5), KNOB-FREE
  9. z = z_attn + tau * z_reg                                  [H]
 10. y = classifier(z).view(-1)

CONDITIONAL DE-CONCENTRATION, honestly stated:
  tau is a closed-form monotone function of the *relative* attention-weighted variance of
  the deviations (var_delta normalised by the bag's energy). A tightly clustered read
  (attention pulled onto a few near-identical patches, var_delta -> 0, uni2's over-peaked
  failure mode shows LOW residual energy because the few attended patches dominate the
  weighted mean) yields tau near sigmoid(0)-0.5 = 0 -> z ~ z_attn (near-identity). A read
  whose attended mass is spread across structurally different patches (high relative
  var_delta) yields a larger tau, activating the learned-direction correction proportional
  to the measured spread. The correction DIRECTION is learned (Linear_corr); the AMOUNT is
  the data-derived statistic. So the model infers concentration from its own variance and
  reshapes the read along learned axes, per-bag, with no fixed global rule and no knob.

  APPROXIMATION DISCLOSED HONESTLY: tau here is bounded to (-0.5, 0.5) by the sigmoid-minus-
  half form taken verbatim from the approved candidate spec; sigmoid(x)-0.5 is >= 0 for
  x >= 0 and var_delta/scale >= 0, so in practice tau in [0, 0.5). It is a MONOTONE proxy
  for "how diffuse is the attended mass", NOT a calibrated probability or a guaranteed
  identity at a single backbone -- it is a smooth statistic, so it is never *exactly* 0 for
  a multi-patch bag, only small. Whether this proxy de-concentrates uni2 enough without
  perturbing titan/virchow2 is an EMPIRICAL question the gate decides; nothing here proves
  it clears the wall.

CONSTRAINTS satisfied: self-contained (torch/nn/F only); permutation-invariant (all reduces
are sum_i a_i ... over the bag) and bag-size-invariant (a sums to 1; no /N, no /(N-1), no
std); n=1 safe (single patch -> a=[1], z_attn=h_0, delta=0, var_delta=0, z_reg=0, tau=0 ->
z=z_attn, no division by zero/std/(N-1)); deterministic in eval() (dropout off, all ops
deterministic). Concept-free (no text prompts, no bone/fibrosis labels). NOT norm-weighted
(||h|| enters only inside the scalar variance-ratio gate, never as a per-patch weight on the
pooled vector). MPS-safe: only matmul / elementwise / softmax / tanh / sigmoid / norm.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # Gated attention (V * U -> W), the kept backbone read.
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        # Learned-DIRECTION correction in the residual (deviation) space.
        # Zero-initialised so corr == 0 at init -> z == z_attn -> exact gated-attention baseline.
        self.correction = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                                   # [N,H]
        a = F.softmax(
            self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1), dim=0
        )                                                               # [N], sums to 1
        z_attn = a @ h                                                  # [H] kept backbone read

        delta = h - z_attn.unsqueeze(0)                                 # [N,H] attention-centred residual
        sq_delta = (delta * delta).sum(dim=1)                           # [N] ||delta_i||^2
        var_delta = (a * sq_delta).sum()                                # [] attn-weighted spread
        sq_h = (h * h).sum(dim=1)                                       # [N] ||h_i||^2
        scale = (a * sq_h).sum() + self.eps                             # [] attn-weighted energy

        corr = self.correction(delta)                                   # [N,H] learned-direction, zero-init
        z_reg = a @ corr                                                # [H] zero-centred residual read
        tau = torch.sigmoid(var_delta / scale) - 0.5                    # [] knob-free, in (-0.5, 0.5)

        z = z_attn + tau * z_reg                                        # [H]
        y = self.classifier(z).view(-1)                                 # [num_classes]
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
