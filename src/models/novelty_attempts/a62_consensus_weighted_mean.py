"""a62 — Consensus-weighted mean (robust diffuse-density via self-agreement).

Hypothesis (Hn_consensus). A reticulin grade is the *diffuse fibre density*
read over the whole ROI. A plain attention-weighted mean (baseline) lets a
free attention scorer concentrate on whatever it likes, and a plain mean lets
a handful of bone/artifact patches drag the bag summary toward an off-tissue
direction. The grade-relevant signal is the *bag-wide CONSENSUS* of fibre
content: patches whose fibre reading AGREES with the bag's own consensus are
the diffuse meshwork we want to measure; patches that DISAGREE (bone trabecula,
fold, stain artifact) are the ones to down-weight. So:

    h_i  = Dropout(ReLU(Linear(1280->128) f_i))        # baseline encoder
    s_i  = <h_i, w>                                     # per-patch fibre score (1-d)
    s_bar= mean_i s_i                                   # bag CONSENSUS (self-referential)
    a_i  = softplus( beta * (s_i - s_bar) * sign(s_bar - center) )   # AGREEMENT
    p_i  = a_i / sum_j a_j                              # normalised, sum=1
    z    = sum_i p_i * h_i                              # consensus-weighted FEATURE mean
    y    = clamp( Linear(z) , 0, 3 )                    # linear, unbounded readout

Agreement = how far a patch's fibre score s_i sits ON THE CONSENSUS SIDE of the
bag mean. The factor `sign(s_bar - center)` makes "agreement" direction-aware:
in a high-fibre bag (s_bar > center) the agreeing patches are the *high*-score
ones (the dense meshwork); in a low-fibre bag they are the *low*-score ones
(the clean marrow). Either way we up-weight the patches that look like the
majority of the bag and suppress the minority outliers — a robust M-estimator-
style location read of the dominant tissue, NOT a coverage count and NOT a free
attention head. `center` is a learned scalar reference; `beta` a learned
temperature (clamped >0). `softplus` keeps every weight strictly positive (no
patch is ever fully gated out), so this is a *re-weighting*, never a selection.

Why this is GENUINELY different from refuted families
------------------------------------------------------
  - NOT plain mean-pool / a53/a57 (LINEAR): the weights p_i are a NONLINEAR
    function of the bag's own statistics (s_bar, softplus), so
    z != mean_i h_i in general; the bag readout depends on the agreement
    DISTRIBUTION, not just the centroid. (If all s_i equal, weights are uniform
    and it gracefully reduces to mean — the right limit.)
  - NOT coverage / extent / threshold / fraction (REFUTED 3x): nothing counts
    "what fraction of patches exceed tau". The output is a weighted FEATURE
    mean, then a linear head; the agreement enters as a continuous weight on
    each feature vector, never as a soft 0/1 indicator that is then averaged.
  - NOT ||h||-norm prior (refuted): the weight uses the SIGNED PROJECTION s_i
    along a learned fibre direction relative to the bag consensus — magnitude
    ||h|| never appears.
  - NOT a free attention scorer / attention concentration (a25, baseline gate):
    weights are not produced by an independent learned scorer that can latch
    onto anything; they are *tied to the bag's own consensus* (s_i - s_bar).
    A patch cannot get high weight on its own merit, only by agreeing with the
    majority — the opposite of attention's "find the salient one".
  - NOT top-k / argmax / trimmed / median: softplus weights are smooth, dense,
    strictly positive; no hard selection, no order statistics.
  - NOT a sigmoid/softmax GATE between bag rep and scalar (HARD constraint):
    the only nonlinearity touching the scalar path is softplus applied PER
    PATCH, BEFORE aggregation, to compute weights. The readout from the
    aggregated feature z to y is a single Linear, unbounded; clamp only at end.
    No gate multiplies/biases the final scalar.
  - DE01 (DeepSet overfit): capacity is the baseline encoder only; the
    consensus weighting adds just `w` (128), `beta` (1), `center` (1) and a
    Linear(128->1) head. Deliberately low-capacity.

Cosine variant (`agree_mode='cosine'`): instead of the 1-d signed projection,
weight by cosine agreement between h_i and the bag-consensus feature direction
(mean h, detached as the reference) along the fibre subspace. The default
'signed' mode is preferred: it is direction-aware about HIGH vs LOW fibre,
whereas raw cosine to the centroid would up-weight whatever dominates the
feature mean (often background), which we explicitly do not want.

Permutation-invariant (all ops are symmetric: mean, sum, softplus over patches)
and bag-size-invariant (p_i are normalised; duplicating the bag leaves s_bar,
the weight pattern, and z unchanged -> same y).

Ablation companion (suggested): a63 = set beta=0 (frozen) -> uniform weights ->
exact mean-pool + linear head. a62 vs a63 isolates the active ingredient:
"does CONSENSUS re-weighting beat the plain feature mean?".

Kill criterion: abandon if a62 val_qwk < 0.78 at seed=2 AND a62 <= a63
(consensus weighting adds nothing over mean). DoD = multi-seed audit {0,1,2,3,42}.

Param count (input_dim=1280, hidden=128):
    bottleneck Linear(1280,128)+b = 163,968
    fibre dir  w (128)            =     128
    beta (1) + center (1)         =       2
    head       Linear(128,1)+b    =     129
    -------------------------------------------
    total                         = 164,227   (< 197,250 baseline)
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
        agree_mode: str = "signed",     # "signed" (a62 main) | "cosine"
        beta_init: float = 1.0,         # agreement temperature (>0, learnable)
        center_init: float = 0.0,       # consensus side reference (learnable)
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        assert agree_mode in ("signed", "cosine"), agree_mode
        self.agree_mode = agree_mode
        self.clamp_output = clamp_output

        # Baseline-identical encoder (capacity lives here).
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Learned fibre direction in the encoded space (1-d per-patch score).
        self.w = nn.Parameter(torch.randn(hidden_dim) * (1.0 / hidden_dim ** 0.5))
        # Agreement temperature (kept > 0 via softplus on a raw param).
        self._beta_raw = nn.Parameter(torch.tensor(float(beta_init)))
        # Consensus-side reference: defines HIGH-fibre vs LOW-fibre bags.
        self.center = nn.Parameter(torch.tensor(float(center_init)))

        # Linear, unbounded readout from the consensus-weighted feature mean.
        self.head = nn.Linear(hidden_dim, num_classes)
        with torch.no_grad():
            self.head.bias.fill_(1.5)   # start in interior of [0,3]

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag; batch_size=1 trainer contract)
        h = self.bottleneck(features)                 # [N, hidden]
        s = h @ self.w                                # [N] per-patch fibre score
        s_bar = s.mean()                              # bag consensus (scalar)
        beta = F.softplus(self._beta_raw) + 1e-4      # > 0

        if self.agree_mode == "signed":
            # Direction-aware agreement: how far a patch sits on the consensus
            # side of the bag mean. side = +1 for high-fibre bags, -1 for low.
            side = torch.tanh(s_bar - self.center)    # smooth sign in (-1,1)
            agree = beta * (s - s_bar) * side         # [N]
        else:  # "cosine": agreement of h_i with the (detached) bag centroid
            ref = h.mean(dim=0).detach()              # [hidden] consensus feature
            ref = ref / ref.norm().clamp(min=1e-8)
            hn = h / h.norm(dim=1, keepdim=True).clamp(min=1e-8)
            agree = beta * (hn @ ref)                 # [N] in [-beta, beta]

        # Strictly-positive smooth weights (re-weighting, never a hard gate).
        a = F.softplus(agree)                         # [N] > 0
        p = a / a.sum().clamp(min=1e-8)               # [N] normalised, sum=1

        z = (p.unsqueeze(0) @ h).squeeze(0)           # [hidden] consensus mean
        y = self.head(z)                              # [1] linear, unbounded
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            return y, p, None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    agree_mode="signed",     # a62 main = direction-aware consensus agreement
    beta_init=1.0,
    center_init=0.0,
    clamp_output=True,
)
