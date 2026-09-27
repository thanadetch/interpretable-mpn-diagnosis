"""a247_witness_mmd_cumulative — WitnessMMDCumulative.

Mechanism
---------
Self-contained MIL aggregator that fuses two open directions for the
ROI-level reticulin fibrosis grading problem:

  (1) DISTRIBUTION-MATCHING via the (biased) MMD *witness function*, evaluated
      against LEARNED reference clouds — NOT Sinkhorn-OT, NOT random Fourier
      features.
  (2) ORDINAL cumulative read INSIDE the aggregation, exploiting the ordinal
      G0..G3 grade structure with three threshold contrasts {>=1, >=2, >=3}.

Patches are projected X = Dropout(Linear(input_dim, d, bias=False))(features).
A single shared RBF kernel k(a,b) = exp(-gamma * ||a-b||^2) is used, where
`gamma` is a FIXED registered buffer (= 1/d). It is deliberately NOT learnable:
prior experiments showed lone learnable shape-scalars converge back to their
init (inert, flat gradient). The learnable content instead lives in the
*coordinates* of the reference clouds, whose gradients are non-flat.

For each ordinal threshold t in {1,2,3} we keep an "above" cloud A^(t) and a
matched "below" cloud B^(t) (Parameters of shape [3, M, d]). The signed
MMD-witness statistic for the bag is

    w_t = mean_{i,m} k(h_i, A^(t)_m) - mean_{i,m} k(h_i, B^(t)_m)

i.e. the bag's kernel-mean similarity to the "above" reference minus the
"below" reference — the MMD witness direction integrated over the bag's
empirical distribution. We map each contrast to a monotone cumulative
probability P(grade >= t) = sigmoid(w_t) and read the expected ordinal grade

    y = sum_{t=1}^{3} sigmoid(w_t) = sum_t P(grade >= t) = E[grade] in [0, 3].

Why this addresses THE WALL (and is genuinely new)
--------------------------------------------------
- No per-patch attention and no sharpness/temperature knob. Each w_t is a
  kernel-MEAN over ALL patches — the lowest-variance distribution read short of
  plain mean, yet nonlinear — so it cannot crash UNI2 via over-concentration.
  This is what blocked the all-backbone gate before.
- Splitting into 3 independent low-variance threshold contrasts decorrelates
  errors and lowers the variance of the summed grade relative to a single
  softmax read, which is structurally more robust to val-selection resampling
  (the actual enemy: val<->test QWK anti-correlation on the tiny val cohort).
- Distinct from prior families: a236 mean-pools FIXED random Fourier cos
  features of one bag then regresses; here there are NO RFF — we evaluate the
  EXACT RBF kernel-mean against LEARNED above/below clouds and form signed
  MMD-witness contrasts. a74 used 1-D Wasserstein to an external (prohibited)
  file. a47/a150 cumulative-link fit free thresholds on a single 1-D score;
  here the cumulative probabilities come from distribution-level MMD-witness
  contrasts in d dimensions. Not Sinkhorn (a181).

Hard-constraint compliance
--------------------------
- Self-contained single nn.Module; only torch / torch.nn used. No external
  artifact files, no new pip deps.
- Permutation-invariant and bag-size-invariant (everything reduces via a mean
  over patches).
- n=1 safe: every statistic is a mean of kernel evaluations; no .std, no
  unbiased variance, no division by (N-1) or by bag-derived counts.
- Deterministic at inference: Dropout is the only stochastic op and is disabled
  in eval(); argmax tie-breaking is deterministic.
- Output contract: forward returns (y, attn_or_None, None); y.view(-1) has
  shape (num_classes,) = (1,) for the locked scalar-regression head.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5,
                 d=32, M=8):
        super().__init__()
        assert num_classes == 1, "Locked scalar-regression head expects num_classes == 1"
        self.proj = nn.Linear(input_dim, d, bias=False)
        self.drop = nn.Dropout(dropout)
        # grade>=1, grade>=2, grade>=3 reference clouds (learnable coordinates)
        self.above = nn.Parameter(torch.randn(3, M, d) * (d ** -0.5))
        self.below = nn.Parameter(torch.randn(3, M, d) * (d ** -0.5))
        # FIXED kernel bandwidth (NOT a learnable shape-scalar -> avoids inert knob)
        self.register_buffer('gamma', torch.tensor(1.0 / d))

    @staticmethod
    def _sqdist(X, R):
        # ||x-r||^2 = ||x||^2 + ||r||^2 - 2 x.r  via matmul (MPS-native;
        # torch.cdist backward 'aten::_cdist_backward' is not implemented on MPS).
        x2 = (X * X).sum(dim=1, keepdim=True)      # [N,1]
        r2 = (R * R).sum(dim=1).unsqueeze(0)       # [1,M]
        return (x2 + r2 - 2.0 * (X @ R.t())).clamp_min(0.0)  # [N,M]

    def _kmean(self, X, R):
        # Mean RBF similarity between every patch in X [N,d] and every
        # reference point in R [M,d]; scalar. Mean over patches -> bag-size
        # invariant; n=1 safe (a plain mean over N*M kernel evals).
        d2 = self._sqdist(X, R)              # [N, M]
        return torch.exp(-self.gamma * d2).mean()

    def forward(self, features, return_attention=False, metrics=None):
        X = self.drop(self.proj(features))   # [N, d]
        ws = [self._kmean(X, self.above[t]) - self._kmean(X, self.below[t])
              for t in range(3)]
        w = torch.stack(ws)                  # [3]
        cdf = torch.sigmoid(w)               # P(grade >= t), t = 1, 2, 3
        y = cdf.sum().view(-1)               # expected ordinal grade in [0, 3]
        if return_attention:
            tstar = int(w.argmax())
            sim = torch.exp(-self.gamma * self._sqdist(X, self.above[tstar])).mean(dim=1)
            a = torch.softmax(sim, dim=0)    # per-patch weights, sum to 1, length N
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
