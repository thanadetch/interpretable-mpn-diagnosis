"""
a246_ordinal_survival_curve_pool — OrdinalSurvivalCurvePool
============================================================

MECHANISM
---------
A MIL aggregator that reads an ordinal-aware *soft survival curve* (smoothed
complementary ECDF) of per-patch severity over a FIXED, non-learnable grid, then
maps that fixed-length curve to a single regression logit with a linear head.

Per patch:
    h_i = Dropout(ReLU(Linear(input_dim, hidden_dim)))(f_i)
    s_i = Linear(hidden_dim, 1)(h_i)            # scalar "severity" score

Per bag standardization (n=1 safe, NO unbiased std):
    mu     = mean_i(s_i)
    z_i    = (s_i - mu) / (mean_i|s_i - mu| + 1e-3)   # mean-absolute-deviation scale

Soft survival curve on a FIXED grid g = linspace(-2.5, 2.5, 24) (registered buffer):
    surv_g = mean_i  sigmoid((z_i - g_g) / bw),   bw = fixed grid spacing
    -> surv is a length-24, monotone-non-increasing smoothed survival (1 - ECDF).

Read-out:
    y = Linear(24, 1)(surv).view(-1)            # bias init 1.5, RAW regression output

WHY THIS IS NEW (vs the exhausted families and "the wall")
----------------------------------------------------------
- ORDINAL but NOT the inert cumulative-link family. The tried ordinal models
  (a47/a89/a150 cumulative-link, a54 coverage cascade) collapse the bag to ONE
  scalar then apply <=3 LEARNABLE thresholds tau_k plus a LEARNABLE temperature
  -> exactly the "inert learnable scalar knobs" that converge back to init.
  Here there are NO learnable thresholds and NO learnable temperature: the grid
  and bandwidth are FIXED register_buffers and cannot go inert.
- Does NOT collapse to a single scalar. The linear head sees the FULL length-24
  survival profile and can place arbitrary monotone cumulative weights over it
  (a generalized expected-ordinal-count). A single-scalar+threshold model
  provably cannot reproduce this read.
- NOT a sorting / order-statistic / quantile read (not a105/a106/a197): every
  grid point is a full-bag mean of bounded sigmoids over ALL patches, so the
  descriptor is far lower-variance than any single order statistic or learnable
  quantile, and has no learnable q.
- NOT a single Gaussian rank-bump scalar (not a60).
- Wall fit (val<->test anti-correlation, val-selection is the ceiling): the read
  is a smooth, bounded, low-variance distributional summary. No softmax
  concentration and no learnable sharpness to over-fit a lucky val fold.
- Disjoint-helper fix: per-bag MAD standardization auto-normalizes each backbone's
  score scale onto the SAME fixed grid, so one read fits concentrated UNI2 and
  diffuse Virchow2/TITAN without any per-backbone learnable sharpness.

CONTRACT
--------
- Self-contained single nn.Module, torch/torch.nn/torch.nn.functional only.
- Permutation-invariant and bag-size/duplication-invariant (mean over patches).
- Deterministic in eval() (only Dropout is stochastic, disabled in eval()).
- n=1 safe: mad=0 -> denom=1e-3 -> z=0, all finite; no unbiased std.
- Receives only the feature tensor [N, D]; no coords/images/labels.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, n_grid=24):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.scorer = nn.Linear(hidden_dim, 1)

        grid = torch.linspace(-2.5, 2.5, n_grid)
        self.register_buffer("grid", grid)                       # FIXED, not learnable
        self.bw = float((grid[1] - grid[0]).item())              # FIXED bandwidth

        self.head = nn.Linear(n_grid, num_classes)
        with torch.no_grad():
            self.head.bias.fill_(1.5)

    def forward(self, features, return_attention=False, metrics=None):
        # features: [N, input_dim]
        h = self.bottleneck(features)                            # [N, hid]
        s = self.scorer(h).view(-1)                              # [N]

        mu = s.mean()
        sc = s - mu                                              # centered
        mad = sc.abs().mean()                                    # n=1-safe (no unbiased std)
        z = sc / (mad + 1e-3)                                    # [N]

        diff = (z.unsqueeze(1) - self.grid.unsqueeze(0)) / self.bw   # [N, G]
        surv = torch.sigmoid(diff).mean(dim=0)                   # [G] survival curve

        y = self.head(surv).view(-1)                             # [num_classes] RAW

        if return_attention:
            return y, None, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
