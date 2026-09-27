"""a55 — Mean-projection cascade (ABLATION companion of a54).

Removes EXACTLY the active ingredient of a54: the per-patch ORDINAL COVERAGE
(sigmoid-then-mean soft fraction). Everything else is bit-for-bit identical to
a54 — same Linear(1280->128)+ReLU+Dropout(0.5) encoder, same THREE learnable
per-boundary directions d_k in bottleneck space, same monotone thresholds
(here used as additive per-boundary offsets), same Linear(3,1) cumulative-link
readout warm-started to sum_k c_k.

The ONLY change: step 4-5 of a54 (per-patch sigmoid indicator, then bag-wide
mean) is replaced by the LINEAR mean-projection

    c_k = mean_i (s_{i,k} - tau_k) = <mean_i h_i, d_k> - tau_k

i.e. mean-pool the bottleneck features first, then project onto d_k. This is a
linear functional of the mean-pooled bag (results/diag note: mean-projection ==
mean-pool + linear head), so a55 is a structured "mean-pool baseline" wearing
the same cascade head as a54.

a54 vs a55 therefore isolates the single question:
  "Does diffuse fibrosis EXTENT (coverage — a nonlinear fraction that
   mean-pooling CANNOT represent) beat fibrosis AVERAGE (mean-pool +
   linear) under the SAME ordinal-cascade structure?"
This is implemented by flipping `use_coverage=False` in the shared a54 Model,
so there is zero code drift between main and ablation.

If a54 > a55: the nonlinear coverage extent is the active ingredient.
If a54 ~= a55: extent buys nothing over the average -> the cascade structure
(directions + thresholds + linear readout) carries whatever signal there is,
and the contribution is the structure, not the coverage (clean negative on the
coverage hypothesis, still reportable for the systematic study).
"""
from __future__ import annotations

from models.novelty_attempts.a54_ordinal_coverage_cascade import Model  # re-export

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    n_boundaries=3,
    tau0_init=0.0,
    delta_init=0.5413248,
    beta_init=0.5413248,
    dir_init_scale=0.1,
    use_coverage=False,  # ablation: linear mean-projection instead of coverage
    clamp_output=True,
)
