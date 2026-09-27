"""a46 — rank-norm soft-mean, CONSTANT temperature (H22 ablation companion).

Philosophy bucket: norm_based_salience

Ablation companion to `a45_rank_norm_lengthnorm.py`. Identical architecture
with `length_norm=False`, so the softmax temperature is:
    τ = softplus(c_raw)            (NO √N factor)

This is the a44-equivalent re-implemented inside the
`norm_based_salience` philosophy bucket with a LEARNABLE constant τ
(a44 used a hard-coded τ = 8.0). The two differences from a44 are:
    1. τ is learnable rather than fixed at 8.0
    2. The module is now tagged with a Philosophy bucket: header so it
       feeds the bucket-saturation diagnostics correctly.

Active ingredient under test (a45 ↔ a46 contrast)
-------------------------------------------------
The **bag-size adaptive √N factor** in the softmax temperature. If
a45 > a46 by ≥ 0.005 val_qwk at seed=2, length-normalised temperature
is doing useful work on top of rank-by-norm pooling and should be
kept in any future compound. If a45 ≤ a46, drop the √N factor and
revisit τ alone as the only knob.

Predictions
-----------
- a45 > a46 → length-norm temperature is the active ingredient.
- a46 > a45 → constant-τ wins; the rank-by-norm pool is already
  bag-size-aware enough through ranking alone.
- a45 ≈ a46 ≈ a44 → the norm_based_salience family is saturated at
  ~0.798 val_qwk; close the bucket and move to ordinal_head (§12.4).

Kill criterion
--------------
Inherited from a45 / H22. This ablation companion has no independent
kill threshold beyond the family-level rule.

Param count: same as a45 = 164,098 trainable.
"""
from __future__ import annotations

from .a45_rank_norm_lengthnorm import Model as _Model


class Model(_Model):
    """Inherits a45's architecture; disables the √N temperature scaling."""


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    c_raw_init=7.999664,  # softplus(7.9997) ≈ 8.0; matches a44's fixed τ at init
    length_norm=False,
    clamp_output=True,
)


