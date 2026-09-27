"""a48 — gated-attention aggregator with vanilla regression head (H23 ablation).

Philosophy bucket: ordinal_head

Ablation companion to ``a47_ordinal_cumulative_link.py``. Identical
architecture with ``use_ordinal_head=False``, so the cumulative-link
expected-class head is replaced by the standard regression head:

    ŷ = clamp(Linear(128, 1)(bag_rep), 0, 3)

Active ingredient under test
----------------------------
The **cumulative-link ordinal head**. The a47 ↔ a48 contrast isolates
whether the ordinal expected-class parameterisation (sum of 3
cumulative-tail sigmoids with monotone thresholds) carries any value
on top of the same gated-attention aggregator.

Positive-control role
---------------------
a48 also serves as a re-implementation of the locked baseline
ABMIL (minus the ``attention_logit_bias`` border-white prior
that lives inside ``src/models/simple_mil.py``). If a48's val_qwk is
significantly below the baseline 0.8182, that gap quantifies the
contribution of the border-white prior to the baseline number — a
useful side-finding for the systematic study chapter.

Predictions
-----------
- a48 ≈ baseline → aggregator is faithful; any a47 − a48 gap is purely
  the ordinal head.
- a48 < baseline by > 0.01 → the border-white logit-bias prior is
  contributing materially; record this for the thesis.
- a47 > a48 → ordinal head is the active ingredient.
- a47 ≤ a48 → ordinal head not worth it; family essentially dead.

Kill criterion (family-level)
-----------------------------
Inherited from a47 / H23: if BOTH a47 AND a48 have val_qwk < 0.79 at
seed=2, declare H23 family dead.

Param count: same as a47 minus 3 threshold scalars = 197,250 (matches
the baseline exactly).
"""
from __future__ import annotations

from .a47_ordinal_cumulative_link import Model as _Model


class Model(_Model):
    """Inherits a47's aggregator + score head; disables the ordinal head."""


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    tau0_init=0.5,       # unused when use_ordinal_head=False
    delta_init=0.5413248,  # unused when use_ordinal_head=False
    use_ordinal_head=False,
)

