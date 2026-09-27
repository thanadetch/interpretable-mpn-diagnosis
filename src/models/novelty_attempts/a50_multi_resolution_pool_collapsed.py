"""a50 - multi-resolution pool with single-cluster coarse branch (H21 ablation).

Philosophy bucket: multi_scale_fusion

Ablation companion to `a49_multi_resolution_pool.py`. Identical
architecture except `n_clusters=1`: the coarse branch collapses to a
single global-mean cluster rep instead of M=8 content-driven clusters.

The a49 vs a50 contrast isolates the active ingredient:
    a49 wins (>= +0.005 val) -> multi-cluster coarsening carries real
        regional signal beyond a global side-channel; the H21
        mechanism is meaningful and worth iterating on.
    a49 ~= a50              -> the gain (if any) over baseline is
        from the side-channel + bias-init 1.5 + concat-fuse head, NOT
        from multi-resolution per se. Fold into DE table; close H21.
    a50 wins                -> bigger N=8 coarse pool over-fits; close
        H21 and add a "soft-cluster centroids overfit" DE row.

When `n_clusters=1`, `a49.Model.__init__` skips allocating the
`centroids` parameter and the cluster-level gated attention (they
would be unused / gradient-free), and `_coarse_pool` short-circuits
to `h.mean(0)`. So a50's coarse branch is exactly a plain global mean.
a50 is mechanically equivalent to:
    "patch-gated-attn concat global-mean -> Linear(2*hidden, 1)
     with classifier bias init = 1.5".

Param count (input_dim=1280, hidden_dim=128):
    bottleneck Linear(1280, 128) + bias    = 163,968
    fine: V_p / U_p / W_p                  =  33,153
    fuse  Linear(256, 1) + bias            =     257
    --------------------------------------------------
    total trainable                        = 197,378
    (a49 M=8: 231,555; baseline 197,250 -> a50 +128 params,
     within rounding of the locked baseline -> a50 is also a clean
     "what does swapping border-white prior for a global-mean side-
     channel buy us?" control vs the locked ABMIL.)

Kill criterion: shared with a49 — abandon the H21 family if both
a49.val_qwk < 0.79 AND a50.val_qwk < 0.79.
"""
from __future__ import annotations

from .a49_multi_resolution_pool import Model as _Base


class Model(_Base):
    """Inherits a49's architecture; M=1 collapses coarse branch to global mean."""


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    n_clusters=1,        # ablation: single cluster -> global-mean coarse branch
    dropout=0.5,
    centroid_init_std=0.02,
)


