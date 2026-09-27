"""a108 — Ablation of a107: dual-view sign-AGREEMENT term REMOVED.

Imports a107's Model and pins gamma_fixed=0.0, which deletes EXACTLY the active
ingredient (the sign-agreement product of the raw-level and rank-shape views).
With gamma==0:

    y = w_raw * m_raw + b          # plain warm diffuse-mean readout (TIED baseline)

The bottleneck, the shared projection w, w_raw, the bias and the warm-start are
byte-identical to a107. a107 vs a108 isolates whether the dual-view agreement
veto moves the PAIRED cross-fold delta, with everything else held fixed.
"""
from __future__ import annotations

from .a107_dual_view_agreement_pool import Model  # re-export same class

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    gamma_init=0.1,        # ignored when gamma_fixed is set
    gamma_fixed=0.0,       # <-- removes the agreement term (the active ingredient)
    warm_start=True,
    prototype_path=None,
    eps=1e-6,
)
