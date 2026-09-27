"""a402_tsa_no_monotone — Threshold-Specific Attention MIL variant.

CONTROL for the chain. Three separate attentions but y = sum_k p_k with no monotonicity
enforced - isolates the structural ordering from the extra attention capacity.

    thresholds=3  share_attn=False  monotone=False  pool='entmax'

Mechanism, motivation and the falsification plan: see ``tsa_mil.py``.
"""
from __future__ import annotations

from .tsa_mil import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, thresholds=3, share_attn=False,
              monotone=False, pool="entmax")
