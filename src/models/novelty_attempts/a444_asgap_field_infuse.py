"""a444 — ASGAP with the global view conditioning the scorer AND entering the pooled vector.

Same conditioning as a443, but the pooled vector is built from the conditioned features, so the
field contributes a common shift to the bag representation. Still never scored, still never given
an attention weight. Parameter count identical to ASGAP (197,250).

Mechanism, modes and rationale: see ``field_before_pool.py``.
"""
from __future__ import annotations

from .field_before_pool import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, mode="infuse", field="roi")
