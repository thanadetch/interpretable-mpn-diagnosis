"""a443 — ASGAP with the global view conditioning the attention scorer only.

The field shapes WHICH patches are selected; the pooled vector is built from the patches alone,
so no component of the bag representation comes from the field. Parameter count is identical to
ASGAP (197,250) because the field reuses the patch bottleneck.

Mechanism, modes and rationale: see ``field_before_pool.py``.
"""
from __future__ import annotations

from .field_before_pool import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, mode="context", field="roi")
