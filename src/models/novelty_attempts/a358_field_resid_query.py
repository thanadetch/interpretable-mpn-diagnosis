"""a358_field_resid_query — Field-Conditioned MIL variant.

RESIDUAL FIELD QUERY — the control turned into a mechanism. The field view is projected off
the patch mean before it becomes the query, so the query carries ONLY what no patch in the
bag already contained. If this still beats the baseline, the a347 'it is just a bag-adaptive
query' explanation is dead by construction, not just by comparison.

    attn='query'  pool='entmax'  readout='concat'  field='resid'   base: ASGAP (alpha=1.5 entmax pooling)

Mechanism, controls and rationale: see ``field_mil.py``.
"""
from __future__ import annotations

from .field_mil import Model  # noqa: F401

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    attn="query",
    pool="entmax",
    readout="concat",
    field="resid",
)
