"""a445 — TransMIL where the whole-ROI ("global") view IS the readout token.

a381 places the field view in the sequence as an extra token, and the prediction is still read
from a learned [CLS] that is identical for every bag. This module changes the field's ROLE rather
than its position — the field vector replaces [CLS], so the aggregation starts from this ROI's
own summary instead of from a learned constant.

    seq = [field] + patches          (a381: seq = [CLS] + [field] + patches)

Everything else — shared projection, PPEG on the patch block only, two attention layers, readout
from position 0 — is unchanged. ``cls_token`` stays allocated but receives no gradient, so the
allocated count equals plain TransMIL (2,805,249 on virchow2) and the effective count is 512 lower
(2,804,737). a381 is 512 higher (2,805,761) because of its modality embedding.

Mechanism and controls: see ``two_branch_transmil.py``.
"""
from __future__ import annotations

from .two_branch_transmil import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, fuse="cls", field="roi")
