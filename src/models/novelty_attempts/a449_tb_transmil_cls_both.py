"""a449 — a381 and a445 combined: the field is BOTH the readout token and an extra token.

    a381      seq = [CLS]   + [field+mod] + patches    field is READ
    a445      seq = [field]               + patches    field READS
    a449      seq = [field] + [field+mod] + patches    field does both      <- this one

The two positions play different roles, so this is not a duplicated input: position 0 is the vector
the prediction is read from, position 1 is an element the patches can attend to and which carries
the learned modality marker. The learned ``cls_token`` is unused, as in a445.

Parameter count matches a381 (2,805,761) because the modality embedding is allocated.
"""
from __future__ import annotations

from .two_branch_transmil import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, fuse="clsboth", field="roi")
