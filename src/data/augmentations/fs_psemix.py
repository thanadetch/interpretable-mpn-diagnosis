"""fs_psemix - PseMix with the two-branch field-bank lookup preserved.

Identical augmentation to ``psemix`` (pseudo-bag Mixup with a CROSS-GRADE partner and an
area-aligned soft target). The field token stays the clean whole-ROI embedding of THIS bag's ROI
while the patch bag becomes a mixture of two ROIs -- the strongest correspondence break in the
registry, which is exactly what makes it worth running. See ``_field_safe.py``.
"""
from __future__ import annotations

from ._field_safe import FieldSafe

KWARGS = dict(strength=0.5)


class Augmentation(FieldSafe):
    INNER = "psemix"
