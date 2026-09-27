"""fs_rxjoint - remixp_joint with the two-branch field-bank lookup preserved.

Identical augmentation to ``remixp_joint`` (ReMix reduce-then-mix, all four latent ops).
NOTE this one can CHANGE THE BAG LENGTH (the `append` op concatenates bank prototypes), so the
augmented bag's fingerprint differs in shape as well as value; the wrapper aliases it regardless.
See ``_field_safe.py``.
"""
from __future__ import annotations

from ._field_safe import FieldSafe

KWARGS = dict(strength=0.5)


class Augmentation(FieldSafe):
    INNER = "remixp_joint"
