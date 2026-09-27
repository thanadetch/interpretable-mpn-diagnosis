"""fs_cwg - cutmix_within_grade, with the two-branch field-bank lookup preserved.

Identical augmentation to ``cutmix_within_grade``; the only difference is that the augmented
bag keeps its link to the ROI's whole-ROI field embedding, so ``a381``/``a380`` still receive a
real field token instead of silently degrading to the a384 patch-mean control.
See ``_field_safe.py`` for the mechanism and for what this means when reporting.
"""
from __future__ import annotations

from ._field_safe import FieldSafe

KWARGS = dict(strength=0.8)


class Augmentation(FieldSafe):
    INNER = "cutmix_within_grade"
