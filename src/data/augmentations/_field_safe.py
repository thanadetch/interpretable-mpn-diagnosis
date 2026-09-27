"""_field_safe - keep the two-branch field-bank lookup alive under feature augmentation.

WHY THIS EXISTS
    ``two_branch_transmil`` (a380/a381/a384/a386) resolves its whole-ROI "field" token by
    VALUE FINGERPRINT: ``_fingerprint(bag) = (N, D, bag[0,0], bag[0,-1], bag[-1,0], bag[-1,-1])``.
    That design keeps the model self-contained (no trainer edits, no extra dataloader field),
    but it assumes the tensor the model receives is byte-identical to the one on disk.

    Every feature-space augmentation breaks that assumption - it edits patch values, and
    frequently the bag length too. The lookup then MISSES and ``_field`` silently falls back to
    ``features.mean(dim=0)``. That fallback is not a degraded a381: it is exactly a384, the
    mean-patch control. So "a381 + augmentation", run naively, measures a384 + augmentation and
    the comparison against no-aug a381 is meaningless.

WHAT THIS DOES
    Wraps any registry augmentation. It fingerprints the bag BEFORE augmenting, runs the inner
    augmentation, fingerprints the RESULT, and registers the result's fingerprint in the shared
    field bank as an alias pointing at the original ROI's field vector. ``_BANKS`` in
    ``field_mil`` is a module-level cache, so the model sees the alias immediately and its
    lookup hits again - with zero trainer edits, matching the registry contract.

SEMANTICS (state this when reporting)
    The field token stays the CLEAN whole-ROI embedding of the ROI this bag came from, while
    the patch bag is augmented. That is the intended reading: the augmentation resamples the
    ROI's patch population, it does not create a new ROI. It also means the field token is a
    fixed, un-augmented anchor - which is precisely the "bag-consistent context token" a381 is
    claimed to exploit, now tested under patch-level perturbation.

    Aliases accumulate for the life of the process (one per augmented bag per epoch, values are
    shared references, so the cost is dict overhead only). Eval passes are un-augmented and hit
    the original fingerprints directly. A fingerprint that would collide with a real bag is
    never overwritten.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from . import BaseAugmentation, load_augmentation
from models.novelty_attempts.field_mil import _bank, _fingerprint


class FieldSafe(BaseAugmentation):
    """Generic wrapper. Subclasses pin ``INNER``; ``strength`` is forwarded to the inner aug."""

    INNER: Optional[str] = None

    def __init__(self, strength: Optional[float] = None, inner: Optional[str] = None,
                 data_root: Optional[str] = None):
        name = inner or self.INNER
        if not name:
            raise ValueError("_field_safe: no inner augmentation specified")
        self.name = name
        self.inner = load_augmentation(name, strength)
        self.requires_regression = getattr(self.inner, "requires_regression", False)
        self.data_root = Path(data_root or os.environ.get("MPN_DATA_ROOT", "data"))
        self.aliased = 0
        self.unresolved = 0

    def __call__(self, features, label, pool=None):
        key_before = _fingerprint(features)
        out, target = self.inner(features, label, pool=pool)
        try:
            bank = _bank(int(out.shape[1]), self.data_root)
        except Exception:                                    # no bank for this dim -> nothing to do
            return out, target
        vec = bank.table.get(key_before)
        if vec is None:                                      # bag itself absent from the bank
            self.unresolved += 1
            return out, target
        key_after = _fingerprint(out)
        if key_after != key_before and key_after not in bank.table:
            bank.table[key_after] = vec
            self.aliased += 1
        return out, target
