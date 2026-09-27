"""a456 — a445 with a DEDICATED projection for the whole-ROI ("global") view.

THE QUESTION THIS ANSWERS
    a445 replaces TransMIL's learned [CLS] with this ROI's global view, and the replacement leaves
    ``cls_token`` unused — so a445 trains on 512 FEWER effective parameters than plain TransMIL and
    still wins 3/3. The obvious objection is the opposite of the usual one: not "you added
    capacity" but "did you REMOVE the capacity the readout token needed?".

    The two attempts already on record that add capacity back both fail, and they fail the same
    way: a448 (``clsadd``) adds ``cls_token`` to the field vector and a381/a449 add a ``modality``
    embedding. Both additions are CONSTANTS shared by every bag, so they drag every ROI's readout
    back toward one common point — undoing the conditioning a445 just introduced. a448 loses to
    a445 on virchow2 (.9177 vs .9675, G3 recall 57.5 vs 96.2) and titan; a449 loses 3/3.

    This module adds capacity that stays a FUNCTION OF THE ROI. In a445 the patches and the field
    share ``_fc1``:

        patches = _fc1(features)          # 224 px crops at native resolution
        fvec    = _fc1(global_view)       # the whole ROI downsized to 224 px

    Same frozen encoder, so the same 1280-d space — but not the same kind of input, and on this
    cohort the two views differ by roughly an order of magnitude in um/px. One matrix serving both
    is a compromise. Here the field gets ``_fc1g`` of its own.

    Nothing else changes: fuse="cls", PPEG on the patch block only, readout from position 0.

WHAT WOULD MAKE THIS A WIN, AND WHY WINNING IS NOT FREE
    A win means the readout token wanted a dedicated projection, not a bag-independent bias — and
    it would say the a448/a449 failures were about CONSTANTS, not about capacity.

    But ``_fc1g`` costs input_dim*hidden_dim + hidden_dim params: 655,872 at virchow2, ~+23% over
    plain TransMIL's 2,805,249. That spends a445's strongest writing position — that it beats
    TransMIL while using 512 params FEWER, so no reviewer can attribute the gain to capacity. If
    a456 wins, the parameter-matched comparison has to be rebuilt from scratch; if it loses, a445's
    shared projection stops being an untested limitation and becomes a measured design choice.
    Either outcome is reportable, which is why it is worth one run per backbone.

Gate: val QWK AND test QWK above a445 on the same backbone (seed 2, no augmentation).
Mechanism and controls: see ``two_branch_transmil.py``.
"""
from __future__ import annotations

from .two_branch_transmil import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, fuse="cls", field="roi", sep_proj=True)
