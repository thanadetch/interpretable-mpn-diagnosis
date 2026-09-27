"""a460 - CONTROL for a445/a458/a459: PPEG replaced by a per-token FFN that mixes NOTHING.

Removing PPEG outright costs up to .074 on this data (a451), so PPEG is doing real work. This asks
what kind of work. A per-token bottleneck MLP has PPEG's depth and roughly its parameter budget
(43,562 vs 44,032) but zero communication between tokens.

If a460 matches PPEG, the gain is capacity and depth, not neighbourhood mixing, and the positional
reading of PPEG on ROI-scale bags should be dropped entirely. If a460 loses, mixing is the active
ingredient and a458/a459 say which geometry it wants.

Gate: this is a control, not a candidate -- report it beside a445/a458/a459, do not select on it.
"""
from __future__ import annotations
from .two_branch_transmil import Model  # noqa: F401
KWARGS = dict(input_dim=1280, num_classes=1, fuse="cls", field="roi", pos_mode="ffn")
