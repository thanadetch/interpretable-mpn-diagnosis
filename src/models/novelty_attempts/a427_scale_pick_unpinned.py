"""a427 — a425 with the pinning REMOVED: the unmeasurable ROIs get a fourth row and learn an
order of their own.

Single-change ablation against a425. Same four orders {1.00, 1.25, 1.50, 2.00}, same
straight-through Gumbel pick, same everything — except that the 47% of ROIs whose scale bar
cannot be read now compete for the same gradient instead of sitting at 1.5.

a425 found its zoomed-out band choosing 1.25 in 12 of 18 runs while its shuffled control never
did (Fisher p = 1.1e-05). If that separation survives here, the pinning was not what produced it;
if it collapses, the pinning is doing the work. Its own control is a428.
"""
from .scale_pick_pinned import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, shuffle=False, pin=False)
