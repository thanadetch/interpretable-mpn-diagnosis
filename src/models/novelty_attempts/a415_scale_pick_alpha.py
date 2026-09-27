"""a415 — each magnification band PICKS one order from {1.00, 1.25, 1.50, 1.75, 2.00}.

Hard discrete choice per band (straight-through Gumbel in training, argmax at eval), so exactly
one alpha reaches the pooling — not a blend. See `scale_pick_alpha.py`. Control is a416.
"""
from .scale_pick_alpha import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, shuffle=False)
