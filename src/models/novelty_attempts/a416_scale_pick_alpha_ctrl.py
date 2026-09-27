"""a416 — CONTROL for a415: same four rows of five logits, bands assigned to ROIs at random.

The model can still commit four different orders; it just cannot tie them to the real
magnification. Separates "the scale matters" from "having a choice of orders matters".
"""
from .scale_pick_alpha import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, shuffle=True, seed=0)
