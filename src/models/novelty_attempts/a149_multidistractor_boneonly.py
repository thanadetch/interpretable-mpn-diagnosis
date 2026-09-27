"""a149 — ablation of a148: use_adipose=False (bone-only avoidance) = the a145 flagship.
a148 (bone+adipose) vs a149 (bone-only) isolates the effect of ADDING fat-avoidance:
faithfulness gain (lower pool↔adipose correlation) vs grade cost (adipose corr +0.60 w/ grade).
Reads data_distract bag = [feat | fibrosis | bone | adipose].
"""
from __future__ import annotations
from .a148_multidistractor_density import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, use_adipose=False, warm_start=True)
