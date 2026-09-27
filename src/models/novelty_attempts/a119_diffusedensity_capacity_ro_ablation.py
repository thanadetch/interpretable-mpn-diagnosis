"""a119 — ablation of a117 (DDCRF grading coordinate): flip use_fibrosis_axis OFF.

Imports a117's `Model` unchanged and sets `use_fibrosis_axis=False` in KWARGS.

ACTIVE INGREDIENT being flipped
-------------------------------
a117's readout reads the DDCRF grading coordinate — the bag-wide DIFFUSE
FIBRE-DIRECTION projection density

    d(bag) = (1/N) sum_i <h_i, v> / ||v||      (the +0.882-Spearman coordinate),

i.e. "how much of the diffuse fibre signal is there, bag-wide", along the
train-only fibrosis axis v = mean(G2/G3) - mean(G0/G1).

a119 replaces that coordinate with the grade-UNINFORMATIVE negative control —
the bag-mean feature-NORM

    d(bag) = || mean_i h_i ||                    (the ||h||-style magnitude),

which the diagnostic showed has Spearman -0.005 vs grade (grade-uninformative,
explicitly on the kill-list as #5). Everything else is byte-identical: the same
2-parameter affine readout, the same diffuse mean-pool, the same perm/size-
invariance and determinism.

a117 (fibre-axis density, ON) vs a119 (bag-norm, OFF) therefore isolates EXACTLY
the active ingredient of the DDCRF coordinate: "is the readout reading the
diffuse FIBRE-DIRECTION density (the grade-informative +0.882 axis) and not just
bag-norm magnitude (the -0.005 negative control)?". The kill-list / diagnostic
predicts a117 >> a119, confirming the grading principle is carried by the
fibre-direction projection, never by ||h||.

forward uses ONLY 'features'. RAW scalar logit (trainer rounds/clips at eval).
"""
from __future__ import annotations

from .a117_diffusedensity_capacity_ro import Model  # noqa: F401

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    use_fibrosis_axis=False,   # OFF: bag-norm ||mean h|| negative control (||h||-style)
    axis_path=None,
)
