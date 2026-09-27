"""a96 — ABLATION of a95 (== a94 in the design): free-affine readout instead of anchors.

This imports a95's Model UNCHANGED and flips exactly ONE documented flag:
    use_anchors = False.

What stays BYTE-IDENTICAL to a95:
    - the DIFFUSE pool z = mean_i f_i (every patch equal),
    - the warm-started projection direction v (train axis, seed=2),
    - the scalar diffuse fibrosis score s = <z, v>,
    - permutation-invariance, bag-size-invariance, determinism at eval,
    - forward reads ONLY 'features'.

What changes (the ablated active ingredient, and NOTHING else):
    The frozen-anchor BARYCENTRE readout y = sum_g g * softmax(-(s-a_g')^2/tau^2)_g
    is replaced by a FREE AFFINE head y = w_s * s + c_s (slope + intercept both
    learnable). It is warm-started so the anchor span [a_min, a_max] maps to [0, 3]
    at init, i.e. a96 starts numerically close to a95 but is then free to slide its
    slope/intercept. This removes EXACTLY the anchor-pinned, slope/intercept-free
    calibration — the active ingredient — and nothing else.

a95 vs a96 on PAIRED cross-fold Δ answers precisely: does pinning the score->grade
map to the train prototype anchors reduce the val<->test gap vs a free affine readout
on the same diffuse score? Kill a95 if a95 <= a96 paired.
"""
from __future__ import annotations

from .a95_a93_prototype_barycentre import Model  # noqa: F401  (re-exported drop-in)

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    num_grades=4,
    rho_max=0.5,
    tau_init=4.0,
    use_anchors=False,       # ABLATION: free affine readout (w_s*s + c_s) instead of anchors
    warm_start=True,
    prototype_path=None,     # None -> data/prototypes_virchow2_reti_train_seed2.pt
    clamp_output=False,      # RAW logits out; trainer rounds+clips at eval
)
