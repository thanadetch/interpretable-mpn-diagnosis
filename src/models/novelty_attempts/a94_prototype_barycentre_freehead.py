"""a94 — Ablation for a93: SAME diffuse score, FREE affine head (no anchors).

This removes EXACTLY the active ingredient of a93. a93 maps the single diffuse
fibrosis score s = <mean_i f_i, v> to a grade via the FROZEN-anchor barycentre
(grade levels PINNED at the train prototype coordinates a_g; only a temperature
and a shrinkage scalar are learnable — NO free slope/intercept). a94 keeps the
diffuse pool and the projection v BYTE-IDENTICAL but replaces the barycentre
readout with a FREE affine head y = w_s * s + c_s (slope and intercept learnable).

a93 (frozen-anchor barycentre) vs a94 (free affine head) therefore isolates the
single question: does PINNING the score->grade map to the train prototype anchors
(removing the free slope/intercept that a tiny fold can overfit) reduce the
val<->test gap vs a free affine readout on the same diffuse score? Everything
else — diffuse mean pool, learnable warm-started direction v, determinism,
permutation- & bag-size-invariance — is identical.

Implemented by importing a93's Model and flipping use_anchors=False, so the two
modules cannot drift apart.
"""
from __future__ import annotations

from .a93_prototype_barycentre import Model  # re-export identical class

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    use_anchors=False,       # a94 ablation = FREE affine head on the diffuse score
    warm_start=True,
    prototype_path=None,
    tau_init=4.0,
    tau_eps=1e-3,
    rho_max=0.5,
    rho_init=0.1,
    random_seed=2,
    clamp_output=False,
)

__all__ = ["Model", "KWARGS"]
