"""a67 — mean-only ablation companion to a66 (a64_fibrosis_moment_field).

Identical to a66 in EVERY respect (same warm-started fibrosis axis v_hat, same
frozen prototype-grade scale C,S, same Linear(3->1) head, same init, same 1284
trainable params) except the readout masks the 2nd and 3rd central moments to
zero:

    u = [m1, 0, 0]
    y = clamp( Linear(3 -> 1)([m1, 0, 0]) , 0, 3 )  ==  clamp( a*m1 + b , 0, 3 )

Because `m1 = mean_i <f_i, v_hat>` standardised is a linear functional of the
mean-pooled feature, this readout is EXACTLY mean-pool + a linear head along the
warm-started axis — i.e. a53 / mean-projection. It removes the ONLY active
ingredient a66 adds: the variance (m2) and skew (m3) of the within-bag
projection distribution (the diffuseness shape).

a66 (moments) vs a67 (mean-only) is therefore the decisive, capacity-matched
test of the hypothesis: *do the higher moments of the fibrosis projection — the
diffuse-vs-focal SHAPE of the density distribution — beat its mean (mean-pool)?*
a66 even STARTS at this readout (its head's m2,m3 weights init to 0), so the
ablation is the guaranteed safe floor with zero capacity confound.

A secondary control (warm_start=False) tests whether the grounded init is what
buys the cross-seed robustness; left at the warm-started default here so the
isolated variable is exactly the higher moments.

Trainable params identical to a66: v_hat 1280 + Linear(3,1) = 1284 (the head's
m2,m3 weights simply receive no signal because their inputs are masked to 0).
"""
from .a66_a64_fibrosis_moment_field import Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    moment_mode="mean",      # ablation = drop m2,m3 -> mean-pool along the axis
    warm_start=True,
    prototype_path=None,
    random_seed=2,
    eps=1e-4,
    clamp_output=True,
)
