"""a65 — mean-only ablation companion to a64 (removes the higher moments).

Identical to a64 in EVERY respect (same warm-started fibrosis axis v_hat, same
frozen prototype-grade scale C,S, same Linear(3->1) head, same init) except the
readout masks the 2nd and 3rd central moments to zero:

    u = [m1, 0, 0]
    y = clamp( Linear(3 -> 1)([m1, 0, 0]) , 0, 3 )  ==  clamp( a*m1 + b , 0, 3 )

Because `m1 = mean_i <f_i, v_hat>` standardised = a linear functional of the
mean-pooled feature, this readout is EXACTLY mean-pool + a linear head along the
warm-started axis — i.e. a53 / mean-projection. It removes the ONLY active
ingredient a64 adds: the variance (m2) and skew (m3) of the within-bag
projection distribution (the diffuseness shape).

a64 (moments) vs a65 (mean-only) is therefore the decisive, capacity-matched
test of the hypothesis: *do the higher moments of the fibrosis projection — the
diffuse-vs-focal SHAPE of the density distribution — beat its mean (mean-pool)?*
If a64 > a65 across seeds, diffuseness-as-a-statistic is a real grading-aligned
contribution beyond mean-pooling. If not, the honest finding is that it reduces
to mean-pool along this axis.

Trainable params identical to a64: v_hat 1280 + Linear(3,1) = 1284 (the head's
m2,m3 weights simply receive no signal because their inputs are masked to 0).
"""
from .a64_fibrosis_moment_field import Model

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
