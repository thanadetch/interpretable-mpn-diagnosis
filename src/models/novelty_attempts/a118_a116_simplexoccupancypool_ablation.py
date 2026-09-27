"""a118 — SimplexOccupancyPool ABLATION: linear-readout-of-the-mean.

Imports a116's Model and flips EXACTLY ONE flag in KWARGS: readout="meanproj".
Everything else — the FROZEN rank-3 grade subspace basis P, the train-only
prototype coords A, the coord scale v0, the frozen in-subspace fibrosis axis —
is byte-identical (SAME Model class, SAME KWARGS except the one flag).

What the flip turns OFF (the ablated active ingredient = the 4-way OCCUPANCY SHAPE):
    main a116 (occupancy): q   = mean_i softmax_k(-d-weighted ||z_i - A_k||^2 / tau)
                           y   = sum_k g_k q_k        # nonlinear functional of the
                                                      # full within-bag occupancy
    abl  a118 (meanproj):  s_i = <z_i, axis_coord> / v0
                           y   = w0 + w1 * mean_i s_i # linear-readout-of-the-mean
                                                      # along ONE frozen axis

a118 is EXACTLY the "single linear-readout-of-the-mean along the fibrosis
direction" that the HARD-CONSTRAINED-CAPACITY lens says a116 must beat: it reads
only the bag CENTROID's coordinate on one axis and is provably blind to the
occupancy SHAPE (two bags with the same centroid but different grade-mixtures
read identically). a116 vs a118 isolates the single claim of this candidate:
"does the diffuse 4-way grade-occupancy distribution add over the 1-D centroid?"
If a116 <= a118 on the paired cross-fold delta, the occupancy shape adds nothing
over mean-projection (the lens's null hypothesis) and a116 is dead.

a118 has 2 LEARNABLE free params (w0, w1; axis_coord + P + A frozen, and the
occupancy DOF rho/theta/bias/log_span/log_tau are frozen requires_grad=False in
meanproj mode) — a clean, minimal mean-projection control that sits at the
UNDERFIT floor on purpose, so the comparison is "does the richer-but-still-hard-
constrained occupancy escape the underfit floor WITHOUT entering the val-overfit
regime?".
"""
from .a116_a116_simplexoccupancypool import Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    subspace_rank=3,
    readout="meanproj",      # ABLATION: occupancy OFF -> linear-readout-of-the-mean
    tau_init=1.0,
    span_init=3.0,
    prototype_path=None,
)
