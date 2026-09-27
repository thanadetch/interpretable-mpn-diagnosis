"""a65 — Fibrosis-projection MEAN-only readout (ablation companion to a64).

Identical to a64 except moment_mode='mean': the head receives z = [m1, 0, 0],
so the prediction depends ONLY on the mean projection along the same frozen,
warm-started fibrosis axis:

    s_i = <f_i, v>
    y   = clamp( w * mean_i s_i + b , 0, 3 )

Because mean_i <f_i, v> = <mean_i f_i, v>, this is exactly mean-pool of the
features followed by a linear head (== a53). It removes EXACTLY the active
ingredient that a64 adds: the higher central moments of the projection
distribution (spread m2 and skew g1).

a64 (moments) vs a65 (mean) is therefore the decisive, clean test of the
distributional hypothesis:
    "Do the SPREAD and SKEW of the per-patch fibrosis projection — i.e. the
     shape of the diffuse-density distribution (diffuse vs focal) — beat its
     MEAN density alone?"
If a64 > a65 on val across seeds, the higher moments are a real, grading-
aligned contribution beyond mean-pooling. If not, the honest finding is that
this reduces to mean-pooling (and, unlike the a56->a57 collapse, here that
collapse is by DESIGN of the ablation, not an unintended consequence of a
per-patch nonlinearity).

No sigmoid anywhere (DE11-13 safe). Same frozen axis, same ~5 trainable
params (head 4 + feat_scale 1; the spread/skew weights receive no useful
gradient since their inputs are forced to 0).
"""
from .a64_fibrosis_projection_moments import Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    moment_mode="mean",      # ablation = mean projection ~ mean-pool + linear head
    warm_start=True,
    freeze_axis=True,
    feat_scale_init=0.3,
    head_w_init=1.0,
    head_b_init=1.5,
    prototype_path=None,
    random_seed=2,
    eps=1e-6,
    clamp_output=True,
)
