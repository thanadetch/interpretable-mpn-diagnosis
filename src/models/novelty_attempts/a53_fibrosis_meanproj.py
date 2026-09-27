"""a53 — fibrosis MEAN-projection ablation (companion to a52).

Identical to a52 except the readout is the MEAN projection along the (same,
warm-started) fibrosis direction instead of coverage:

    s_i = <f_i, w>
    y   = clamp(Linear(mean_i s_i), 0, 3)

Because `mean_i <f_i, w> = <mean_i f_i, w>`, this readout is a *linear
functional of the mean-pooled feature* — i.e. exactly mean-pool + a linear
head. It removes the only active ingredient that a52 adds: the nonlinear
**coverage** (soft fraction of fibre-positive patches).

a52 (coverage) vs a53 (mean) is therefore the decisive test of the
hypothesis: *does modelling fibrosis EXTENT (density/coverage — the clinical
grading definition) beat modelling fibrosis AVERAGE (mean-pooling)?*
If a52 > a53 on val (multi-seed), coverage is a real, grading-aligned
contribution beyond mean-pooling. If not, the honest finding is that this
reduces to mean-pooling.

Also serves as a DE11-13-safe control: a53 has no sigmoid anywhere.

Same param count minus tau (which is unused by the mean readout): w 1280 +
head 2 = 1282 trainable (tau exists but receives no gradient).
"""
from .a52_fibrosis_coverage import Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    readout="mean",         # ablation = linear mean projection ~ mean-pool
    init_scale=0.1,
    tau_init=0.0,
    warm_start=True,
    prototype_path=None,
    random_seed=2,
    clamp_output=True,
)
