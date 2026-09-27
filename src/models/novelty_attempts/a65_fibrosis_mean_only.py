"""a65 — fibrosis MEAN-ONLY ablation (companion to a64).

Identical to a64 except `use_spread=False`: the readout drops the SPREAD
(standard deviation of the projection) and keeps ONLY the mean:

    s_i = <f_i, w>
    m1  = mean_i s_i
    y   = clamp(Linear(1 -> 1)(m1), 0, 3)

Because mean_i <f_i, w> = <mean_i f_i, w>, this readout is a *linear
functional of the mean-pooled feature* — i.e. exactly mean-pool + a linear
head (the same object as a53_fibrosis_meanproj). It removes EXACTLY the one
active ingredient a64 adds — the second-order within-bag SPREAD term — and
nothing else (same warm-started axis w, same warm-start, same scale, same
clamp).

a64 (mean + spread) vs a65 (mean only) is therefore the decisive,
mechanism-isolating test of the lens:

    "Do the higher-order (dispersion) statistics of the fibrosis projection
     beat its mean alone — i.e. does encoding DIFFUSE-vs-FOCAL density as a
     spread statistic add over plain mean density?"

If a64 > a65 across seeds, the second-order dispersion term is a real,
grading-aligned contribution beyond mean-pooling. If a64 ~ a65, the honest
finding is that the spread of the projection is inert on Virchow2 and the
design reduces to mean-pooling (recorded as such — no one-seed lottery).

No sigmoid anywhere (DE11-13 safe). Param count (input_dim=1280): w 1280 +
head Linear(1,1) weight 1 + bias 1 = 1282 trainable.
"""
from .a64_fibrosis_mean_spread import Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    use_spread=False,       # ablation = mean-only == mean-pool + linear head (~ a53)
    init_scale=1.0,
    warm_start=True,
    prototype_path=None,
    random_seed=2,
    var_eps=1e-6,
    clamp_output=True,
)
