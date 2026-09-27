"""a73 — Pairwise PRODUCT kernel (ablation companion to a72).

Identical to a72 but kernel='product': the bag-level pairwise statistic uses
phi = s_i * s_j instead of phi = |s_i - s_j|. For the product kernel the
relational term degenerates to a pure function of the mean,

    D = mean_{i,j} s_i s_j = (mean_i s_i)^2 = mean^2,

so the bag descriptor [mean_s, D] = [mean_s, mean_s^2] carries NO information
beyond the single mean-pool scalar, and clamp(Linear([mean, mean^2])) is a fixed
quadratic in mean_s with no diffuseness degree of freedom. This removes EXACTLY
the active ingredient of a72 — the genuinely 2nd-order density-CONTRAST
(mean-absolute-pairwise-difference / Gini spread) that is NOT a function of the
mean — while holding the warm-started fibrosis axis, the linear per-patch
projection, and the 2-in linear head fixed.

a72 (spread) vs a73 (product) isolates: does the relational within-bag
density-CONTRAST add over the mean projection (a53 / mean-pool along the same
axis)? If not, a72 collapses to mean-pool — the recurring outcome this session,
recorded honestly.
"""
from .a72_fibrosis_pairwise_spread import Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    kernel="product",        # ablation = s_i*s_j -> D = mean^2 (pure function of the mean)
    warm_start=True,
    prototype_path=None,     # None -> data/prototypes_virchow2_reti_train_seed2.pt
    random_seed=2,
    clamp_output=True,
)
