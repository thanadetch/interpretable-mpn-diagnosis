"""a75 — Mean-only distribution match (ablation companion to a74).

Identical to a74 but match_mode='mean': each per-grade reference is collapsed to
its MEAN R_g_mean and the bag is collapsed to its mean projection mean_i s_i, so

    d_g = (mean_i <f_i, v> - R_g_mean)^2
    y   = clamp( sum_g g * softmax(-d_g / tau) , 0, 3 )

i.e. a NEAREST-PROTOTYPE-MEAN 1-D classifier on the bag mean — mean-pool along
the warm-started fibrosis axis, then soft-assign to the closest grade mean.

This removes EXACTLY the active ingredient of a74 — matching the bag's full
quantile DISTRIBUTION to per-grade reference DISTRIBUTIONS. a75 keeps the same
axis v, the same references file, the same soft-argmin readout, and the same
learnable temperature tau; it only drops the quantile vector down to its mean.

a74 (dist) vs a75 (mean) isolates: does matching the full per-grade DENSITY
DISTRIBUTION beat matching just the per-grade MEAN? (If not, a74 collapses to
the mean-only baseline — the recurring outcome this design family must beat.)

Param count is identical to a74 (input_dim=1280): v 1280 + log_tau 1 = 1281;
the unused R quantile buffer is non-trainable.
"""
from .a74_fibrosis_dist_match import Model, _DEFAULT_LEVELS

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    match_mode="mean",       # ablation = match only the bag MEAN to each grade's ref mean
    levels=_DEFAULT_LEVELS,
    warm_start=True,
    prototype_path=None,     # None -> data/prototypes_virchow2_reti_train_seed2.pt
    refs_path=None,          # None -> data/fibrosis_refs_seed2.pt (TRAIN-only, seed=2)
    rank_sigma=1.0,
    sel_tau=0.02,
    tau_init=1.0,
    random_seed=2,
    clamp_output=True,
)
