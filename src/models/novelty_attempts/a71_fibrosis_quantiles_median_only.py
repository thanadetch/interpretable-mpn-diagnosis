"""a71 — Median-only soft quantile (ablation companion to a70).

Identical to a70 but quantile_levels=(0.5,): the readout keeps ONLY the soft
median of the fibrosis projection,

    y = clamp( Linear(1 -> 1)(q_0.5) , 0, 3 )

which is a ROBUST CENTRAL DENSITY of the projection {s_i} (~ a soft median of
mean-pool along the same axis). This removes EXACTLY the active ingredient of
a70 — the lower/upper TAIL quantiles q_0.1 and q_0.9 — leaving everything else
(the warm-started axis v, the soft-rank machinery, the temperatures) fixed.

a70 (3 quantiles) vs a71 (median only) isolates: do the TAIL quantiles add over
the CENTRAL density? If a71 matches a70 on val, the tails carry no extra grade
signal beyond the median of the projection (the honest null this design tests).

Param count (input_dim=1280): v 1280 + Linear(1,1) weight 1 + bias 1 = 1282.
"""
from .a70_fibrosis_quantiles import Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    quantile_levels=(0.5,),   # ablation = soft median only (drops the tail quantiles)
    tau=0.05,
    rank_tau=0.05,
    warm_start=True,
    prototype_path=None,      # None -> data/prototypes_virchow2_reti_train_seed2.pt
    random_seed=2,
    clamp_output=True,
)
