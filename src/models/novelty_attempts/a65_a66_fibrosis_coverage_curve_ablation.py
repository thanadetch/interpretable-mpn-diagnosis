"""a67 — fibrosis-coverage SINGLE-threshold MIL (ablation companion to a66).

Identical SHARED Model as a66 but with n_thresholds=1 (K=1): the monotone
ladder collapses to ONE learned soft threshold, so the curve degenerates to a
SINGLE calibrated mean-coverage scalar read by Linear(1,1):

    s_i  = <f_i, w>
    c_0  = (1/N) sum_i sigmoid((s_i - tau_0) / beta)      # ONE coverage scalar
    y    = clamp(Linear(1 -> 1)(c_0), 0, 3)

This is EXACTLY the a52 mechanism, rebuilt INSIDE the same Model (zero code
drift — only n_thresholds differs), so a66 vs a67 isolates exactly the active
ingredient and nothing else.

a66 (K=7 curve) vs a67 (K=1 single coverage) tests:
  "Does reading the coverage CURVE (the survival-function SHAPE / local slope =
   diffuse-vs-focal density) at MULTIPLE staged thresholds beat reading coverage
   at ONE threshold (a52, which already LOST to its mean-pool ablation a53)?"

If a66 ~= a67 across seeds, the honest negative is that the curve adds nothing
over a single extent count — a single soft coverage is first-order-dominated by
mean-pool, and the per-band slopes carried no seed-stable grade signal.

Param count (input_dim=1280, K=1): w 1280 + tau0_raw 1 + delta_raw 0
  + beta_raw 1 + head Linear(1,1) 2 = 1284.
"""
from .a64_a66_fibrosis_coverage_curve import Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    n_thresholds=1,         # ablation = collapse the ladder to ONE threshold (a52)
    init_scale=0.1,
    tau0_init=0.0,
    delta_init=0.5,
    beta_init=1.0,
    warm_start=True,
    prototype_path=None,    # None -> data/prototypes_virchow2_reti_train_seed2.pt
    random_seed=2,
    clamp_output=True,
)
