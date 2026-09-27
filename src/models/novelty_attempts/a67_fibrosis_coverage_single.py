"""a67 — single-threshold COVERAGE ablation (companion to a66).

Identical to a66 except `n_thresholds=1` (K=1): the staged threshold LADDER
collapses to ONE learned soft threshold, so the readout is a single calibrated
mean-coverage scalar read by a Linear(1,1):

    s_i = <f_i, w>                         # warm-started fibrosis axis (same as a66)
    c   = mean_i sigmoid((s_i - tau)/beta) # single soft fraction-of-bag-above-threshold
    y   = clamp(Linear(1->1)(c), 0, 3)

This is EXACTLY the a52 mechanism (single soft-threshold coverage on the
warm-started fibrosis axis), reconstructed inside the shared a66 Model so there
is zero code drift between main and ablation — only `n_thresholds` differs.

a66 (K thresholds, the survival CURVE) vs a67 (K=1, one coverage) therefore
isolates the SINGLE active ingredient a66 adds:

    "Does reading the COVERAGE CURVE — the soft survival function of the
     fibrosis projection at multiple staged thresholds, whose local slope
     encodes diffuse-vs-focal density — beat reading coverage at ONE
     threshold (a52, which already lost to mean-pool a53)?"

Recall the refuted single-threshold result: a52 val_qwk 0.7675 < its
mean-projection ablation a53 0.7725. If a66 > a67 on a multi-seed
{0,1,2,3,42} val median, the survival-curve SHAPE is a real, grading-aligned
contribution that a single coverage count (and hence mean-pool) cannot
represent. If a66 ~= a67, the honest finding is that the extra thresholds add
nothing beyond one extent count.

Param count (input_dim=1280, n_thresholds=1):
    w 1280 + tau0_raw 1 + beta_raw 1 + head Linear(1,1) 2 = 1284
    (delta_raw is None at K=1, so the ladder increments contribute 0 params).
"""
from __future__ import annotations

from models.novelty_attempts.a66_fibrosis_coverage_curve import Model  # re-export

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    n_thresholds=1,         # ablation: single-threshold coverage (== a52)
    init_scale=0.1,
    tau0_init=0.0,          # center the single threshold (a52's tau_init)
    delta_init=0.5413248,   # unused at K=1
    beta_init=0.0,
    warm_start=True,
    prototype_path=None,
    random_seed=2,
    clamp_output=True,
)
