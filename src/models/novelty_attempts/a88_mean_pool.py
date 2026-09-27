"""a88 — Plain diffuse mean-pool (ablation companion to a87).

Identical to a87 in EVERY respect (same bottleneck encoder, same linear head,
same warm-start along the train fibrosis axis, same RAW-logit readout) except
robust=False: the outlier-robust soft-trimmed pooling is removed. The forward
then reduces to

    h_i = Dropout(ReLU(Linear(1280->128) f_i))
    z   = mean_i h_i                 # PLAIN diffuse mean-pool (uniform weights)
    y   = Linear(128 -> 1)(z)        # raw grade logit

which is exactly the diffuse mean-pool + linear head with NO outlier
robustness. This removes EXACTLY the active ingredient of a87 — the Weiszfeld
robust-centroid + soft-trimmed (residual-distance) weighting that down-weights
feature-space outliers — and nothing else (the encoder, head and warm-start are
byte-identical because we import a87's Model and only flip the documented flag).

a87 (robust trimmed) vs a88 (mean) isolates: does outlier-robust diffuse
pooling generalise better (smaller paired cross-fold variance / val<->test gap)
than the plain arithmetic mean? a88 is the strict no-robustness limit of a87
(equivalently a87 with trim temperature s -> inf gives uniform weights), so any
robustness benefit shows up purely as a87 > a88 on the paired cross-fold audit.
"""
from .a87_robust_trimmed_pool import Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    robust=False,          # ablation = NO robustness => plain diffuse mean-pool
    irls_steps=3,          # inert when robust=False (no IRLS / trimming in forward)
    scale_init=1.0,        # inert when robust=False
    trim_init=4.0,         # inert when robust=False
    warm_start=True,
    prototype_path=None,
    clamp_output=False,
)
