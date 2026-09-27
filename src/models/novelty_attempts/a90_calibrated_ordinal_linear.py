"""a90 — Plain Linear head on the diffuse-pooled scalar (ablation for a89).

Identical to a89 in EVERY respect (same baseline bottleneck encoder, same
diffuse mean-pool z = mean_i h_i, same warm-started scalar projection
s = <z, w> + c along the train fibrosis axis) except use_ordinal=False: the
CALIBRATED MONOTONE CUMULATIVE-SIGMOID ordinal readout is removed and the score
s is returned directly as the grade. The forward then reduces to

    h_i = Dropout(ReLU(Linear(1280->128) f_i))
    z   = mean_i h_i                 # plain DIFFUSE pool
    y   = <z, w> + c                 # plain Linear(z -> 1) regression head (RAW)

i.e. exactly the diffuse mean-pool + free linear head with NO ordinal
calibration, NO monotone boundaries, NO learnable temperature.

This removes EXACTLY the active ingredient of a89 — the calibrated ordinal
readout y = sum_k sigmoid((s - b_k)/t) — and nothing else (the encoder, the
diffuse pool, and the score projection are byte-identical because we import
a89's Model and only flip the documented `use_ordinal` flag; the unused ordinal
parameters b1_raw/delta_2_raw/delta_3_raw/t_raw still exist but never touch the
forward path, so they do not affect the output or its gradient).

a89 (calibrated ordinal) vs a90 (linear head) isolates: does an
ordinal-calibrated readout reduce the val<->test variance vs a free linear head
on the same diffuse-pooled scalar? If a89 ≈ a90 on PAIRED cross-fold (Δ≈0), the
calibrated readout buys no robustness here and the cumulative-link family is
another honest tie (consistent with a47).
"""
from .a89_calibrated_ordinal import Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    use_ordinal=False,       # ablation = NO ordinal calibration => plain Linear(z->1)
    b1_init=0.5,             # inert when use_ordinal=False (ordinal params unused)
    delta_init=0.5413248,    # inert
    temp_init=1.0,           # inert
    temp_eps=1e-3,           # inert
    warm_start=True,
    prototype_path=None,
    clamp_output=False,
)
