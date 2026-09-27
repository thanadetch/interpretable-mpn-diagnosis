"""a84 — Plain diffuse pool (ablation companion to a83).

Identical to a83 in EVERY respect (same bottleneck encoder, same linear head,
same warm-start along the train fibrosis axis) except smooth_steps=0 (K=0): the
multi-step feature-graph smoothing is removed. The forward then reduces to

    h_i = Dropout(ReLU(Linear(1280->128) f_i))
    z   = mean_i h_i                 # plain DIFFUSE pool, no smoothing
    y   = Linear(128 -> 1)(z)        # raw grade logit

which is exactly the diffuse mean-pool + linear head with NO contiguity prior.
This removes EXACTLY the active ingredient of a83 — the K=2 learnable graph
smoothing h <- (1-a)*h + a*(A@h) — and nothing else (the encoder and head are
byte-identical because we import a83's Model and only flip the documented flag).

a83 (K=2) vs a84 (K=0) isolates: does iterated graph smoothing of the
contiguous fibre meshwork beat the plain diffuse pool on val? If not, a83
collapses to the diffuse pool (the honest null this session keeps surfacing).
"""
from .a83_graph_smooth_pool import Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    smooth_steps=0,        # ablation = NO graph smoothing => plain diffuse pool
    mix_init=0.5,          # inert when smooth_steps=0 (mix_logit unused in forward)
    warm_start=True,
    prototype_path=None,
    clamp_output=False,
)
