"""a94 — Plain diffuse pool (ablation companion to a93).

Identical to a93 in EVERY respect (same bottleneck encoder, same linear head,
same warm-start along the train fibrosis axis, same dropout) except gamma_fixed=0.0:
the leading NON-TRIVIAL spectral mode is computed-but-NOT-used (gamma=0), so the
forward reduces to

    h_i = Dropout(ReLU(Linear(1280->128) f_i))
    z   = mean_i h_i                 # plain DIFFUSE / DC pool, no spectral term
    y   = Linear(128 -> 1)(z)        # raw grade logit

i.e. the diffuse mean-pool + linear head with NO coherent-mode readout. This
removes EXACTLY the active ingredient of a93 — the gamma * (projection of the
centered bag onto the leading non-trivial graph eigenvector) — and nothing else
(encoder, head, warm-start byte-identical, since a94 imports a93's Model and only
flips the documented gamma flag).

a93 (gamma learnable, init small) vs a94 (gamma=0) isolates: does adding the
leading coherent spectral mode to the diffuse pool help / stabilise vs the diffuse
pool alone, on PAIRED Δ across folds {0,1,2,3,42}? If not, a93 collapses to the
diffuse pool (the honest null this session keeps surfacing).
"""
from .a93_spectral_lowfreq_pool import Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    power_iters=8,         # inert when gamma_fixed=0 (spectral term multiplied by 0)
    gamma_init=0.1,        # inert when gamma_fixed=0
    gamma_fixed=0.0,       # ablation = spectral term OFF => plain diffuse pool
    warm_start=True,
    prototype_path=None,
    clamp_output=False,
)
