"""a92 — Standard diffuse mean-pool baseline (ablation companion to a91).

Identical to a91 in EVERY respect (same diffuse mean-pool encoder, same linear
head, same warm-start along the train fibrosis axis — we import a91's Model and
only flip the documented regularisation flags) except the regularisation budget:

    dropout    = 0.5   (baseline, vs a91's heavier 0.7)
    noise_frac = 0.0   (NO train-time input noise, vs a91's 0.1)

so the forward reduces to the STANDARD diffuse mean-pool baseline:

    h_i = Dropout(p=0.5)(ReLU(Linear(1280->128) f_i))
    z   = mean_i h_i                 # plain DIFFUSE pool
    y   = Linear(128 -> 1)(z)        # raw grade logit

This removes EXACTLY the active ingredient of a91 — the extra regularisation
(train-time Gaussian input noise + heavier dropout) — and NOTHING else (the
encoder, head and warm-start are byte-identical because we import a91's Model).

a91 vs a92 isolates: does heavier regularisation REDUCE the val<->test gap /
improve PAIRED cross-fold generalisation over the plain diffuse pool? If not,
a91 collapses to this baseline (the honest null this session keeps surfacing).
Note a92 is already deterministic at eval; the determinism check matters for
a91 (its randomness must be gated OFF under model.eval()).
"""
from .a91_heavyreg_diffuse import Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,           # ablation = baseline dropout
    noise_frac=0.0,        # ablation = NO train-time input noise
    warm_start=True,
    prototype_path=None,
    clamp_output=False,
)
