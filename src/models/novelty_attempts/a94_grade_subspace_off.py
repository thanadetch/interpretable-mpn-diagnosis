"""a94 — Ablation companion to a93: subspace projector OFF (identity / full 1280-d).

Identical to a93 in EVERY respect (same encoder Linear(1280->128), same diffuse
mean-pool, same linear head, same train-only warm-start) EXCEPT it flips
`use_subspace=False`, so the rank-3 grade-subspace projection
`x_proj = (x P^T) P` is NOT applied — the encoder sees the FULL raw 1280-d
feature. This removes EXACTLY the active ingredient of a93 (the low-rank
grade-subspace REPRESENTATION restriction) and NOTHING else; the bottleneck,
head and warm-start are byte-identical because we import a93's Model.

a93 (project onto rank-3 grade-subspace) vs a94 (full 1280-d) isolates:
    "does restricting the per-patch representation to the train-only rank-3
     between-grade-mean subspace BEFORE diffuse pooling reduce the val<->test
     gap / improve PAIRED cross-fold generalisation over the plain diffuse pool?"

With use_subspace=False and dropout=0.5, a94's forward reduces to the STANDARD
diffuse mean-pool baseline (same null a88/a92 surface):
    h_i = Dropout(0.5)(ReLU(Linear(1280->128) f_i))
    z   = mean_i h_i
    y   = Linear(128->1)(z)

Both are deterministic at eval, permutation- and bag-size-invariant. Same
trainable param count (164,097); the projector is a frozen buffer either way.
"""
from .a93_grade_subspace_diffuse import Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    subspace_rank=3,
    use_subspace=False,    # ABLATION: no subspace projection (full 1280-d)
    warm_start=True,
    prototype_path=None,
    clamp_output=False,
)
