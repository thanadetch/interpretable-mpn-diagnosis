"""a26 - random-axis projection MIL (H3 ablation companion to a25).

Identical architecture to a25_fibrosis_axis_projection.py but the
frozen `axis` is a *random unit vector* instead of the prototype-
derived (c_G3 - c_G0) direction. Same trainable param count
(164,098), same forward graph, same tau scalar, same softmax-over-
scalar-score pooling.

Active ingredient being removed: the *direction* in which we project
patches. If a25 >> a26 on val_qwk, the grade-prototype direction is
the active ingredient (the *labels* used to compute it carry the
signal). If a25 ~= a26, the active ingredient is the pooling shape
itself (softmax-weighted mean over a scalar-projection attention),
not the direction - which would be a much weaker / less defensible
H3 claim.

Random seed is fixed (=2) so the ablation is bit-for-bit reproducible.

Kill criterion: shared with a25 (family-level) - abandon H3 if BOTH
a25 and a26 val_qwk < 0.78 at seed=2.

Param count: identical to a25 (164,098 trainable).
"""
from .a25_fibrosis_axis_projection import Model as _Model

Model = _Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    tau_init=1.0,
    prototype_path=None,    # ignored when random_axis=True
    random_axis=True,       # <-- the only flipped flag vs a25
    random_seed=2,
)

