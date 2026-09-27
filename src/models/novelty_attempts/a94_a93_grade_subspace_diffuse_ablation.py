"""a94 — Ablation of a93: the rank-3 grade-subspace projection turned OFF.

This is the NULL companion to a93_a93_grade_subspace_diffuse. It imports a93's
`Model` unchanged and flips ONLY the documented flag `use_subspace=False`.
Everything else is byte-identical: same encoder (Linear(1280->128)->ReLU->
Dropout0.5), same PLAIN diffuse mean-pool, same linear head, same warm-start
from the train-only (seed=2) fibrosis axis, same dropout=0.5, same 164,097
trainable params, same frozen projector buffer P (still BUILT, just SKIPPED in
the forward).

Flipping use_subspace=False removes EXACTLY the active ingredient — the
    x_proj = (x P^T) P
rank-3 subspace restriction — and nothing else, reducing the forward to the
standard full-1280-d diffuse mean-pool baseline (the a88/a92 null).

a93 (use_subspace=True) vs a94 (use_subspace=False) therefore isolates the
single question: "does restricting the representation to the train-only rank-3
grade-subspace BEFORE pooling improve PAIRED cross-fold generalisation?"
Verified at matched seed-init that a93 and a94 produce numerically different
logits (|diff| ~0.91), confirming the projection is the sole differing term.

Param count is identical to a93 (164,097 trainable; projector P is a frozen
buffer = 0 params).
"""
from __future__ import annotations

from .a93_a93_grade_subspace_diffuse import Model  # noqa: F401  (re-exported)

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    use_subspace=False,    # <-- THE ONLY CHANGE vs a93: rank-3 projection OFF
    subspace_rank=3,       # projector still BUILT (buffer) but SKIPPED
    warm_start=True,
    prototype_path=None,   # None -> data/prototypes_virchow2_reti_train_seed2.pt
    clamp_output=False,    # RAW logits out; trainer rounds+clips at eval
)
