"""a10 — coverage-aware attention re-weight, HARD threshold (H1 ablation).

Ablation companion to a09_coverage_attn_weight.py. Identical wiring
except `beta` is FIXED at 0.05, turning the per-patch coverage prior
    c_i = sigmoid((||h_i|| - tau) / beta)
into a near-step function around tau. Gradients to tau still flow (the
sigmoid slope at the threshold is finite) but the smooth transition
zone in which patches partially contribute is essentially gone --
patches are either "in" (c_i ~ 1) or "out" (c_i ~ 0).

Active ingredient under test: the SOFT transition zone of a09 (beta
learnable, full sigmoid slope). If a09 beats a10, smoothness matters;
if both fail, the per-patch coverage-prior-in-attention family is dead
under simple + virchow2.

Kill criterion (family-level): if both a09 AND a10 have val_qwk < 0.82
at seed=2, abandon H1 and add the family to §4 DE.

Param count: baseline 197,250 + 2 scalars (tau, alpha_raw) = 197,252
    (beta is fixed -- registered as a buffer, not a parameter).
"""
from .a09_coverage_attn_weight import Model as _Model

Model = _Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    tau_init=1.0,
    alpha_init=1e-3,
    beta_learnable=False,
    beta_fixed=0.05,
)

