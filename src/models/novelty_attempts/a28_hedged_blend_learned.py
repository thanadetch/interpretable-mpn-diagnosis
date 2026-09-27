"""a28 - hedged prediction-level blend with LEARNED alpha (H14 ablation companion to a27).

Identical architecture to `a27_hedged_blend_fixed.py` but the blend
coefficient is `alpha = sigmoid(gamma)` with `gamma` a single
learnable scalar (initialised so alpha = 0.5 at training start).

Active ingredient being tested: the *blending itself*, by letting
alpha drift away from 0.5 if one branch is strictly better.

Expected outcomes and what they mean (vs a27 fixed=0.5):
    - a28 val ~ a27, alpha drifts to ~0.5 -> the 50/50 blend really
      is the active ingredient; the hedged prior is correct.
    - a28 val > a27, alpha drifts to ~1.0 -> a17 is the right model,
      the mean branch is dead weight, hedging hurts; do not retry.
    - a28 val > a27, alpha drifts to ~0.0 -> mean-pool alone beats
      a17; a17's mechanism stack is a net negative on this dataset.
    - a28 val < a27 -> the learned alpha overfits the val signal that
      drove training, supporting a fixed 50/50 (a27) over learned.

Kill criterion: shared with a27 (family-level). Abandon H14 if BOTH
a27 and a28 val_qwk < a17's 0.7970 at seed=2.

Param count: a27 + 1 (gamma) = 198,536 trainable.
"""
from .a27_hedged_blend_fixed import Model as _Model

Model = _Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    learned_alpha=True,     # <-- the only flipped flag vs a27
    alpha_init=0.5,
    tau_init=1.0,
    beta_init=1.0,
    alpha17_init=1e-3,
    c0_init=6.324555,
    use_coverage=True,
)

