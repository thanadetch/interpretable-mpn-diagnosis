"""a23 — a17 stack with hidden_dim=64 (H13 capacity sweep, smaller bottleneck).

Hypothesis (H13_hidden_dim_sweep): a17 (hidden_dim=128) has 197,254
trainable params yet trains to train_qwk → 0.998 in ~5 epochs on
857 train bags. Halving the bottleneck width to 64 reduces total
params to ~90k and may force a more compact representation that
overfits less. Combined with the H10 mechanism stack this could
push val above 0.80 while preserving test.

Mechanism: identical to a17, only hidden_dim changes.

Kill criterion: abandon if a23 val_qwk < 0.78 at seed=2.

Approximate param counts:
    hidden_dim=64  → ~90,438 (this module)
    hidden_dim=128 → 197,254 (a17 baseline)
    hidden_dim=256 → ~537,k  (a24 companion)
"""
from .a17_coverage_length_norm import Model as _Model

Model = _Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=64,
    dropout=0.5,
    tau_init=1.0,
    beta_init=1.0,
    alpha_init=1e-3,
    c0_init=6.324555,
    use_coverage=True,
)

