"""a21 — a17 stack with alpha_init = 5e-4 (H12 alpha sensitivity, smaller initial coverage).

Hypothesis (H12_alpha_init_sweep): the H10 stack (a17) hit val 0.7970 /
test 0.9530 with alpha_init=1e-3. The coverage prior helps val but
slightly hurts test (a18 alpha=0: val 0.7868 / test 0.9596). Because
val_qwk peaks at epoch 5-10, the optimizer barely moves alpha from
its init in time — so alpha_init effectively SETS where on the
val<->test tradeoff curve the model sits.

Goal: sweep alpha_init smaller (a21 at 5e-4 = halfway between a18's 0
and a17's 1e-3) and see if val stays above a18's 0.7868 while test
recovers toward a18's 0.9596.

Mechanism: identical to a17, only alpha_init changes. All other
KWARGS unchanged.

Kill criterion: abandon if a21 val_qwk < 0.78 at seed=2.
Param count: same as a17 = 197,254.
"""
from .a17_coverage_length_norm import Model as _Model

Model = _Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    tau_init=1.0,
    beta_init=1.0,
    alpha_init=5e-4,
    c0_init=6.324555,
    use_coverage=True,
)

