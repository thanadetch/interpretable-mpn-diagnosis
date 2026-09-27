"""a22 — a17 stack with alpha_init = 5e-3 (H12 ablation, larger initial coverage).

Ablation companion to a21_a17_alpha_init_5e4.py. Identical wiring but
alpha_init=5e-3 (5x larger than a17). Tests whether *pushing harder*
on coverage pulls val even higher (and test even lower).

Together a21 (5e-4) + a17 (1e-3, already on leaderboard) + a22 (5e-3)
form a 3-point alpha sweep at seed=2.

Kill criterion (family-level): if neither a21 nor a22 beats both a17
on val AND a18 on test, the alpha-sweep family is dead at seed=2.

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
    alpha_init=5e-3,
    c0_init=6.324555,
    use_coverage=True,
)

