"""a20 — a17 stack with input-feature dropout DISABLED (H11 ablation, ~a17 positive control).

Ablation companion to a19_a17_input_dropout.py. Identical wiring but
input_dropout_p=0.0 so the input dropout layer is a no-op — the model
reduces to a17.

Active ingredient under test: input-feature dropout p=0.2 (a19) vs
no input dropout (a20 ≡ a17 positive control).

Positive control: a20 should reproduce a17 val ~ 0.7970 / test ~
0.9530. If a20 diverges, there is an architecture/RNG discrepancy.

Kill criterion (family-level): if a19 val_qwk < 0.78 OR test_qwk <
0.94 at seed=2, abandon H11 and add family to section 4 DE.

Param count: same as a17 = 197,254.
"""
from .a19_a17_input_dropout import Model as _Model

Model = _Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    tau_init=1.0,
    beta_init=1.0,
    alpha_init=1e-3,
    c0_init=6.324555,
    use_coverage=True,
    input_dropout_p=0.0,
)

