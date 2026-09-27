"""a24 — a17 stack with hidden_dim=256 (H13 ablation, larger bottleneck).

Ablation companion to a23_a17_hidden64.py. Tests the upper end of the
capacity sweep: hidden_dim=256 = doubled bottleneck width vs a17.
~537k params. Prior dead-ends DE01 (DeepSet) and DE02 (Set Transformer)
both overfit when capacity went above ~200k under the legacy baseline;
this re-tests that prior under the new full-data split + H10 stack.

Kill criterion (family-level): if neither a23 nor a24 val_qwk > 0.80
AND test_qwk > 0.95 at seed=2, abandon H13.
"""
from .a17_coverage_length_norm import Model as _Model

Model = _Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=256,
    dropout=0.5,
    tau_init=1.0,
    beta_init=1.0,
    alpha_init=1e-3,
    c0_init=6.324555,
    use_coverage=True,
)

