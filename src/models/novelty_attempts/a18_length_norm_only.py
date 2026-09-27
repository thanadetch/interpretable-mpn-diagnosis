"""a18 — length-normalised softmax only (H10 ablation, a07 positive control).

Ablation companion to a17_coverage_length_norm.py. Identical wiring but
use_coverage=False, so the coverage-prior branch is dropped and the
model reduces to a07 (length-normalised softmax temperature only:
T = c0 / sqrt(N)).

Active ingredient under test: whether ADDING the per-patch coverage
prior on top of length-normalised attention pushes val and test above
a07 alone.

Positive control: a18 should reproduce a07 val ~ 0.7868 / test ~
0.9596. If a18 diverges, there is an architecture-vs-RNG discrepancy
worth investigating before believing a17.

Kill criterion (family-level): if both a17 and a18 val_qwk < 0.78 at
seed=2, abandon H10 and add family to section 4 DE.

Param count: baseline 197,250 + c0_raw = 197,251.
"""
from .a17_coverage_length_norm import Model as _Model

Model = _Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    c0_init=6.324555,
    use_coverage=False,
)

