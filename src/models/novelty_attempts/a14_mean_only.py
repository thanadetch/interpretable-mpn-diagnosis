"""a14 — mean-only pool (H8 ablation, no attention branch).

Ablation companion to a13_attn_mean_concat.py. Identical bottleneck
but the attention branch is gone -- bag = mean(h), Linear(128, 1) head.

Active ingredient under test: the COMBINATION of attention + mean pool
(a13) vs mean pool alone (a14). If a13 beats a14, the attention branch
is contributing useful signal on top of the mean anchor. If a14 wins
or both fail to beat baseline (0.8182), the concat ensemble adds
nothing of value.

This is essentially the legacy `mean_pool + bottleneck` setup; we
expect it to underperform baseline.

Kill criterion (family-level): if BOTH a13 AND a14 have val_qwk < 0.82
at seed=2, abandon H8 and add the family to §4 DE.
"""
from .a13_attn_mean_concat import Model as _Model

Model = _Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    use_attention=False,
)

