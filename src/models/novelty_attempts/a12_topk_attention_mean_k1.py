"""a12 — top-K attention-weighted mean, K=1 (H7 ablation).

Ablation companion to a11_topk_attention_mean.py. Identical wiring
except K=1, i.e. the bag representation collapses to a single patch:
the one with the highest gated-attention logit.

Active ingredient under test: "averaging across multiple worst patches"
(K=5 in a11) vs "use only the single worst patch" (K=1 here). If a11
beats a12, mean-over-top-K is the active ingredient; if a12 wins or
both fail, the top-K replacement family is not buying anything beyond
hard max-pooling.

Kill criterion (family-level): if BOTH a11 AND a12 have val_qwk < 0.82
at seed=2, abandon H7 and add the family to §4 DE.

Param count: same as ABMIL = 197,250 (no new params).
"""
from .a11_topk_attention_mean import Model as _Model

Model = _Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    K=1,
)

