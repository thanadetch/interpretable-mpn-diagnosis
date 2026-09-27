"""a86 — Ablation of a85: lambda FIXED = 1 (= standard gated attention).

Identical to a85 (same encoder, same gated-attention scorer, same RAW-logit head)
but the diffuseness blend is frozen: lambda_fixed=1.0. With lambda == 1 the pooling
weights become

    w_i = (1 - 1) * (1/N) + 1 * softmax(scores)_i = softmax(scores)_i

i.e. the UNIFORM (mean-pool) anchor is removed entirely and aggregation reduces to
plain gated-attention-weighted mean. Freezing lambda also drops the lambda_logit
parameter from training (registered as a constant buffer instead), so this ablation
removes EXACTLY the active ingredient of a85: the explicit, learnable diffuseness
prior on the pooling weights (start diffuse, sharpen only if needed).

a85 (uniform-anchored, lambda init 0.1, learnable) vs a86 (lambda fixed 1.0,
plain gated attention) isolates: "does an explicit diffuseness prior on the
pooling weights help / stabilise grading on this 214-ROI cohort, versus standard
gated attention?".

Param count: a85 has 197,251 (baseline 197,250 + 1 lambda_logit); a86 freezes
lambda into a buffer => 197,250 (= the baseline gated-attention count).
"""
from .a85_diffuse_attention import Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    lambda_init=0.1,        # ignored when lambda_fixed is set
    lambda_fixed=1.0,       # ablation: freeze blend at 1 => standard gated attention
    axis_weight=0.0,        # match a85 default (purely learned scorer)
    prototype_path=None,
)
