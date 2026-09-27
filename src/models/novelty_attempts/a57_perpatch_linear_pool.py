"""a57 — Per-patch LINEAR severity pooling (ablation companion to a56).

Identical to a56 but severity_mode='linear': the per-patch severity is a
single Linear(128->1) instead of a 2-layer MLP. Then

    y = clamp(mean_i <h_i, w> + b) = clamp(<mean_i h_i, w> + b)

which is exactly mean-pool of the bottleneck features + a linear head. This
removes EXACTLY the active ingredient of a56 — the NONLINEAR per-patch
severity map (the only thing that makes mean_i phi(h_i) differ from
phi(mean_i h_i)).

a56 (mlp) vs a57 (linear) isolates: does a distribution-aware NONLINEAR
per-patch severity beat plain mean-pooling on val? (If not, a56 collapses to
mean-pool — the recurring outcome this session, recorded honestly.)
"""
from .a56_perpatch_severity_pool import Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    severity_mode="linear",   # ablation = mean-pool + linear head
    sev_hidden=32,
    clamp_output=True,
)
