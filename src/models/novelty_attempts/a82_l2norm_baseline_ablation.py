"""a82 — ablation companion to a81 (NO L2-normalization = plain baseline).

Imports a81's Model and flips the SINGLE documented active-ingredient flag
l2_normalize=True -> False. With normalization OFF the model operates on the
RAW patch features f_i, so it is EXACTLY the baseline Ilse gated-attention MIL:

    h_i = Dropout(ReLU(Linear(1280->128) f_i))           # raw features
    a_i = softmax_i( W ( tanh(V h_i) * sigmoid(U h_i) ) )
    z   = sum_i a_i h_i
    y   = Linear(128->1) z                                # RAW logit

a81 (l2_normalize=True) vs a82 (l2_normalize=False) isolates EXACTLY the active
ingredient: "does explicitly removing the grade-irrelevant feature MAGNITUDE
(L2-normalize patches before aggregation) help / stabilise vs the identical
baseline run on raw features?". Everything else — architecture, capacity
(197,250 params), init, dropout, attention, head — is bit-for-bit identical.
"""
from .a81_l2norm_coverage import Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    l2_normalize=False,     # ablation: NO L2-norm => plain baseline on raw features
    eps=1e-6,
)
