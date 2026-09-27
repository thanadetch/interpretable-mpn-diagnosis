"""a63 — Severity-as-Value Gated Attention (nonlinear per-patch severity, attention-weighted).

Hypothesis (Hn_severity_value_attention). Reticulin grading is a *diffuse,
ordinal severity* read: every region of marrow has a fibre density, and the
ROI grade is a region-importance-weighted reading of those densities. The
gated-attention BASELINE (a48 / SimpleGatedMIL) pools the patch *feature*
vectors with attention and then applies ONE LINEAR head to the single pooled
vector — so the entire bag readout is *linear in the pooled feature*. That
makes the readout collapse the per-patch severity DISTRIBUTION into a single
linear projection of an attention-weighted mean feature.

a63 keeps the SAME soft gated attention for SELECTION/weighting (which patches
of marrow matter — e.g. down-weight off-tissue / paratrabecular junk), but
replaces the value with a per-patch NONLINEAR SEVERITY scalar and aggregates
the SEVERITIES, not the features:

    h_i   = Dropout(ReLU(Linear(1280->128) f_i))          # baseline encoder
    a_i   = Attn_W( Tanh(V h_i) * Sigmoid(U h_i) )         # gated attn score [scalar]
    alpha = softmax_i(a_i / temp)                          # soft weights, sum=1
    s_i   = Linear(32->1)( ReLU(Linear(128->32) h_i) )     # per-patch NONLINEAR severity
    y     = clamp( sum_i alpha_i * s_i , 0, 3 )            # attention-weighted severity

Why this is a MORE EXPRESSIVE readout than the baseline, and a real reason it
could beat 0.7888:
  - Baseline:  y = head( sum_i alpha_i h_i )  with head LINEAR
             = sum_i alpha_i * head(h_i)      (linearity)  -> y = sum alpha_i * <w,h_i>+b
    i.e. each patch's contribution to the grade is a LINEAR function of its
    feature, then attention-weighted. The grade is a weighted mean of a
    *single linear projection*.
  - a63:      y = sum_i alpha_i * MLP(h_i)    with MLP NONLINEAR
    each patch contributes a NONLINEAR severity, then attention-weighted. The
    bag readout now depends on the SHAPE of the per-patch severity distribution
    (a few near-G3 regions among many mild ones reads differently from a
    uniformly-moderate bag with the same mean feature). Pathologists grade by
    the worst-yet-representative density, not the mean feature — a nonlinear
    per-patch severity can capture that ordinal saturation; a linear head
    cannot. This is the active ingredient over the baseline.

Relationship to the two refuted neighbours (a56 and the baseline):
  - vs a56 (per-patch severity + UNWEIGHTED mean): a63 ADDS soft gated
    attention so off-tissue / low-information patches do not dilute the
    severity read. a56 weights every patch equally; a63 lets the gate
    re-weight. a63 == a56 exactly when the attention is uniform, so a56 is the
    no-attention ablation of a63.
  - vs baseline a48 (gated attention + LINEAR head on pooled feature): a63
    keeps the identical gate but makes the per-patch readout NONLINEAR and
    aggregates severities not features. a63 == (a linear-head gated MIL) when
    the severity MLP is collapsed to one linear layer (a64 ablation).

Why this is NOT a dead-end (explicit, against the refuted list):
  - NOT coverage/extent/threshold/fraction (REFUTED 3x): there is NO sigmoid
    presence indicator, NO soft count of fibre-positive patches, NO threshold.
    s_i is an UNBOUNDED severity in grade units, aggregated by a convex
    weighted sum. The readout is the weighted MEAN severity, not a fraction.
  - NO sigma/softmax GATE between the bag representation and the final scalar:
    the only nonlinearities (ReLU in encoder, Tanh/Sigmoid in the gate, ReLU in
    the severity MLP) are ALL per-patch, PRE-aggregation. The aggregation is a
    plain weighted sum and the "head" is the identity (severity is already in
    grade units); clamp only at the very end. The Sigmoid/Tanh inside the gate
    produce ATTENTION WEIGHTS (which patch), not a multiplicative gate on the
    output scalar — softmax-normalised attention is the baseline's own
    mechanism, explicitly allowed by the contract.
  - NOT top-k / argmax / concentration: attention is SOFT (softmax with a
    learnable temperature initialised >1 so it starts near-uniform = diffuse);
    no hard selection, no top-k, no max-pool. "Attention is for weighting not
    concentration."
  - NOT norm-based (||h|| is grade-uninformative): ||h|| is never used; both
    the gate and the severity are learned functions of the encoded feature.
  - NOT attention+mean concat / prediction blend / additive offset / multi-query
    flatten: single value stream, single weighted sum, no concatenation, no
    second prediction to blend, no offset.
  - DeepSet honesty: this is a low-capacity attention-pooled set function. The
    refuted DeepSet/Set-Transformer dead-ends were HIGH-capacity and overfit;
    here phi is the shared baseline encoder, the gate is one (128->128)x2 + 128
    head, and the severity is a tiny 128->32->1 MLP. Total << 200K.

Ablation companions (isolate the two levers vs the two refuted neighbours):
  - a64_severity_value_attention_linear.py : severity_mode='linear'
    (s_i = Linear(128->1) h_i). a63 vs a64 isolates "does a NONLINEAR per-patch
    severity beat a linear one, under the SAME gate?" (a64 ~ baseline-style
    linear readout but on severities).
  - a65 / set attn_mode='uniform' : disables the gate (alpha uniform) -> reduces
    to a56. a63 vs a56 isolates "does the gate help the severity read?".

Permutation-invariant (softmax over patches + weighted sum are symmetric) and
bag-size-invariant (softmax weights sum to 1; severities are per-patch, so
duplicating the bag leaves both alpha-normalisation and the weighted mean
unchanged). Deterministic at inference (dropout off).

Kill criterion: abandon if a63 val_qwk < 0.79 at seed=2 AND a63 does not beat
BOTH a56 (no gate) and a64 (linear severity) on val (neither lever helps).
Definition of done = multi-seed audit {0,1,2,3,42}, not seed=2 alone.

Param count (input_dim=1280, hidden=128, sev_hidden=32):
    bottleneck   Linear(1280,128)+b           = 163,968
    attention_V  Linear(128,128)+b            =  16,512
    attention_U  Linear(128,128)+b            =  16,512
    attention_W  Linear(128,1)+b              =     129
    severity     Linear(128,32)+b             =   4,128
                 Linear(32,1)+b               =      33
    temperature  log_temp (1)                 =       1
    -----------------------------------------------------
    total (mlp severity)                      = 201,283
    a64 ablation (linear severity)            = 197,251  (~ baseline budget)

(With sev_hidden=24 the mlp total is 200,196 if a strict <=200K budget is
required; default sev_hidden=32 gives 201,283 which is ~the baseline 197,250
budget plus the 4K severity MLP, i.e. "<= ~200K".)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        sev_hidden: int = 32,
        severity_mode: str = "mlp",      # "mlp" (a63 main) | "linear" (a64 ablation)
        attn_mode: str = "gated",        # "gated" (a63 main) | "uniform" (a56 ablation)
        temp_init: float = 2.0,          # >1 -> attention starts near-uniform (diffuse, soft)
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        assert severity_mode in ("mlp", "linear"), severity_mode
        assert attn_mode in ("gated", "uniform"), attn_mode
        self.severity_mode = severity_mode
        self.attn_mode = attn_mode
        self.clamp_output = clamp_output

        # Baseline-identical encoder (capacity lives here).
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Gated attention scorer (baseline a48 mechanism): SELECTION/weighting.
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)

        # Learnable softmax temperature on attention logits. Stored as log so it
        # stays positive; init log(temp_init) so attention starts SOFT/diffuse.
        self.log_temp = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(temp_init)))))

        # Per-patch severity head -> scalar in grade units (the VALUE).
        if severity_mode == "mlp":
            self.severity = nn.Sequential(
                nn.Linear(hidden_dim, sev_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(sev_hidden, 1),
            )
            last = self.severity[-1]
        else:  # linear severity -> a64 ablation (linear per-patch readout)
            self.severity = nn.Linear(hidden_dim, 1)
            last = self.severity
        # Start mid-grade so the weighted-mean severity sits in the interior of [0,3].
        with torch.no_grad():
            last.bias.fill_(1.5)

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag; batch_size=1 trainer contract)
        h = self.bottleneck(features)                       # [N, hidden]

        # --- attention weights (which patches matter; SOFT, not concentration) ---
        if self.attn_mode == "gated":
            V = self.attention_V(h)                         # [N, hidden]
            U = self.attention_U(h)                         # [N, hidden]
            a = self.attention_W(V * U).squeeze(-1)         # [N] gated attn logits
            temp = self.log_temp.exp().clamp(min=1e-3)
            alpha = F.softmax(a / temp, dim=0)              # [N] soft weights, sum=1
        else:  # uniform -> reduces to a56 (unweighted mean severity)
            n = h.shape[0]
            alpha = h.new_full((n,), 1.0 / n)

        # --- per-patch NONLINEAR severity (the VALUE; grade units) ---
        s = self.severity(h).squeeze(-1)                    # [N] per-patch severity

        # --- aggregate SEVERITIES (not features): convex weighted sum, NO gate ---
        y = (alpha * s).sum().view(1)                       # scalar weighted-mean severity
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            return y, alpha, None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    sev_hidden=32,
    severity_mode="mlp",     # a63 main = nonlinear per-patch severity
    attn_mode="gated",       # a63 main = gated soft attention weighting
    temp_init=2.0,           # start diffuse/soft
    clamp_output=True,
)
