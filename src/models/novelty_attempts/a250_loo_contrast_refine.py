"""a250 - LOO-contrast attention refinement (closed-form diversity reweighting, NO DPP / NO solve).

MECHANISM
---------
Backbone = the working gated-attention pool, KEPT. We add exactly ONE refinement step that re-weights
each patch by how DISTINCTIVE it is relative to the bag context computed WITHOUT its own contribution --
a closed-form leave-one-out (LOO) redundancy/diversity reweighting that is matmul/algebra-only:

  First pass (standard gated pool):
    h   = bottleneck(features)                       # [N,H]
    e0  = attention_W(tanh(attention_V(h)) * sigmoid(attention_U(h))).squeeze(-1)   # [N] gated energies
    a0  = softmax(e0)                                # [N]
    c   = h.t() @ a0                                 # [H] first-pass pooled context

  LOO context per patch, in closed form (no Gram, no inverse, no DPP solve):
    c_loo_i = (c - a0_i h_i) / (1 - a0_i)            # [N,H]  guarded by denom=clamp(1-a0,eps)
    r_i     = h_i - c_loo_i                          # [N,H]  contrast vs the REST of the salient bag

  Learned additive energy correction from the contrast, then a SINGLE refinement of the SAME pool:
    delta_e = contrast_head(ReLU(R(r))).squeeze(-1)  # R=Linear(H,H), contrast_head=Linear(H,1); [N]
    e1      = e0 + delta_e                            # ADDITIVE residual: delta_e=0 -> baseline pool
    a1      = softmax(e1)                             # [N] refined weights, sum=1 -> KEEP THE POOL
    z       = h.t() @ a1                              # [H] refined attention-weighted pool
    y       = classifier(z).view(-1)                  # shape (1,)

WHY-NEW (vs the directive's exhausted list and near-neighbours)
---------------------------------------------------------------
- Fills the explicitly-open lane: a NEW closed-form redundancy/diversity reweighting of attention that is
  NOT DPP and NOT a linear solve. The LOO context is pure subtract-and-renormalise algebra + matmuls --
  it NEVER forms or inverts a Gram/kernel matrix, so it is MPS-safe where a244's true DPP marginal needed
  structured linear algebra (solve/eigh, unsupported on MPS).
- Distinct from a244 (pre-softmax softplus quality + q/(1+r) cosine-similarity-sum repulsion; no LOO,
  multiplicative not additive-residual).
- Distinct from a221 (operates on the pooled VECTOR via soft-threshold shrinkage; a250 operates on the
  attention WEIGHTS).
- Distinct from a217 (cosine-to-pooled-vector query + lone log_s scalar, mean-pool init that ABANDONS the
  gated scorer; a250 keeps the gated scorer and adds an additive residual on top).
- Distinct from null-patch / dropout: deterministic, no extra slot, no stochasticity.
- Capacity lives in linear maps R and the readout head (NO lone learnable shape-scalar); additive residual
  keeps the gated pool as the EXACTLY recoverable backbone (delta_e=0); exactly one refinement step
  conditioned on the first-pass LOO context.
- Effect: down-weights redundant near-duplicate clumps (uni2's over-concentration failure mode) or
  up-weights a coherent minority -> lowers read-variance WITHOUT a global sharpness/temperature knob.

CONTRACT
--------
Self-contained nn.Module; torch / nn / F only. forward(features[N,D], return_attention=False, metrics=None)
-> (y, attn_or_None, None) with y shape (1,). Permutation- & size-invariant, deterministic in eval(),
n=1 safe (a0=1 -> denom=clamp(1-a0,1e-4)=eps; contrast finite). MPS-safe ops only: matmul, elementwise,
clamp, relu, tanh, sigmoid, softmax.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        # gated attention scorer (baseline-style two-branch gate)
        self.attention_V = nn.Linear(hidden_dim, hidden_dim)
        self.attention_U = nn.Linear(hidden_dim, hidden_dim)
        self.attention_W = nn.Linear(hidden_dim, 1)
        # LOO-contrast readout -> additive energy correction (capacity lives here, no lone scalar)
        self.R = nn.Linear(hidden_dim, hidden_dim)
        self.contrast_head = nn.Linear(hidden_dim, 1)
        # zero-init the additive residual so training STARTS from the exact baseline gated pool
        nn.init.zeros_(self.contrast_head.weight)
        nn.init.zeros_(self.contrast_head.bias)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                                  # [N,H]
        # first-pass gated energies + standard attention pool
        e0 = self.attention_W(torch.tanh(self.attention_V(h)) * torch.sigmoid(self.attention_U(h))).squeeze(-1)  # [N]
        a0 = F.softmax(e0, dim=0)                                       # [N]
        c = h.t() @ a0                                                 # [H] first-pass pooled context (matmul)
        # closed-form leave-one-out context per patch (no Gram, no inverse, no solve)
        denom = (1.0 - a0).clamp(min=1e-4)                             # [N] n=1 safe (a0=1 -> denom=eps)
        c_loo = (c.unsqueeze(0) - a0.unsqueeze(1) * h) / denom.unsqueeze(1)  # [N,H]
        r = h - c_loo                                                  # [N,H] contrast vs rest-of-bag
        # learned additive energy correction; zero-init -> recovers baseline exactly at start
        delta_e = self.contrast_head(F.relu(self.R(r))).squeeze(-1)    # [N]
        e1 = e0 + delta_e                                              # additive residual
        a1 = F.softmax(e1, dim=0)                                      # [N] refined weights, sum=1
        z = h.t() @ a1                                                 # [H] refined attention-weighted pool
        y = self.classifier(z).view(-1)                               # shape (1,)
        if return_attention:
            return y, a1, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
